import unittest
import time

import llaisys


class FakeModel:
    def __init__(self):
        self.contexts = []

    def generate(self, inputs, max_new_tokens=None, top_k=1, **_):
        assert top_k == 1
        context = list(inputs)
        self.contexts.append(context)
        generated = [9000 + len(context) + i for i in range(max_new_tokens)]
        return context + generated


class FailingModel:
    def generate(self, inputs, **_):
        raise RuntimeError("synthetic backend failure")


class StepModel:
    eos_token_id = 99

    def __init__(self):
        self.calls = []

    def generate(self, inputs, max_new_tokens=None, top_k=1, **_):
        assert max_new_tokens == 1
        assert top_k == 1
        context = list(inputs)
        self.calls.append(context)
        first = context[0]
        if first == 3:
            token = self.eos_token_id
        elif first == 4:
            token = 77
        else:
            token = first * 10 + len(context)
        return context + [token]


class ServingTest(unittest.TestCase):
    def test_priority_fifo_and_independent_sessions(self):
        model = FakeModel()
        scheduler = llaisys.RequestScheduler(model)
        scheduler.sessions.create(
            "conversation-a", user_id="user-1", initial_tokens=[10]
        )
        scheduler.sessions.create(
            "conversation-b", user_id="user-1", initial_tokens=[20]
        )
        scheduler.sessions.create("other-user", user_id="user-2")

        self.assertEqual(
            [session.session_id for session in scheduler.sessions.list("user-1")],
            ["conversation-a", "conversation-b"],
        )

        request_a = scheduler.submit(
            "conversation-a", [11], max_new_tokens=2, priority=0
        )
        request_b = scheduler.submit(
            "conversation-b", [21], max_new_tokens=1, priority=5
        )

        completed = scheduler.run_until_idle(raise_on_error=True)
        self.assertEqual(
            [request.request_id for request in completed],
            [request_b.request_id, request_a.request_id],
        )
        self.assertEqual(model.contexts, [[20, 21], [10, 11]])
        self.assertEqual(request_a.generated_tokens, (9002, 9003))
        self.assertEqual(request_b.generated_tokens, (9002,))
        self.assertEqual(
            scheduler.sessions.get("conversation-a").token_history,
            [10, 11, 9002, 9003],
        )
        self.assertEqual(
            scheduler.sessions.get("conversation-b").token_history,
            [20, 21, 9002],
        )
        self.assertEqual(scheduler.sessions.get("other-user").token_history, [])

    def test_same_session_has_only_one_in_flight_turn(self):
        scheduler = llaisys.RequestScheduler(FakeModel())
        scheduler.sessions.create("conversation")
        scheduler.submit("conversation", [1])

        with self.assertRaisesRegex(RuntimeError, "pending request"):
            scheduler.submit("conversation", [2])

    def test_cancel_releases_session(self):
        scheduler = llaisys.RequestScheduler(FakeModel())
        scheduler.sessions.create("conversation")
        request = scheduler.submit("conversation", [1])

        self.assertTrue(scheduler.cancel(request.request_id))
        self.assertEqual(request.status, llaisys.RequestStatus.CANCELLED)
        replacement = scheduler.submit("conversation", [2])
        self.assertEqual(replacement.status, llaisys.RequestStatus.WAITING)

    def test_failure_does_not_commit_partial_history(self):
        scheduler = llaisys.RequestScheduler(FailingModel())
        scheduler.sessions.create("conversation", initial_tokens=[1, 2])
        request = scheduler.submit("conversation", [3])

        completed = scheduler.run_once()
        self.assertIs(completed, request)
        self.assertEqual(request.status, llaisys.RequestStatus.FAILED)
        self.assertIn("synthetic backend failure", request.error)
        self.assertEqual(
            scheduler.sessions.get("conversation").token_history, [1, 2]
        )


class RoundRobinServingTest(unittest.TestCase):
    def test_priority_admission_then_round_robin_stream(self):
        model = StepModel()
        scheduler = llaisys.RoundRobinScheduler(model)
        scheduler.sessions.create("a", initial_tokens=[1])
        scheduler.sessions.create("b", initial_tokens=[2])
        request_a = scheduler.submit("a", [], max_new_tokens=2, priority=0)
        request_b = scheduler.submit("b", [], max_new_tokens=2, priority=5)

        events = list(scheduler.run_until_idle_stream())

        self.assertEqual(
            [event.request_id for event in events],
            [
                request_b.request_id,
                request_a.request_id,
                request_b.request_id,
                request_a.request_id,
            ],
        )
        self.assertEqual([event.token_id for event in events], [21, 11, 22, 12])
        self.assertTrue(events[-1].finished)
        self.assertEqual(request_a.finish_reason, llaisys.FinishReason.LENGTH)
        self.assertEqual(request_b.finish_reason, llaisys.FinishReason.LENGTH)
        self.assertEqual(request_a.generated_tokens, (11, 12))
        self.assertEqual(request_b.generated_tokens, (21, 22))
        self.assertEqual(scheduler.sessions.get("a").token_history, [1, 11, 12])
        self.assertEqual(scheduler.sessions.get("b").token_history, [2, 21, 22])

    def test_eos_and_custom_stop(self):
        scheduler = llaisys.RoundRobinScheduler(StepModel())
        scheduler.sessions.create("eos", initial_tokens=[3])
        scheduler.sessions.create("stop", initial_tokens=[4])
        eos_request = scheduler.submit("eos", [], max_new_tokens=5)
        stop_request = scheduler.submit(
            "stop", [], max_new_tokens=5, stop_token_ids=[77]
        )

        events = list(scheduler.run_until_idle_stream())

        self.assertEqual(len(events), 2)
        self.assertEqual(eos_request.finish_reason, llaisys.FinishReason.EOS)
        self.assertEqual(stop_request.finish_reason, llaisys.FinishReason.STOP)
        self.assertTrue(all(event.finished for event in events))

    def test_running_cancel_commits_streamed_prefix(self):
        scheduler = llaisys.RoundRobinScheduler(StepModel())
        scheduler.sessions.create("conversation", initial_tokens=[1])
        request = scheduler.submit("conversation", [], max_new_tokens=5)

        first = scheduler.step()
        self.assertEqual(first.token_id, 11)
        self.assertTrue(scheduler.cancel(request.request_id))
        terminal = scheduler.step()

        self.assertTrue(terminal.finished)
        self.assertEqual(terminal.finish_reason, llaisys.FinishReason.CANCELLED)
        self.assertEqual(request.status, llaisys.RequestStatus.CANCELLED)
        self.assertEqual(
            scheduler.sessions.get("conversation").token_history, [1, 11]
        )

    def test_waiting_cancel_releases_session_and_streams_terminal_event(self):
        scheduler = llaisys.RoundRobinScheduler(StepModel())
        scheduler.sessions.create("conversation", initial_tokens=[1])
        request = scheduler.submit("conversation", [], max_new_tokens=5)

        self.assertTrue(scheduler.cancel(request.request_id))
        events = list(scheduler.events(request.request_id, timeout=0.1))

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].finished)
        self.assertEqual(events[0].finish_reason, llaisys.FinishReason.CANCELLED)
        replacement = scheduler.submit("conversation", [], max_new_tokens=1)
        self.assertEqual(replacement.status, llaisys.RequestStatus.WAITING)

    def test_timeout_and_error_do_not_corrupt_history(self):
        timeout_scheduler = llaisys.RoundRobinScheduler(StepModel())
        timeout_scheduler.sessions.create("timeout", initial_tokens=[1])
        timeout_request = timeout_scheduler.submit(
            "timeout", [], max_new_tokens=5, timeout_seconds=0.001
        )
        time.sleep(0.01)
        timeout_event = timeout_scheduler.step()

        self.assertEqual(timeout_event.finish_reason, llaisys.FinishReason.TIMEOUT)
        self.assertEqual(timeout_request.status, llaisys.RequestStatus.CANCELLED)
        self.assertEqual(
            timeout_scheduler.sessions.get("timeout").token_history, [1]
        )

        error_scheduler = llaisys.RoundRobinScheduler(FailingModel())
        error_scheduler.sessions.create("error", initial_tokens=[5])
        error_request = error_scheduler.submit("error", [], max_new_tokens=5)
        error_event = error_scheduler.step()

        self.assertEqual(error_event.finish_reason, llaisys.FinishReason.ERROR)
        self.assertEqual(error_request.status, llaisys.RequestStatus.FAILED)
        self.assertIn("synthetic backend failure", error_event.error)
        self.assertEqual(error_scheduler.sessions.get("error").token_history, [5])

    def test_background_worker_event_stream(self):
        scheduler = llaisys.RoundRobinScheduler(
            StepModel(), token_decoder=lambda token: f"<{token}>"
        )
        scheduler.sessions.create("conversation", initial_tokens=[1])
        scheduler.start()
        try:
            request = scheduler.submit("conversation", [], max_new_tokens=2)
            events = list(scheduler.events(request.request_id, timeout=2.0))
        finally:
            scheduler.stop()

        self.assertEqual([event.token_id for event in events], [11, 12])
        self.assertEqual([event.text for event in events], ["<11>", "<12>"])
        self.assertTrue(events[-1].finished)


if __name__ == "__main__":
    unittest.main(verbosity=2)
