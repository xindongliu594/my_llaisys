import unittest

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
