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


class LimitedModel(FakeModel):
    max_sequence_length = 4


class SplitStopModel:
    def generate(self, inputs, max_new_tokens=None, **_):
        assert max_new_tokens == 1
        context = list(inputs)
        generated = (10, 11, 12)
        return context + [generated[len(context) - 1]]


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


class BatchedStepModel(StepModel):
    supports_sequence_batching = True

    def __init__(self):
        super().__init__()
        self.sequences = set()
        self.batch_sizes = []

    def create_sequence(self, sequence_id, capacity):
        assert capacity > 0
        self.sequences.add(sequence_id)

    def destroy_sequence(self, sequence_id):
        self.sequences.discard(sequence_id)

    def prefill_sequence(self, sequence_id, input_tokens, **_):
        assert sequence_id in self.sequences
        return int(input_tokens[0]) * 10 + len(input_tokens)

    def decode_batch(self, sequence_ids, token_ids, sampling_configs):
        assert len(sequence_ids) == len(token_ids) == len(sampling_configs)
        assert all(sequence_id in self.sequences for sequence_id in sequence_ids)
        self.batch_sizes.append(len(sequence_ids))
        return [int(token) + 1 for token in token_ids]


class SlowBatchedStepModel(BatchedStepModel):
    def __init__(self, prefill_delay=0.0, decode_delay=0.0):
        super().__init__()
        self.prefill_delay = prefill_delay
        self.decode_delay = decode_delay

    def prefill_sequence(self, sequence_id, input_tokens, **kwargs):
        time.sleep(self.prefill_delay)
        return super().prefill_sequence(sequence_id, input_tokens, **kwargs)

    def decode_batch(self, sequence_ids, token_ids, sampling_configs):
        time.sleep(self.decode_delay)
        return super().decode_batch(
            sequence_ids, token_ids, sampling_configs
        )


class FakeTokenizer:
    def __init__(self):
        self.conversations = []

    def apply_chat_template(
        self, conversation, add_generation_prompt=True, tokenize=False
    ):
        assert add_generation_prompt
        assert not tokenize
        copied = [dict(message) for message in conversation]
        self.conversations.append(copied)
        return "|".join(
            f"{message['role']}:{message['content']}" for message in copied
        ) + "|assistant:"

    def encode(self, text):
        assert text
        return [1]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token) for token in token_ids)


class ServingTest(unittest.TestCase):
    def test_request_pool_capacity_releases_rejected_session(self):
        pool = llaisys.RequestPool(max_pending_requests=1)
        scheduler = llaisys.RequestScheduler(FakeModel(), request_pool=pool)
        scheduler.sessions.create("first")
        scheduler.sessions.create("rejected")
        scheduler.submit("first", [1])

        with self.assertRaises(llaisys.ServiceOverloadedError):
            scheduler.submit("rejected", [2])

        self.assertFalse(scheduler.sessions.get("rejected").busy)

    def test_context_limit_rejects_or_left_truncates(self):
        model = LimitedModel()
        scheduler = llaisys.RequestScheduler(model)
        scheduler.sessions.create("limited", initial_tokens=[1, 2])

        with self.assertRaisesRegex(ValueError, "Prompt has 3 tokens"):
            scheduler.submit("limited", [3], max_new_tokens=2)

        request = scheduler.submit(
            "limited", [3], max_new_tokens=2, truncate_prompt=True
        )
        scheduler.run_until_idle(raise_on_error=True)

        self.assertEqual(model.contexts, [[2, 3]])
        self.assertEqual(request.context_length, 2)

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
    def test_active_request_limit_keeps_excess_work_waiting(self):
        scheduler = llaisys.RoundRobinScheduler(
            StepModel(), max_active_requests=1
        )
        scheduler.sessions.create("active", initial_tokens=[1])
        scheduler.sessions.create("queued", initial_tokens=[2])
        active = scheduler.submit("active", [], max_new_tokens=2)
        queued = scheduler.submit("queued", [], max_new_tokens=1)

        scheduler.step()

        self.assertEqual(active.status, llaisys.RequestStatus.RUNNING)
        self.assertEqual(queued.status, llaisys.RequestStatus.WAITING)
        list(scheduler.run_until_idle_stream())
        self.assertEqual(queued.status, llaisys.RequestStatus.FINISHED)

    def test_stop_string_can_span_tokens_without_leaking_prefix(self):
        pieces = {10: "answer<", 11: "END>", 12: "ignored"}
        scheduler = llaisys.RoundRobinScheduler(
            SplitStopModel(),
            sequence_decoder=lambda tokens: "".join(pieces[token] for token in tokens),
        )
        scheduler.sessions.create("stop-text", initial_tokens=[1])
        request = scheduler.submit(
            "stop-text", [], max_new_tokens=3, stop_strings=["<END>"]
        )

        events = list(scheduler.run_until_idle_stream())

        self.assertEqual([event.text for event in events], ["answer", None])
        self.assertEqual(request.finish_reason, llaisys.FinishReason.STOP)
        self.assertEqual(request.streamed_text, "answer")
        self.assertEqual(request.generated_tokens, (10, 11))


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


class OrcaServingTest(unittest.TestCase):
    def test_queue_prefill_and_decode_timeouts_release_sequences(self):
        queue_model = SlowBatchedStepModel()
        queue_scheduler = llaisys.OrcaScheduler(
            queue_model, max_active_requests=1
        )
        queue_scheduler.sessions.create("active")
        queue_scheduler.sessions.create("queued")
        active = queue_scheduler.submit("active", [1], max_new_tokens=3)
        queued = queue_scheduler.submit(
            "queued", [2], max_new_tokens=2, queue_timeout_seconds=0.001
        )
        queue_scheduler.step_batch()
        time.sleep(0.005)
        queue_scheduler.step_batch()
        self.assertEqual(queued.finish_reason, llaisys.FinishReason.TIMEOUT)
        self.assertEqual(queued.timeout_phase, "queue")
        self.assertEqual(queued.phase, llaisys.RequestPhase.FINISHED)
        self.assertFalse(queue_scheduler.sessions.get("queued").busy)
        queue_scheduler.cancel(active.request_id)
        queue_scheduler.step_batch()

        prefill_model = SlowBatchedStepModel(prefill_delay=0.005)
        prefill_scheduler = llaisys.OrcaScheduler(prefill_model)
        prefill_scheduler.sessions.create("prefill")
        prefill = prefill_scheduler.submit(
            "prefill", [1], prefill_timeout_seconds=0.001
        )
        prefill_scheduler.step_batch()
        self.assertEqual(prefill.timeout_phase, "prefill")
        self.assertEqual(prefill.finish_reason, llaisys.FinishReason.TIMEOUT)
        self.assertNotIn(prefill.request_id, prefill_model.sequences)

        decode_model = SlowBatchedStepModel(decode_delay=0.005)
        decode_scheduler = llaisys.OrcaScheduler(decode_model)
        decode_scheduler.sessions.create("decode")
        decode = decode_scheduler.submit(
            "decode", [1], max_new_tokens=3, decode_timeout_seconds=0.001
        )
        decode_scheduler.step_batch()
        decode_scheduler.step_batch()
        self.assertEqual(decode.timeout_phase, "decode")
        self.assertEqual(decode.finish_reason, llaisys.FinishReason.TIMEOUT)
        self.assertNotIn(decode.request_id, decode_model.sequences)

    def test_dynamic_iteration_batching_and_sequence_cleanup(self):
        model = BatchedStepModel()
        scheduler = llaisys.OrcaScheduler(
            model, max_active_requests=2, max_prefill_per_iteration=1
        )
        scheduler.sessions.create("a", initial_tokens=[1])
        scheduler.sessions.create("b", initial_tokens=[2])
        request_a = scheduler.submit("a", [], max_new_tokens=3)
        request_b = scheduler.submit("b", [], max_new_tokens=3)

        events = list(scheduler.run_until_idle_stream())

        self.assertTrue(scheduler.supports_continuous_batching)
        self.assertEqual(request_a.generated_tokens, (11, 12, 13))
        self.assertEqual(request_b.generated_tokens, (21, 22, 23))
        self.assertEqual(model.batch_sizes, [1, 2, 1])
        self.assertEqual(model.sequences, set())
        self.assertEqual(len(events), 6)
        self.assertEqual(request_a.status, llaisys.RequestStatus.FINISHED)
        self.assertEqual(request_b.status, llaisys.RequestStatus.FINISHED)


class ChatServiceTest(unittest.TestCase):
    def test_structured_messages_multi_turn_and_export_import(self):
        tokenizer = FakeTokenizer()
        scheduler = llaisys.RoundRobinScheduler(StepModel())
        chat = llaisys.ChatService(scheduler, tokenizer)
        chat.create_session(
            "chat-1", user_id="user-1", system_prompt="Be concise"
        )

        first = chat.submit_message("chat-1", "Hello", max_new_tokens=2)
        events = list(chat.run_until_idle_stream())

        self.assertEqual(first.generated_tokens, (11, 12))
        self.assertTrue(events[-1].finished)
        session = scheduler.sessions.get("chat-1")
        self.assertEqual(
            [(message.role, message.content) for message in session.messages],
            [
                ("system", "Be concise"),
                ("user", "Hello"),
                ("assistant", "11 12"),
            ],
        )

        second = chat.submit_message("chat-1", "Continue", max_new_tokens=1)
        list(chat.run_until_idle_stream())
        self.assertEqual(second.generated_tokens, (11,))
        self.assertEqual(
            [message["role"] for message in tokenizer.conversations[-1]],
            ["system", "user", "assistant", "user"],
        )

        exported = chat.export_session("chat-1")
        scheduler.sessions.delete("chat-1")
        imported = chat.import_session(exported, session_id="chat-copy")
        self.assertEqual(imported.user_id, "user-1")
        self.assertEqual(
            [message.as_dict() for message in imported.messages],
            exported["messages"],
        )

    def test_background_chat_stream_finalizes_assistant_message(self):
        tokenizer = FakeTokenizer()
        scheduler = llaisys.RoundRobinScheduler(StepModel())
        chat = llaisys.ChatService(scheduler, tokenizer)
        chat.create_session("chat")
        scheduler.start()
        try:
            request = chat.submit_message("chat", "Hello", max_new_tokens=1)
            events = list(chat.events(request.request_id, timeout=2.0))
        finally:
            scheduler.stop()

        self.assertTrue(events[-1].finished)
        self.assertEqual(
            scheduler.sessions.get("chat").messages[-1],
            llaisys.ChatMessage("assistant", "11"),
        )

    def test_invalid_chat_role(self):
        with self.assertRaisesRegex(ValueError, "Unsupported chat role"):
            llaisys.ChatMessage("tool", "not enabled yet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
