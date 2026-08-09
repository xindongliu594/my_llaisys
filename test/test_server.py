import json
import time
import unittest
import urllib.error
import urllib.request

import llaisys


class ServerModel:
    eos_token_id = 99

    def generate(self, inputs, max_new_tokens=None, **_):
        assert max_new_tokens == 1
        context = list(inputs)
        token = 65 if len(context) == 1 else self.eos_token_id
        return context + [token]


class ServerTokenizer:
    def apply_chat_template(
        self, conversation, add_generation_prompt=True, tokenize=False
    ):
        assert add_generation_prompt
        assert not tokenize
        return "|".join(
            f"{message['role']}:{message['content']}" for message in conversation
        )

    def encode(self, text):
        assert text
        return [1]

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token in token_ids:
            if token == 65:
                pieces.append("A")
            elif token == 99 and not skip_special_tokens:
                pieces.append("<eos>")
        return "".join(pieces)


class HTTPServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scheduler = llaisys.RoundRobinScheduler(ServerModel())
        chat = llaisys.ChatService(scheduler, ServerTokenizer())
        cls.server = llaisys.OpenAIAPIServer(
            chat, model_id="test-model", host="127.0.0.1", port=0
        )
        cls.server.start()
        host, port = cls.server.address
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def request(self, method, path, data=None):
        payload = None
        headers = {}
        if data is not None:
            payload = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=payload,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers, response.read()

    def test_health_models_and_chat_completion(self):
        status, _, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

        status, _, body = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "test-model")

        status, _, body = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 2,
            },
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["choices"][0]["message"]["content"], "A")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["completion_tokens"], 2)

    def test_text_completion_and_sse(self):
        status, _, body = self.request(
            "POST",
            "/v1/completions",
            {"model": "test-model", "prompt": "Hello", "max_tokens": 2},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["text"], "A")

        status, headers, body = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Stream"}],
                "max_tokens": 2,
                "stream": True,
            },
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        text = body.decode("utf-8")
        self.assertIn("chat.completion.chunk", text)
        self.assertIn("data: [DONE]", text)

    def test_session_crud_request_status_and_metrics(self):
        status, _, body = self.request(
            "POST",
            "/sessions",
            {
                "session_id": "http-session",
                "user_id": "user-1",
                "system_prompt": "Be concise",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(json.loads(body)["session_id"], "http-session")

        status, _, body = self.request("GET", "/sessions/http-session")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["user_id"], "user-1")

        status, _, body = self.request(
            "POST",
            "/sessions/http-session/messages",
            {"content": "Hello", "max_tokens": 1},
        )
        self.assertEqual(status, 202)
        request_id = json.loads(body)["request_id"]

        status, _, body = self.request("GET", f"/requests/{request_id}")
        self.assertEqual(status, 200)
        self.assertIn(json.loads(body)["status"], {"waiting", "running", "finished"})

        status, headers, body = self.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn("text/plain", headers["Content-Type"])
        self.assertIn("llaisys_requests_total", body.decode("utf-8"))

        # Wait for the short asynchronous request before deleting its session.
        terminal = None
        for _ in range(100):
            _, _, request_body = self.request("GET", f"/requests/{request_id}")
            terminal = json.loads(request_body)["status"]
            if terminal not in {"waiting", "running"}:
                break
            time.sleep(0.01)
        self.assertEqual(terminal, "finished")

        messages = []
        for _ in range(100):
            _, _, session_body = self.request("GET", "/sessions/http-session")
            messages = json.loads(session_body)["messages"]
            if messages and messages[-1]["role"] == "assistant":
                break
            time.sleep(0.01)
        self.assertEqual(messages[-1], {"role": "assistant", "content": "A"})

        _, _, metrics_body = self.request("GET", "/metrics")
        self.assertIn("llaisys_requests_active 0", metrics_body.decode("utf-8"))

        status, _, body = self.request("DELETE", "/sessions/http-session")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["deleted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
