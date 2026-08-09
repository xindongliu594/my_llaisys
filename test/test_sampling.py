import ctypes
import unittest

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.models import LlaisysQwen2Meta, LlaisysSamplingConfig


class SamplingBackendTest(unittest.TestCase):
    def setUp(self):
        self.meta = LlaisysQwen2Meta(
            dtype=int(llaisys.DataType.F32),
            nlayer=0,
            hs=2,
            nh=1,
            nkvh=1,
            dh=2,
            di=2,
            maxseq=16,
            voc=4,
            epsilon=1e-6,
            theta=10000.0,
            end_token=99,
        )
        device_ids = (ctypes.c_int * 1)(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), int(llaisys.DeviceType.CPU), device_ids, 1
        )
        self.assertTrue(self.model)
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model).contents

        # Every token maps to the same hidden vector. Output rows then produce
        # strictly increasing logits 0, sqrt(2), 2*sqrt(2), 3*sqrt(2).
        self._load(weights.in_embed, [1.0, 0.0] * 4)
        self._load(
            weights.out_embed,
            [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
        )
        self._load(weights.out_norm_w, [1.0, 1.0])

    def tearDown(self):
        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)

    @staticmethod
    def _load(handle, values):
        data = (ctypes.c_float * len(values))(*values)
        LIB_LLAISYS.tensorLoad(handle, ctypes.cast(data, ctypes.c_void_p))

    def _sample(self, prompt_token, **overrides):
        values = {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "seed": 1234,
        }
        values.update(overrides)
        config = LlaisysSamplingConfig(**values)
        prompt = ctypes.c_int64(prompt_token)
        LIB_LLAISYS.llaisysQwen2ModelReset(self.model)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelInferSample(
                self.model, ctypes.byref(prompt), 1, ctypes.byref(config)
            )
        )

    def test_greedy_and_small_top_p_choose_maximum(self):
        prompt = ctypes.c_int64(0)
        LIB_LLAISYS.llaisysQwen2ModelReset(self.model)
        greedy = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model, ctypes.byref(prompt), 1
        )
        self.assertEqual(greedy, 3)
        self.assertEqual(self._sample(0, top_p=0.01), 3)

    def test_top_k_limits_candidates_and_seed_is_reproducible(self):
        first = self._sample(0, top_k=2, seed=7)
        second = self._sample(0, top_k=2, seed=7)
        self.assertIn(first, {2, 3})
        self.assertEqual(first, second)

    def test_repetition_penalty_changes_greedy_choice(self):
        # Token 3 would normally win, but it is present in the prompt and is
        # strongly penalized, so token 2 becomes the largest candidate.
        result = self._sample(3, top_k=1, repetition_penalty=100.0)
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
