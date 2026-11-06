import os
import unittest
from unittest import mock

from op_builder.builder import CUDAOpBuilder, parse_compute_capability


class _Builder(CUDAOpBuilder):
    def __init__(self):
        super().__init__(name="test_op")

    def absolute_name(self):
        return "test_op"

    def sources(self):
        return []


class ParseComputeCapabilityTests(unittest.TestCase):
    def test_single_and_double_digit_majors(self):
        self.assertEqual(parse_compute_capability("9.0"), (9, 0, False))
        self.assertEqual(parse_compute_capability("10.0"), (10, 0, False))
        self.assertEqual(parse_compute_capability("12.0+PTX"), (12, 0, True))
        self.assertEqual(parse_compute_capability("9.0a"), (9, 0, False))

    def test_malformed_entry_is_rejected(self):
        for bad in ("", "90", "sm_90", "9.x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_compute_capability(bad)


class ComputeCapabilityArgsTests(unittest.TestCase):
    def test_arch_list_with_blackwell_entries(self):
        builder = _Builder()
        builder.jit_mode = False
        with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0 10.0 12.0+PTX"}):
            args = builder.compute_capability_args()
        self.assertEqual(args, [
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_100,code=sm_100",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_120,code=compute_120",
        ])
        self.assertTrue(builder.enable_bf16)

    def test_pre_ampere_disables_bf16(self):
        builder = _Builder()
        builder.jit_mode = False
        with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "7.5;9.0"}):
            builder.compute_capability_args()
        self.assertFalse(builder.enable_bf16)


if __name__ == "__main__":
    unittest.main()
