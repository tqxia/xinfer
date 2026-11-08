"""CPU checks for MegaQwen's explicit tensor and operator-registration contract."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from compiler.qwen3 import megaqwen


class TensorMetadata:
    """Exercise metadata validation without allocating the model or using CUDA."""

    def __init__(self, shape, dtype=torch.bfloat16, device="cuda:0", pointer=4096):
        self.shape, self.dtype, self.device = tuple(shape), dtype, torch.device(device)
        self.pointer = pointer

    def is_contiguous(self):
        return True

    def dim(self):
        return len(self.shape)

    def data_ptr(self):
        return self.pointer

    def new_empty(self, shape, *, dtype):
        return TensorMetadata(shape, dtype, self.device)


def metadata_inputs(capacity=1024, length=129):
    return (
        TensorMetadata((1, 1), torch.int64),
        [TensorMetadata(shape) for _, shape in megaqwen.WEIGHT_SPECS],
        [TensorMetadata((1, 8, capacity, 128)) for _ in range(56)],
        TensorMetadata((capacity, 128)),
        TensorMetadata((capacity, 128)),
        [
            TensorMetadata(shape, torch.bfloat16 if index == 0 else torch.float32)
            for index, shape in enumerate(megaqwen.WORKSPACE_SHAPES)
        ],
        TensorMetadata((length + 1,), torch.uint8, "cpu"),
    )


class MegaQwenContract(unittest.TestCase):
    def test_decode_export_reuses_parameters_and_keeps_only_rope_buffers(self):
        model = torch.nn.Linear(2, 2)
        model.config = object()
        parameters = list(model.parameters())
        pointers = [tensor.data_ptr() for tensor in parameters]
        cos, sin = megaqwen.rope_tables(512, "cpu")
        with (
            patch.object(megaqwen, "validate_geometry"),
            patch.object(megaqwen, "_adapt_flash_model", return_value=model),
            patch.object(megaqwen, "load_megaqwen_ops"),
            patch.object(megaqwen, "rope_tables", return_value=(cos, sin)),
        ):
            step = megaqwen.MegaDecodeStep(model, 512)
        self.assertIs(step.model, model)
        self.assertEqual([tensor.data_ptr() for tensor in step.parameters()], pointers)
        self.assertTrue(all(not tensor.requires_grad for tensor in parameters))
        self.assertEqual(set(dict(step.named_parameters())), {"model.weight", "model.bias"})
        self.assertEqual(set(dict(step.named_buffers())), {"megaqwen_cos", "megaqwen_sin"})
        kv = tuple(torch.empty(0) for _ in range(56))
        for position in (0, 128):
            ids = torch.tensor([[13]])
            extent = torch.empty(position + 2, dtype=torch.uint8)
            expected = object()
            with (
                patch.object(step, "get_parameter", return_value=model.weight) as parameter,
                patch.object(
                    torch.ops.xinfer, "megaqwen_decode", return_value=expected, create=True
                ) as op,
            ):
                self.assertIs(step(ids, torch.tensor([position]), kv, extent), expected)
                self.assertEqual(
                    [call.args[0] for call in parameter.call_args_list],
                    [name for name, _ in megaqwen.WEIGHT_SPECS],
                )
                actual = op.call_args.args
                self.assertIs(actual[0], ids)
                self.assertEqual(len(actual[1]), 311)
                self.assertTrue(all(a is b for a, b in zip(actual[2], kv)))
                self.assertIs(actual[3], cos)
                self.assertIs(actual[4], sin)
                self.assertEqual(
                    [tuple(t.shape) for t in actual[5]], list(megaqwen.WORKSPACE_SHAPES)
                )
                self.assertIs(actual[6], extent)
        self.assertEqual(set(dict(step.named_buffers())), {"megaqwen_cos", "megaqwen_sin"})

    def test_fake_metadata_covers_capacity_and_valid_prefix_without_data_reads(self):
        for capacity, length in ((512, 1), (1024, 129), (16384, 8193), (40960, 40960)):
            with self.subTest(capacity=capacity, length=length):
                inputs = metadata_inputs(capacity, length)
                result = megaqwen.fake_decode(*inputs)
                self.assertEqual(result.shape, (1, 151936))
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertEqual(result.device, inputs[0].device)
                self.assertTrue(
                    all(
                        result is not tensor
                        for group in inputs
                        for tensor in (group if isinstance(group, list) else [group])
                    )
                )
        for mutation in (
            lambda args: args.__setitem__(1, args[1][:-1]),
            lambda args: args.__setitem__(2, args[2][:-1]),
            lambda args: args.__setitem__(5, args[5][:-1]),
            lambda args: args.__setitem__(6, TensorMetadata((1026,), torch.uint8, "cpu")),
            lambda args: args.__setitem__(0, TensorMetadata((1, 2), torch.int64)),
        ):
            inputs = list(metadata_inputs())
            mutation(inputs)
            with self.assertRaises(RuntimeError):
                megaqwen.fake_decode(*inputs)

    def test_registration_is_once_per_library(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "libxinfer_megaqwen.so"
            library.touch()
            decorator = Mock()
            with (
                patch.object(megaqwen, "_loaded", None),
                patch.object(torch.ops, "load_library") as load,
                patch.object(torch.library, "register_fake", return_value=decorator) as register,
            ):
                megaqwen.load_megaqwen_ops(library)
                megaqwen.load_megaqwen_ops(library)
                load.assert_called_once_with(str(library.resolve()))
                register.assert_called_once_with("xinfer::megaqwen_decode")
                decorator.assert_called_once_with(megaqwen.fake_decode)
                self.assertEqual(megaqwen.megaqwen_library(), library.resolve())
                with self.assertRaisesRegex(ValueError, "two different"):
                    megaqwen.load_megaqwen_ops(library.with_name("other.so"))

    def test_workspace_and_rope_are_explicit_tensors(self):
        scratch = megaqwen.allocate_workspace("cpu")
        self.assertEqual([tuple(t.shape) for t in scratch], list(megaqwen.WORKSPACE_SHAPES))
        self.assertEqual(scratch[0].dtype, torch.bfloat16)
        self.assertTrue(all(t.dtype == torch.float32 for t in scratch[1:]))
        self.assertEqual(len({t.data_ptr() for t in scratch}), 12)
        cos, sin = megaqwen.rope_tables(512, "cpu")
        self.assertEqual(cos.shape, (512, 128))
        self.assertTrue(torch.equal(cos[:, :64], cos[:, 64:]))
        self.assertTrue(torch.equal(sin[:, :64], sin[:, 64:]))
        self.assertTrue(bool(cos[0].eq(1).all()) and bool(sin[0].eq(0).all()))


if __name__ == "__main__":
    unittest.main()
