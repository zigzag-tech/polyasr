import unittest
from types import SimpleNamespace

from cuda.runtime_preflight import validate_pytorch_stack


class RuntimePreflightTest(unittest.TestCase):
    def test_rejects_mismatched_torch_and_torchaudio_cuda_versions(self):
        modules = {
            "torch": SimpleNamespace(
                __version__="2.13.0+cu130",
                version=SimpleNamespace(cuda="13.0"),
            ),
            "torchaudio": SimpleNamespace(
                __version__="2.11.0+cu129",
                version=SimpleNamespace(cuda="12.9"),
            ),
            "torchvision": SimpleNamespace(__version__="0.28.0"),
            "silero_vad": SimpleNamespace(),
            "resemblyzer": SimpleNamespace(),
        }

        with self.assertRaisesRegex(RuntimeError, "CUDA mismatch"):
            validate_pytorch_stack(import_module=modules.__getitem__)

    def test_accepts_matching_stack(self):
        modules = {
            "torch": SimpleNamespace(
                __version__="2.11.0+cu129",
                version=SimpleNamespace(cuda="12.9"),
            ),
            "torchaudio": SimpleNamespace(
                __version__="2.11.0+cu129",
                version=SimpleNamespace(cuda="12.9"),
            ),
            "torchvision": SimpleNamespace(__version__="0.26.0+cu129"),
            "silero_vad": SimpleNamespace(),
            "resemblyzer": SimpleNamespace(),
        }

        result = validate_pytorch_stack(import_module=modules.__getitem__)

        self.assertEqual(result["torch"], "2.11.0+cu129")
        self.assertEqual(result["cuda"], "12.9")

    def test_accepts_torchaudio_without_exposed_cuda_version(self):
        modules = {
            "torch": SimpleNamespace(
                __version__="2.11.0+cu129",
                version=SimpleNamespace(cuda="12.9"),
            ),
            "torchaudio": SimpleNamespace(
                __version__="2.11.0+cu129",
                version=SimpleNamespace(cuda=None),
            ),
            "torchvision": SimpleNamespace(__version__="0.26.0+cu129"),
            "silero_vad": SimpleNamespace(),
            "resemblyzer": SimpleNamespace(),
        }

        result = validate_pytorch_stack(import_module=modules.__getitem__)

        self.assertEqual(result["torchaudio"], "2.11.0+cu129")


if __name__ == "__main__":
    unittest.main()
