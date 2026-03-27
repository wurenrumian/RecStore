import pathlib
import unittest


class InitDlrmScriptTest(unittest.TestCase):
    def test_installs_pinned_torchrec_stack_against_global_torch(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        script = (repo_root / 'dockerfiles' / 'init_dlrm.sh').read_text()

        self.assertIn('PINNED_TORCHREC_VERSION="${PINNED_TORCHREC_VERSION:-1.2.0}"', script)
        self.assertIn('PINNED_FBGEMM_GPU_VERSION="${PINNED_FBGEMM_GPU_VERSION:-1.2.0}"', script)
        self.assertIn('PINNED_TORCHMETRICS_VERSION="${PINNED_TORCHMETRICS_VERSION:-1.0.3}"', script)
        self.assertIn('torchrec==${PINNED_TORCHREC_VERSION}', script)
        self.assertIn('fbgemm-gpu==${PINNED_FBGEMM_GPU_VERSION}', script)
        self.assertIn('torchmetrics==${PINNED_TORCHMETRICS_VERSION}', script)
        self.assertIn('Global torch version:', script)
        self.assertIn('Global torch cxx11abi:', script)
        self.assertIn('FORCE_TORCHREC_SOURCE_BUILD="${FORCE_TORCHREC_SOURCE_BUILD:-0}"', script)
        self.assertIn('setup.py bdist_wheel', script)
        self.assertNotIn('/home/shq/', script)

    def test_skips_source_build_when_cuda_toolkit_is_unavailable(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        script = (repo_root / 'dockerfiles' / 'init_dlrm.sh').read_text()

        self.assertIn('Detected CPU-only torch install; skipping TorchRec/FBGEMM install', script)
        self.assertIn('Skipping TorchRec/FBGEMM install because CUDA toolkit is not available', script)
        self.assertIn('FORCE_TORCHREC_SOURCE_BUILD="${FORCE_TORCHREC_SOURCE_BUILD:-0}"', script)
        self.assertIn('if [ "${FORCE_TORCHREC_SOURCE_BUILD}" != "1" ]; then', script)


if __name__ == '__main__':
    unittest.main()
