import pathlib
import unittest


EXPECTED_TORCH_VERSION = "2.7.1"


class TorchVersionAlignmentTest(unittest.TestCase):
    def test_ci_and_build_scripts_use_same_torch_version(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]

        ci_build = (repo_root / ".github" / "workflows" / "ci-build.yml").read_text()
        release = (repo_root / ".github" / "workflows" / "release.yml").read_text()
        init_env = (repo_root / "dockerfiles" / "init_env_inside_docker.sh").read_text()
        build_torch_wheel = (repo_root / "dockerfiles" / "build_torch_wheel.sh").read_text()

        self.assertIn(f'TORCH_VERSION: "{EXPECTED_TORCH_VERSION}"', ci_build)
        self.assertIn(f'TORCH_VERSION: "{EXPECTED_TORCH_VERSION}"', release)
        self.assertIn(f'TORCH_VERSION="{EXPECTED_TORCH_VERSION}"', init_env)
        self.assertIn(
            f'TORCH_VERSION=${{TORCH_VERSION:-{EXPECTED_TORCH_VERSION}}}',
            build_torch_wheel,
        )
        self.assertNotIn('TORCH_VERSION: "2.5.0"', ci_build)
        self.assertNotIn("torch-2.5.0", ci_build)
        self.assertNotIn("TORCH_VERSION=${TORCH_VERSION:-2.5.0}", build_torch_wheel)


if __name__ == "__main__":
    unittest.main()
