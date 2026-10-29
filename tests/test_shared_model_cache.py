"""Exercise the child process environment and refusal before a model starts."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "research/run_with_shared_cache.py"


class SharedModelCacheTests(unittest.TestCase):
    def test_child_uses_one_cache_and_keeps_unrelated_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = (Path(temp) / "shared").resolve()
            env = dict(
                os.environ,
                HF_HOME=str(Path(temp) / "legacy-per-run"),
                HF_HUB_CACHE=str(Path(temp) / "another-cache"),
                TRANSFORMERS_CACHE=str(Path(temp) / "old-override"),
                CACHE_TEST_SENTINEL="preserved",
            )
            program = (
                "import os,json;print(json.dumps({k:os.environ.get(k) for k in "
                "['HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE',"
                "'TRANSFORMERS_CACHE','CACHE_TEST_SENTINEL']}))"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--cache-root",
                    str(cache),
                    "--minimum-free-gib",
                    "0",
                    "--",
                    sys.executable,
                    "-c",
                    program,
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            actual = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(actual["HF_HOME"], str(cache))
            self.assertEqual(actual["HF_HUB_CACHE"], str(cache / "hub"))
            self.assertEqual(actual["HUGGINGFACE_HUB_CACHE"], str(cache / "hub"))
            self.assertIsNone(actual["TRANSFORMERS_CACHE"])
            self.assertEqual(actual["CACHE_TEST_SENTINEL"], "preserved")
            self.assertFalse((Path(temp) / "legacy-per-run").exists())

    def test_refuses_per_run_cache_before_creating_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "runs" / "example" / "cache"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--cache-root",
                    str(cache),
                    "--minimum-free-gib",
                    "0",
                    "--",
                    sys.executable,
                    "-c",
                    "raise RuntimeError('child must not start')",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("outside experiment runs", result.stderr)
            self.assertFalse(cache.exists())
            self.assertNotIn("RuntimeError", result.stderr)

    def test_refuses_low_space_before_starting_command(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = (Path(temp) / "shared").resolve()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--cache-root",
                    str(cache),
                    "--minimum-free-gib",
                    "1000000000",
                    "--",
                    sys.executable,
                    "-c",
                    "raise RuntimeError('child must not start')",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Insufficient free disk", result.stderr)
            self.assertFalse(cache.exists())
            self.assertNotIn("RuntimeError", result.stderr)


if __name__ == "__main__":
    unittest.main()
