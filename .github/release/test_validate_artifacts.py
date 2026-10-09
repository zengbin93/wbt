import io
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from validate_artifacts import create_environment, inspect_archive


class DistributionValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def wheel(self, version="0.9.2", name="wbt", metadata=True, extension=True):
        target = self.root / "candidate.whl"
        with zipfile.ZipFile(target, "w") as archive:
            if metadata:
                archive.writestr("wbt-0.9.2.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
            if extension:
                archive.writestr("wbt/_wbt.abi3.so", b"binary fixture")
        return target

    def source(self, cargo_version="0.9.2"):
        target = self.root / "candidate.tar.gz"
        with tarfile.open(target, "w:gz") as archive:
            for filename, content in {
                "wbt-0.9.2/PKG-INFO": b"Name: wbt\nVersion: 0.9.2\n",
                "wbt-0.9.2/Cargo.toml": f'[package]\nname="wbt"\nversion="{cargo_version}"\n'.encode(),
            }.items():
                member = tarfile.TarInfo(filename)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return target

    def test_wheel_metadata_and_digest_are_recorded(self):
        result = inspect_archive(self.wheel(), "0.9.2")
        self.assertEqual(result["version"], "0.9.2")
        self.assertEqual(len(result["sha256"]), 64)

    def test_wrong_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "version"):
            inspect_archive(self.wheel(version="0.9.1"), "0.9.2")

    def test_wrong_package_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "package"):
            inspect_archive(self.wheel(name="other"), "0.9.2")

    def test_missing_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            inspect_archive(self.wheel(metadata=False), "0.9.2")

    def test_missing_extension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "extension"):
            inspect_archive(self.wheel(extension=False), "0.9.2")

    def test_source_metadata_matches_cargo_version(self):
        self.assertEqual(inspect_archive(self.source(), "0.9.2")["version"], "0.9.2")

    def test_source_cargo_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cargo"):
            inspect_archive(self.source(cargo_version="0.9.1"), "0.9.2")

    def test_corrupt_wheel_is_rejected(self):
        target = self.root / "corrupt.whl"
        target.write_bytes(b"not a zip")
        with self.assertRaises(zipfile.BadZipFile):
            inspect_archive(target, "0.9.2")

    def test_isolated_interpreter_and_pip_are_usable(self):
        python = create_environment(self.root / "venv")
        process = subprocess.run([str(python), "-I", "-m", "pip", "--version"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
