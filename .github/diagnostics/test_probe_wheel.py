import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from probe_wheel import check_wheel_hash, create_environment, inspect_macho


class WheelEvidenceTests(unittest.TestCase):
    def test_environment_interpreter_and_pip_are_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            python = create_environment(Path(directory) / "venv")
            process = subprocess.run([python, "-I", "-m", "pip", "--version"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)

    def macho(self, string_offset):
        header = struct.pack("<8I", 0xFEEDFACF, 0x100000C, 0, 6, 2, 104, 0, 0)
        symbols = struct.pack("<6I", 2, 24, 0x2000, 255, string_offset, 4176)
        indirect = [11, 80, *([0] * 18)]
        indirect[14] = 0x28D3B50
        indirect[15] = 477
        return header + symbols + struct.pack("<20I", *indirect)

    def test_reports_misaligned_string_pool_and_indirect_end(self):
        report = inspect_macho(self.macho(0x28D42C4))
        self.assertEqual(report["string_offset"], "0x28d42c4")
        self.assertEqual(report["string_alignment_mod8"], 4)
        self.assertEqual(report["indirect_count"], 477)
        self.assertEqual(report["indirect_end"], "0x28d42c4")

    def test_distinguishes_aligned_string_pool(self):
        self.assertEqual(inspect_macho(self.macho(0x28D42C8))["string_alignment_mod8"], 0)

    def test_rejects_unsupported_binary(self):
        with self.assertRaisesRegex(ValueError, "64-bit little-endian Mach-O"):
            inspect_macho(b"not a Mach-O file")

    def test_rejects_truncated_load_commands(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            inspect_macho(self.macho(0x28D42C4)[:-1])

    def test_checks_exact_wheel_hash_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "fixture.whl"
            wheel.write_bytes(b"abc")
            expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            self.assertEqual(check_wheel_hash(wheel, expected), expected)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                check_wheel_hash(wheel, "0" * 64)


if __name__ == "__main__":
    unittest.main()
