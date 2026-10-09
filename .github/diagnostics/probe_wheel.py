import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import venv
from pathlib import Path


def check_wheel_hash(wheel: Path, expected: str | None = None) -> str:
    actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"wheel SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def inspect_macho(data: bytes) -> dict:
    if len(data) < 32 or struct.unpack_from("<I", data)[0] != 0xFEEDFACF:
        raise ValueError("expected 64-bit little-endian Mach-O")
    header = struct.unpack_from("<8I", data)
    result = {"cpu_type": hex(header[1])}
    offset = 32
    for _command_index in range(header[4]):
        if offset + 8 > len(data):
            raise ValueError("truncated Mach-O load command")
        command, size = struct.unpack_from("<2I", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError("truncated Mach-O load command")
        if command == 2:
            _, _, string_offset, string_size = struct.unpack_from("<4I", data, offset + 8)
            result.update(
                string_offset=hex(string_offset),
                string_size=string_size,
                string_alignment_mod8=string_offset % 8,
            )
        if command == 11:
            values = struct.unpack_from("<20I", data, offset)
            result.update(indirect_count=values[15], indirect_end=hex(values[14] + values[15] * 4))
        if command == 0x32:
            _, _, _, minimum, sdk, tool_count = struct.unpack_from("<6I", data, offset)
            result.update(minimum_os=hex(minimum), sdk=hex(sdk), tools=[])
            for tool_index in range(tool_count):
                tool, version = struct.unpack_from("<2I", data, offset + 24 + tool_index * 8)
                result["tools"].append({"tool": tool, "version": hex(version)})
        offset += size
    if "string_offset" not in result:
        raise ValueError("Mach-O is missing LC_SYMTAB")
    return result


def run_logged(arguments: list[str], log: Path) -> int:
    with log.open("w") as output:
        process = subprocess.run(arguments, stdout=output, stderr=subprocess.STDOUT, check=False)
    return process.returncode


def create_environment(directory: Path) -> str:
    venv.create(directory, with_pip=True, symlinks=True)
    return str(directory / "bin" / "python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--require-success", action="store_true")
    arguments = parser.parse_args()
    wheel = arguments.wheel.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"case": arguments.case, "wheel_sha256": check_wheel_hash(wheel, arguments.expected_sha256)}
    environment = output.parent / f"venv-{arguments.case}"
    python = create_environment(environment)
    report["install_exit"] = run_logged(
        [python, "-m", "pip", "install", "--index-url", "https://pypi.org/simple", str(wheel), "pytest"],
        output / "install.log",
    )
    if report["install_exit"] != 0:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit("wheel/dependency installation failed; no load conclusion")
    extensions = list(environment.glob("lib/python*/site-packages/wbt/_wbt*.so"))
    if len(extensions) != 1:
        raise ValueError(f"expected one installed WBT extension, got {extensions}")
    extension = extensions[0].read_bytes()
    report["extension_sha256"] = hashlib.sha256(extension).hexdigest()
    report["layout"] = inspect_macho(extension)
    report["import_exit"] = run_logged([python, "-I", "-c", "import wbt; print(wbt.__file__)"], output / "import.log")
    root = Path(__file__).resolve().parents[2]
    tests = [
        root / "python" / "tests" / name
        for name in ("test_imports.py", "test_daily_performance.py", "test_backtest.py", "test_input_validation.py")
    ]
    report["pytest_exit"] = run_logged(
        [python, "-I", "-m", "pytest", "-q", "--import-mode=importlib", *map(str, tests)], output / "pytest.log"
    )
    passed = re.search(r"(\d+) passed", (output / "pytest.log").read_text())
    report["tests_passed"] = int(passed.group(1)) if passed else 0
    run_logged([python, "-m", "pip", "freeze"], output / "installed-dependencies.txt")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as document:
            document.write(f"### {arguments.case}\n\n```json\n{json.dumps(report, indent=2)}\n```\n")
    if arguments.require_success and (
        report["import_exit"] != 0 or report["pytest_exit"] != 0 or report["layout"]["string_alignment_mod8"] != 0
    ):
        raise SystemExit("unstripped control did not pass; diagnostic validation incomplete")


if __name__ == "__main__":
    main()
