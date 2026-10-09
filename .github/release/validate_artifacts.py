import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from email.parser import Parser
from pathlib import Path


def inspect_archive(target: Path, expected_version: str) -> dict:
    result = {"filename": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    if target.suffix == ".whl":
        with zipfile.ZipFile(target) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("expected one wheel metadata file")
            headers = Parser().parsestr(archive.read(metadata[0]).decode())
            extensions = [name for name in archive.namelist() if "/_wbt" in name and name.endswith((".so", ".pyd"))]
            if len(extensions) != 1:
                raise ValueError("expected one native extension")
            binary = archive.read(extensions[0])
            result["extension_sha256"] = hashlib.sha256(binary).hexdigest()
            if len(binary) >= 32 and struct.unpack_from("<I", binary)[0] == 0xFEEDFACF:
                offset = 32
                for _command in range(struct.unpack_from("<I", binary, 16)[0]):
                    command, size = struct.unpack_from("<2I", binary, offset)
                    if size < 8 or offset + size > len(binary):
                        raise ValueError("invalid Mach-O load command")
                    if command == 2:
                        string_offset = struct.unpack_from("<I", binary, offset + 16)[0]
                        result["macho_string_offset"] = hex(string_offset)
                        result["macho_alignment_mod8"] = string_offset % 8
                    offset += size
    elif target.name.endswith(".tar.gz"):
        with tarfile.open(target) as archive:
            metadata = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/PKG-INFO") and len(Path(member.name).parts) == 2
            ]
            manifests = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/Cargo.toml") and len(Path(member.name).parts) == 2
            ]
            if len(metadata) != 1 or len(manifests) != 1:
                raise ValueError("expected root source metadata and Cargo manifest")
            with archive.extractfile(metadata[0]) as stream:
                headers = Parser().parsestr(stream.read().decode())
            with archive.extractfile(manifests[0]) as stream:
                manifest = tomllib.loads(stream.read().decode())
            if manifest["package"]["version"] != expected_version:
                raise ValueError("source Cargo version mismatch")
    else:
        raise ValueError("unsupported distribution archive")
    if headers["Name"] != "wbt":
        raise ValueError("package name mismatch")
    if headers["Version"] != expected_version:
        raise ValueError("distribution version mismatch")
    result["version"] = headers["Version"]
    return result


def create_environment(directory: Path) -> Path:
    venv.EnvBuilder(with_pip=True, symlinks=sys.platform != "win32").create(directory)
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def validate_runtime(wheel: Path, tests_root: Path, expected_version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="wbt-release-") as directory:
        root = Path(directory)
        environment = root / "venv"
        python = create_environment(environment)
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", "--index-url", "https://pypi.org/simple", str(wheel), "pytest"],
            cwd=root,
            check=True,
        )
        probe = """
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import wbt
import wbt._wbt
assert importlib.metadata.version('wbt') == sys.argv[1]
assert Path(wbt.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(wbt._wbt.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print(json.dumps({'version': importlib.metadata.version('wbt'), 'platform': platform.platform(), 'architecture': platform.machine(), 'python': platform.python_version(), 'installed_wheel_import': True}))
"""
        subprocess.run([str(python), "-I", "-c", probe, expected_version], cwd=root, check=True)
        tests = [
            "test_imports.py",
            "test_daily_performance.py",
            "test_backtest.py",
            "test_input_validation.py",
            "test_generate_backtest_report.py",
            "test_report_plotly_boundary.py",
            "test_verdict_component.py",
        ]
        bootstrap = (
            probe
            + """
import runpy
sys.path.insert(0, sys.argv[2])
sys.argv = ['pytest', *sys.argv[3:]]
runpy.run_module('pytest', run_name='__main__')
"""
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                bootstrap,
                expected_version,
                str(tests_root),
                "-q",
                "--import-mode=importlib",
                "-k",
                "not browser",
                *[str(tests_root / name) for name in tests],
            ],
            cwd=root,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["metadata", "runtime"])
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--tests-root", type=Path, default=Path("python/tests"))
    parser.add_argument("--version")
    arguments = parser.parse_args()
    version = arguments.version or tomllib.loads(Path("Cargo.toml").read_text())["package"]["version"]
    directory = arguments.directory.resolve()
    distributions = sorted([*directory.glob("*.whl"), *directory.glob("*.tar.gz")])
    if not distributions:
        raise ValueError("no distribution artifacts found")
    for distribution in distributions:
        print(json.dumps(inspect_archive(distribution, version)), flush=True)
    if arguments.mode == "runtime":
        wheels = [distribution for distribution in distributions if distribution.suffix == ".whl"]
        if len(wheels) != 1:
            raise ValueError("expected one native wheel per verification runner")
        validate_runtime(wheels[0], arguments.tests_root.resolve(), version)


if __name__ == "__main__":
    main()
