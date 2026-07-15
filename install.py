"""Cross-platform one-command installer: ``python install.py``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


MINIMUM = (3, 10)


def virtualenv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sys.version_info < MINIMUM:
        raise SystemExit("APT requires Python 3.10 or newer")
    root = Path(__file__).resolve().parent
    environment = args.venv.resolve()
    python = virtualenv_python(environment)
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(python), "-m", "pip", "install", "-e", str(root)],
    ]
    if not args.skip_tests:
        commands.append(
            [str(python), "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"]
        )
    if args.dry_run:
        print(f"create virtual environment: {environment}")
        for command in commands:
            print(subprocess.list2cmdline(command))
        return 0
    if not python.exists():
        print(f"Creating isolated environment at {environment}")
        venv.EnvBuilder(with_pip=True).create(environment)
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    executable = environment / ("Scripts/apt-cognitive.exe" if os.name == "nt" else "bin/apt-cognitive")
    print("APT installation and validation completed.")
    print(f"Run: {executable} --home .apt init")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

