#!/usr/bin/env python3
"""
docker-pre-build-check.py

Run lightweight checks before building the Docker image to catch common
misconfigurations early. This script is safe to run locally and in CI.

Checks performed:
- Python version compatibility
- Presence of critical files (Dockerfile, requirements.txt, veridika_server.py)
- Existence of workflow config file (from .env or environment)
- Optional .env validation for required keys if present

Exit codes:
  0 -> All checks passed or only non-critical warnings found
  1 -> Critical errors detected (missing files or config)
"""

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent


def print_ok(msg: str) -> None:
    print(f"✅ {msg}")


def print_warn(msg: str) -> None:
    print(f"⚠️  {msg}")


def print_err(msg: str) -> None:
    print(f"❌ {msg}")


def read_dotenv(path: Path) -> dict:
    env = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def main() -> int:
    print("\n🚀 Veridika Docker pre-build checks\n" + ("-" * 40))
    failed = False

    # 1) Python version
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 11:
        print_ok(f"Python {py_ver.major}.{py_ver.minor} detected")
    else:
        print_warn("Python 3.11+ recommended for local checks (container uses 3.11)")

    # 2) Critical files
    critical_files = [
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "veridika_server.py",
    ]
    for f in critical_files:
        if f.exists():
            print_ok(f"Found {f.relative_to(REPO_ROOT)}")
        else:
            print_err(f"Missing required file: {f.relative_to(REPO_ROOT)}")
            failed = True

    # 3) Workflow config path
    # Priority: env var -> .env -> warn
    workflow_config = os.getenv("WORKFLOW_CONFIG_PATH")
    env_file = REPO_ROOT / ".env"
    env_map = {}
    if env_file.exists():
        env_map = read_dotenv(env_file)
        print_ok("Found .env file")
        if not workflow_config:
            workflow_config = env_map.get("WORKFLOW_CONFIG_PATH")
    else:
        print_warn(".env not found (will rely on runtime environment)")

    if workflow_config:
        cfg_path = REPO_ROOT / workflow_config
        if cfg_path.exists():
            print_ok(f"Workflow config present: {workflow_config}")
        else:
            print_err(f"Workflow config missing: {workflow_config}")
            failed = True
    else:
        print_warn("WORKFLOW_CONFIG_PATH not set in env or .env")

    # 4) Optional .env key checks
    if env_map:
        required_keys = ["API_KEY", "REDIS_URL", "WORKFLOW_CONFIG_PATH"]
        missing = [k for k in required_keys if not env_map.get(k)]
        if missing:
            print_warn(".env is missing recommended keys: " + ", ".join(missing))
        else:
            print_ok(".env contains required keys")

    # 5) Source directories sanity
    src_dirs = [
        REPO_ROOT / "veridika",
        REPO_ROOT / "veridika" / "src" / "workflows",
    ]
    for d in src_dirs:
        if d.exists():
            print_ok(f"Found directory: {d.relative_to(REPO_ROOT)}")
        else:
            print_warn(f"Directory not found (skip): {d.relative_to(REPO_ROOT)}")

    print("-" * 40)
    if failed:
        print_err("Pre-build checks failed. Fix issues above and retry.")
        return 1
    print_ok("All critical pre-build checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
