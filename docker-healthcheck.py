#!/usr/bin/env python3
"""
docker-healthcheck.py

Container health check script. It queries the FastAPI /health endpoint and
exits with 0 on success, non-zero otherwise.

Environment variables:
- HEALTHCHECK_URL (optional): override health endpoint URL (default: http://localhost:7860/health)
"""

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    url = os.getenv("HEALTHCHECK_URL", "http://localhost:7860/health")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                print(f"Healthcheck HTTP status: {resp.status}")
                return 1
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "healthy":
                return 0
            print(f"Unhealthy response: {data}")
            return 1
    except urllib.error.URLError as e:
        print(f"Healthcheck error: {e}")
        return 1
    except Exception as e:
        print(f"Healthcheck unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
