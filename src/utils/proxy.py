from __future__ import annotations

from typing import Any

import requests


def verify_proxy_connection(
    base_url: str, timeout: int = 5, verify_ssl: bool = True
) -> bool:
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=timeout, verify=verify_ssl)
    except Exception:
        return False
    return response.status_code == 200


def verify_sdk_endpoint(base_url: str, timeout: int = 5) -> bool:
    """Check that the proxy's SDK key allocation endpoint is reachable.

    Sends a POST to /sdk/allocate-key. Returns True if the proxy responds
    (200 = key available, 503 = all keys exhausted but service is up).
    """
    url = f"{base_url.rstrip('/')}/sdk/allocate-key"
    try:
        resp = requests.post(url, timeout=timeout)
    except Exception:
        return False
    return resp.status_code in (200, 503)
