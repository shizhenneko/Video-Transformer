from utils.proxy import verify_proxy_connection, verify_sdk_endpoint


class DummyResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


# --- verify_proxy_connection ---


def test_proxy_success(monkeypatch):
    def fake_get(url, timeout=5, verify=True):
        return DummyResponse(200)

    monkeypatch.setattr("requests.get", fake_get)
    assert verify_proxy_connection("http://localhost:8000", timeout=1)


def test_proxy_failure_status(monkeypatch):
    def fake_get(url, timeout=5, verify=True):
        return DummyResponse(500)

    monkeypatch.setattr("requests.get", fake_get)
    assert not verify_proxy_connection("http://localhost:8000", timeout=1)


def test_proxy_exception(monkeypatch):
    def fake_get(url, timeout=5, verify=True):
        raise RuntimeError("boom")

    monkeypatch.setattr("requests.get", fake_get)
    assert not verify_proxy_connection("http://localhost:8000", timeout=1)


# --- verify_sdk_endpoint ---


def test_sdk_endpoint_available(monkeypatch):
    def fake_post(url, timeout=5):
        return DummyResponse(200)

    monkeypatch.setattr("requests.post", fake_post)
    assert verify_sdk_endpoint("http://localhost:8000", timeout=1)


def test_sdk_endpoint_exhausted_still_available(monkeypatch):
    def fake_post(url, timeout=5):
        return DummyResponse(503)

    monkeypatch.setattr("requests.post", fake_post)
    assert verify_sdk_endpoint("http://localhost:8000", timeout=1)


def test_sdk_endpoint_unexpected_status(monkeypatch):
    def fake_post(url, timeout=5):
        return DummyResponse(404)

    monkeypatch.setattr("requests.post", fake_post)
    assert not verify_sdk_endpoint("http://localhost:8000", timeout=1)


def test_sdk_endpoint_connection_error(monkeypatch):
    def fake_post(url, timeout=5):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("requests.post", fake_post)
    assert not verify_sdk_endpoint("http://localhost:8000", timeout=1)
