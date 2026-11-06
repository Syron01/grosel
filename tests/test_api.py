"""HTTP sınır testleri.

TestClient lifespan'i çalıştırmadığı için motor yüklenmez; bu dosya GPU
gerektirmeden doğrulama, dosya yolu ve güvenlik başlıklarını sınar.
Tek maliyeti torch/diffusers importudur.
"""

import pytest

httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from web_app import server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


def test_index_serves_the_studio_shell(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "promptInput" in response.text


def test_security_headers_are_present(client):
    headers = client.get("/api/info").headers
    assert "script-src 'self'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"


def test_endpoints_that_need_the_engine_report_503_before_startup(client):
    assert client.get("/api/system").status_code == 503
    assert client.post("/api/cache/clear").status_code == 503


def test_backend_selection_is_reported(client):
    info = client.get("/api/info").json()
    assert info["backend"] == server.BACKEND


def test_info_carries_the_promise_the_stage_panel_reads(client, monkeypatch):
    """Panel'de yazan her sayı bu uçtan gelir; ezberlenen sabit arayüze sızmaz."""
    monkeypatch.setattr(server, "load_runs", lambda: [])
    info = client.get("/api/info").json()
    assert info["vram_ceiling_gb"] == server.VRAM_CEILING_GB
    assert info["tier_ladder"] == list(server.TIER_LADDER)
    assert info["measured"] == {}
    contract = info["contract"]
    assert contract["min_blend_px"] == server.quality_math.MIN_BLEND_PX
    assert contract["seam_z_limit"] == server.quality_math.SEAM_Z_LIMIT
    assert contract["tile_px"] == [server.quality_math.TILE_MIN_PX,
                                   server.quality_math.TILE_STRIDE_PX]
    assert "vram" not in info, "esnek düz metin alanı kaldırıldı: panel yapılandırılmış veriyi kullanır"


def test_shell_and_assets_are_revalidated_after_every_release(client):
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/info").headers


@pytest.mark.parametrize("payload,code", [
    ({}, 422),
    ({"prompt": "   ", "style": "pure_raw"}, 422),
    ({"prompt": "gece sokagi", "style": "matte_chiaroscuro", "unknown": 1}, 422),
])
def test_optimize_prompt_rejects_invalid_payloads(client, payload, code):
    assert client.post("/api/optimize_prompt", json=payload).status_code == code


def test_optimize_prompt_rejects_unknown_style(client):
    response = client.post("/api/optimize_prompt",
                           json={"prompt": "gece sokagi", "style": "yok_boyle_bir_stil"})
    assert response.status_code == 422
    assert "stil" in response.text.lower()


def test_optimize_prompt_compiles_without_touching_the_model(client):
    response = client.post("/api/optimize_prompt",
                           json={"prompt": "tokyo rain street", "style": "cyberpunk_noir"})
    assert response.status_code == 200
    optimized = response.json()["optimized"]
    assert isinstance(optimized, str) and len(optimized) > len("tokyo rain street")
    assert "tokyo" in optimized.lower()


def test_path_traversal_cannot_reach_outside_the_gallery(client):
    assert client.get("/images/..%5C..%5Cpyproject.toml").status_code == 404
    assert client.delete("/api/gallery/..%5Cpyproject.toml").status_code == 404


def test_non_image_files_are_not_deleted(client, monkeypatch):
    response = client.delete("/api/gallery/pyproject.toml")
    assert response.status_code == 404


def test_runs_limit_is_bounded(client, monkeypatch):
    monkeypatch.setattr(server, "RUNS_FILE", "yok.json")
    assert client.get("/api/runs", params={"limit": 5000}).status_code == 422
    assert client.get("/api/runs", params={"limit": 10}).json() == []


def test_websocket_rejects_bad_requests_before_touching_the_engine(client):
    with client.websocket_connect("/ws/generate") as socket:
        socket.send_text('{"prompt": "   "}')
        assert socket.receive_json()["type"] == "error"
        socket.send_json({"prompt": "gece sokagi", "style": "yok_boyle_stil"})
        assert socket.receive_json()["type"] == "error"
        socket.send_json({"prompt": "gece sokagi", "count": 99})
        assert socket.receive_json()["type"] == "error"


def test_runs_do_not_leak_absolute_machine_paths(client, monkeypatch, tmp_path):
    record = tmp_path / "runs.json"
    record.write_text(
        '[{"filename": "a.png", "filepath": "C:/kullanicinin/masasi/a.png", "total_time": 1.2}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "RUNS_FILE", str(record))
    run = client.get("/api/runs").json()[0]
    assert run["filename"] == "a.png"
    assert "filepath" not in run


def test_style_list_is_complete_and_serialisable(client):
    styles = client.get("/api/styles").json()
    assert len(styles) == len(server.STYLE_PRESETS)
    assert all({"key", "name", "desc"} <= set(s) for s in styles)
