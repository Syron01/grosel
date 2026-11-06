import json

import pytest

from web_app.storage import read_json, write_json_atomic


def test_round_trip_keeps_unicode(tmp_path):
    path = tmp_path / "runs.json"
    payload = [{"prompt": "şömine ışığı — 1080p", "peak_gb": 4.43}]
    write_json_atomic(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "alt" / "klasor" / "quality.json"
    write_json_atomic(path, {"ok": True})
    assert read_json(path, None) == {"ok": True}


def test_read_json_returns_default_for_missing_and_corrupt(tmp_path):
    missing = tmp_path / "yok.json"
    assert read_json(missing, {"fallback": 1}) == {"fallback": 1}

    broken = tmp_path / "bozuk.json"
    broken.write_text("{eksik", encoding="utf-8")
    assert read_json(broken, []) == []


def test_failed_replace_leaves_original_and_no_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "runs.json"
    write_json_atomic(path, ["once"])

    def boom(_src, _dst):
        raise OSError("disk dolu")

    monkeypatch.setattr("web_app.storage.os.replace", boom)
    with pytest.raises(OSError):
        write_json_atomic(path, ["sonra"])

    assert read_json(path, None) == ["once"]
    assert list(tmp_path.glob("runs.json.*.tmp")) == []


def test_overwrite_replaces_previous_content(tmp_path):
    path = tmp_path / "runs.json"
    write_json_atomic(path, {"a": 1, "b": 2})
    write_json_atomic(path, {"a": 9})
    assert read_json(path, None) == {"a": 9}
