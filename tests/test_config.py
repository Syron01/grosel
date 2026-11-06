from importlib import reload

from web_app import config


def test_directories_are_created_and_env_overridden(tmp_path, monkeypatch):
    target = tmp_path / "veri"
    monkeypatch.setenv("HYPERFLUX_DATA_DIR", str(target))
    assert config._directory("HYPERFLUX_DATA_DIR", tmp_path / "varsayilan") == str(target)
    assert target.is_dir()


def test_missing_env_falls_back_to_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERFLUX_GALLERY_DIR", raising=False)
    default = tmp_path / "Cizimler"
    assert config._directory("HYPERFLUX_GALLERY_DIR", default) == str(default)
    assert default.is_dir()


def test_derived_paths_live_inside_the_configured_roots():
    assert config.RUNS_FILE.startswith(config.DATA_DIR)
    assert config.QUALITY_FILE.startswith(config.DATA_DIR)
    assert config.META_DIR.startswith(config.GALLERY_DIR)


def test_reload_respects_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERFLUX_DATA_DIR", str(tmp_path / "kendi-verisi"))
    monkeypatch.setenv("HYPERFLUX_FLUX_MODEL", "acme/flux-tiny")
    try:
        reloaded = reload(config)
        assert reloaded.RUNS_FILE == str(tmp_path / "kendi-verisi" / "runs.json")
        assert reloaded.FLUX_MODEL == "acme/flux-tiny"
    finally:
        monkeypatch.delenv("HYPERFLUX_DATA_DIR")
        monkeypatch.delenv("HYPERFLUX_FLUX_MODEL")
        reload(config)
