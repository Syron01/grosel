import json

from web_app import inference_math
from web_app.cli import main


def _run(capsys, argv):
    main(argv)
    return capsys.readouterr().out


def test_plan_json_stays_within_budget_and_lists_phases(capsys):
    out = _run(
        capsys,
        ["plan", "--budget-gb", "5.0", "--width", "1280", "--height", "704", "--json"],
    )
    data = json.loads(out)
    assert data["peak_gb"] <= data["budget_gb"]
    assert [phase["id"] for phase in data["phases"]] == ["encode", "denoise", "decode"]


def test_plan_human_output_reports_verdict_and_storage(capsys):
    from web_app import resident_math

    expected = resident_math.plan(5.0, 1024, 576, 4, True)["verdict"]
    out = _run(capsys, ["plan"])
    assert expected in out
    assert "GB" in out
    assert "depolama" in out


def test_solve_json_without_telemetry_reports_no_data(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs.json"
    runs.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(inference_math, "RUNS_FILE", str(runs))
    out = _run(
        capsys,
        [
            "solve",
            "--budget-gb", "5.0",
            "--max-time-s", "8.0",
            "--min-mp", "0.45",
            "--steps", "4",
            "--objective", "quality",
            "--json",
        ],
    )
    data = json.loads(out)
    assert data["status"] == "no-data"
    assert data["chosen"] is None
