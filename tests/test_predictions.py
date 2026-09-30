import json
import math
from pathlib import Path

import pytest
import synthetic

from fitpipeline import build_predictions, load_all_meshes, validate_predictions, write_predictions
from fitpipeline.__main__ import main
from fitpipeline.loader import MeshData

DATA_DIR = Path(__file__).resolve().parents[1] / "assignment" / "data"


@pytest.fixture(scope="module")
def meshes():
    return load_all_meshes(DATA_DIR)


@pytest.fixture(scope="module")
def predictions(meshes):
    return build_predictions(meshes)


def test_one_valid_entry_per_segment(meshes, predictions):
    assert len(predictions) == 43
    validate_predictions(predictions, meshes)  # raises on any schema violation


def test_primitives_have_a_usable_fit_and_unresolved_do_not(predictions):
    for entry in predictions:
        fit = entry["fit"]
        if entry["kind"] == "primitive":
            assert fit["status"] in ("fitted", "partial")
            assert fit["parameters"] is not None
        else:
            assert entry["primitive_type"] is None
            assert fit["status"] == "failed"
            assert fit["parameters"] is None
            assert "Attempted" in fit["notes"]  # names the attempted type


def test_parameters_follow_the_suggested_schema(predictions):
    expected_keys = {
        "plane": {"point", "normal"},
        "cylinder": {"axis_point", "axis_direction", "radius"},
        "cone": {"apex", "axis_direction", "half_angle_radians"},
        "sphere": {"center", "radius"},
        "torus": {"center", "axis_direction", "major_radius", "minor_radius"},
    }
    for entry in predictions:
        if entry["kind"] != "primitive":
            continue
        parameters = entry["fit"]["parameters"]
        assert set(parameters) == expected_keys[entry["primitive_type"]]
        for key in ("normal", "axis_direction"):
            if key in parameters:
                assert math.hypot(*parameters[key]) == pytest.approx(1.0, abs=1e-9)
        for key in ("radius", "major_radius", "minor_radius"):
            if key in parameters:
                assert parameters[key] > 0
        if "half_angle_radians" in parameters:
            assert 0 < parameters["half_angle_radians"] < math.pi / 2


def test_known_parameters_of_assignment_segments(predictions):
    by_key = {(p["mesh_id"], p["segment_id"]): p for p in predictions}
    assert by_key[("mesh_02", 0)]["fit"]["parameters"]["radius"] == pytest.approx(2.0, rel=1e-6)


def test_validation_rejects_missing_duplicate_and_bad_entries(meshes, predictions):
    with pytest.raises(ValueError):
        validate_predictions(predictions[1:], meshes)
    with pytest.raises(ValueError):
        validate_predictions(predictions + predictions[:1], meshes)
    broken = [dict(predictions[0], confidence=1.5)] + predictions[1:]
    with pytest.raises(ValueError):
        validate_predictions(broken, meshes)
    unresolved_with_type = [dict(predictions[0], kind="unresolved")] + predictions[1:]
    with pytest.raises(ValueError):
        validate_predictions(unresolved_with_type, meshes)


def test_written_file_is_strict_json_and_deterministic(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_predictions(DATA_DIR, first)
    write_predictions(DATA_DIR, second)

    text = first.read_text()
    assert text == second.read_text()
    loaded = json.loads(text, parse_constant=lambda name: pytest.fail(f"non-finite {name}"))
    assert len(loaded["predictions"]) == 43


def test_cli_writes_the_file(tmp_path, capsys):
    output = tmp_path / "predictions.json"

    assert main([str(DATA_DIR), str(output)]) == 0

    assert output.exists()
    assert "43 segments" in capsys.readouterr().out


def test_cli_prints_a_table_and_filters_rows(tmp_path, capsys):
    output = tmp_path / "predictions.json"

    assert main([str(DATA_DIR), str(output), "--mesh", "mesh_09", "-v"]) == 0

    out = capsys.readouterr().out
    assert "mesh_09" in out and "unresolved" in out and "reason:" in out
    assert "mesh_02" not in out
    assert len(json.loads(output.read_text())["predictions"]) == 43  # filter is display-only


def test_cli_no_write_and_quiet(tmp_path, capsys):
    output = tmp_path / "predictions.json"

    assert main([str(DATA_DIR), str(output), "--no-write", "-q"]) == 0

    assert not output.exists()
    assert capsys.readouterr().out.strip().startswith("43 segments:")


@pytest.mark.parametrize("args", [["/nonexistent"], [str(DATA_DIR), "--mesh", "mesh_99"]])
def test_cli_reports_bad_arguments_without_a_traceback(args, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(args)

    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
