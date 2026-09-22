from __future__ import annotations

import json
from pathlib import Path

from visionguard_ai.cli import main


def test_generate_and_validate_commands(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "sample"
    assert main(["generate-sample", "--output", str(output), "--seed", "9"]) == 0
    generated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert Path(generated["manifest"]).is_file()
    assert main(["validate", "--manifest", generated["manifest"]]) == 0
    validated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validated["rows"] == 112


def test_train_and_predict_commands(
    tmp_path: Path,
    trained_project: dict[str, Path | object],
    capsys: object,
) -> None:
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.csv"
    assert (
        main(
            [
                "train",
                "--manifest",
                str(trained_project["manifest"]),
                "--model-output",
                str(model),
                "--metrics-output",
                str(metrics),
                "--predictions-output",
                str(predictions),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["dataset"]["synthetic"]  # type: ignore[attr-defined]

    sample = next((Path(trained_project["root"]) / "sample/test/normal").glob("*.png"))
    overlay = tmp_path / "overlay.png"
    assert (
        main(["predict", str(sample), "--model", str(model), "--overlay-output", str(overlay)]) == 0
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["model_version"] == "patch-statistics-0.1.0"
    assert overlay.is_file()
