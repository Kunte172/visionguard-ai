from __future__ import annotations

import argparse
import base64
import json
from collections.abc import Sequence
from pathlib import Path

from visionguard_ai.data import load_manifest
from visionguard_ai.model import save_training_result, train_detector
from visionguard_ai.sample_data import generate_sample_dataset
from visionguard_ai.service import VisualInspectionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionguard")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-sample", help="Create synthetic image data")
    generate.add_argument("--output", default="data/sample")
    generate.add_argument("--seed", type=int, default=42)

    validate = commands.add_parser("validate", help="Validate an image manifest")
    validate.add_argument("--manifest", default="data/sample/manifest.csv")

    train = commands.add_parser("train", help="Train and evaluate the detector")
    train.add_argument("--manifest", default="data/sample/manifest.csv")
    train.add_argument("--model-output", default="artifacts/detector.joblib")
    train.add_argument("--metrics-output", default="artifacts/metrics.json")
    train.add_argument("--predictions-output", default="artifacts/test_predictions.csv")

    predict = commands.add_parser("predict", help="Inspect one image")
    predict.add_argument("image")
    predict.add_argument("--model", default="artifacts/detector.joblib")
    predict.add_argument("--overlay-output", default="artifacts/prediction_overlay.png")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-sample":
        manifest = generate_sample_dataset(args.output, seed=args.seed)
        print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False))
        return 0
    if args.command == "validate":
        data = load_manifest(args.manifest)
        print(json.dumps({"manifest": args.manifest, "rows": len(data)}, ensure_ascii=False))
        return 0
    if args.command == "train":
        result = train_detector(args.manifest)
        save_training_result(
            result,
            model_path=args.model_output,
            metrics_path=args.metrics_output,
            predictions_path=args.predictions_output,
        )
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        return 0
    if args.command == "predict":
        service = VisualInspectionService(Path(args.model))
        prediction = service.predict_bytes(Path(args.image).read_bytes())
        header, encoded = prediction.heatmap_data_url.split(",", maxsplit=1)
        if header != "data:image/png;base64":
            raise ValueError("Unexpected overlay format")
        Path(args.overlay_output).write_bytes(base64.b64decode(encoded))
        payload = {
            "anomaly_score": prediction.anomaly_score,
            "threshold": prediction.threshold,
            "is_anomaly": prediction.is_anomaly,
            "model_version": prediction.model_version,
            "overlay_output": args.overlay_output,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
