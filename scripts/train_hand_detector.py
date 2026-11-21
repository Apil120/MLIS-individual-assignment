
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent #Project root dir

DATASET_DIR = ROOT / "data" / "yolo_hand" / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

RESULTS_DIR = ROOT / "results" / "metrics"
MODEL_DIR = ROOT / "models" / "hand_detector"

MODEL_NAME = "yolo11n.pt"


def evaluate_on_test(weights_path):
    """Evaluate the final weights on the held-out test split."""

    print("\nEvaluating best weights on the TEST split...\n")

    # Load the saved best weights explicitly so the result never
    # depends on what state the training model object is in.
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        project=str(RESULTS_DIR),
        name="hand_detector_test",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    summary = {
        "split": "test",
        "weights": str(weights_path),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    summary_path = RESULTS_DIR / "hand_detector_test" / "test_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTEST RESULTS")
    print(f"Precision : {summary['precision']:.4f}")
    print(f"Recall    : {summary['recall']:.4f}")
    print(f"mAP50     : {summary['mAP50']:.4f}")
    print(f"mAP50-95  : {summary['mAP50-95']:.4f}")
    print(f"Saved to  : {summary_path}")

    return summary


def validate_dataset():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    for split in ["train", "val", "test"]:

        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(
                f"Missing {split} images or labels directory."
            )

        if not any(image_dir.iterdir()):
            raise RuntimeError(
                f"No images found in {image_dir}"
            )

    DATA_YAML.write_text(
        f"""path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: hand
""",
        encoding="utf-8"
    )


def train_model():

    validate_dataset()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading pretrained YOLO11n model...")

    model = YOLO(MODEL_NAME)

    print("\nStarting hand detection training...\n")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        optimizer="auto",
        patience=20,
        pretrained=True,
        amp=True,
        project=str(RESULTS_DIR),
        name="hand_detector",
        exist_ok=True,
        save=True,
        save_period=10,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
        scale=0.6,               # hands at many distances
        degrees=15,              # rotated fists/thumbs
        translate=0.15,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
        mosaic=1.0,
        close_mosaic=15,         # turn mosaic off for the last epochs
        mixup=0.1,
        conf = 0.05
    )

    best_weights = Path(model.trainer.best)

    if not best_weights.exists():
        raise FileNotFoundError(
            "Best model weights were not generated."
        )

    destination = MODEL_DIR / "best.pt"

    shutil.copy2(best_weights, destination)

    print("\nTRAINING COMPLETE")
    print(f"Best weights: {destination}")
    print(f"Training results: {RESULTS_DIR / 'hand_detector'}")

    evaluate_on_test(destination)



if __name__ == "__main__":
    train_model()
