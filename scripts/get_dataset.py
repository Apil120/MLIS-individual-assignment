
import json
import requests

from pathlib import Path
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "data" / "yolo_hand"

NDJSON_PATH = DATASET_DIR / "egohands-egocentric.ndjson"

OUTPUT_DIR = DATASET_DIR / "dataset"

SPLIT_MAPPING = {
    "train": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test"
}

TIMEOUT = 60


def create_directories():
    """
    Creates the directories where the images and labels will be stored for each split (train, val, test).
    """

    for split in ["train", "val", "test"]:

        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True, exist_ok=True
        )

        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )



def download_image(session, url, image_path):
    """"
    Downlaods images from the url in the ndjson file
    """

    if image_path.exists() and image_path.stat().st_size > 0:
        return True

    try:

        response = session.get(
            url,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        # Avoid leaving partially downloaded files
        temp_path = image_path.with_suffix(
            image_path.suffix + ".tmp"
        )

        temp_path.write_bytes(response.content)
        temp_path.replace(image_path)

        return True

    except requests.RequestException as error:

        print(f"\nDownload failed: {image_path.name}")
        print(f"Reason: {error}")

        return False



def convert_annotations(boxes, label_path):

    with open(label_path, "w", encoding="utf-8") as file:

        for box in boxes:

            if len(box) != 5:
                continue

            _, x_center, y_center, width, height = box

            # Merge all four hand classes into class 0
            class_id = 0

            file.write(
                f"{class_id} "
                f"{x_center} "
                f"{y_center} "
                f"{width} "
                f"{height}\n"
            )



def create_data_yaml():

    yaml_content = """path: .
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: hand
"""

    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8"
    )



def process_dataset():

    if not NDJSON_PATH.exists():

        raise FileNotFoundError(
            f"\nNDJSON file not found:\n{NDJSON_PATH}\n"
            "Place egohands-egocentric.ndjson inside data/yolo_hand/"
        )

    create_directories()

    session = requests.Session()

    processed = 0
    downloaded = 0
    failed = 0
    skipped = 0

    try:

        with open(
            NDJSON_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            for line in tqdm(file, desc="Processing EgoHands"):

                if not line.strip():
                    continue

                data = json.loads(line)

                # Ignore dataset metadata
                if data.get("type") != "image":
                    continue

                filename = data["file"]

                split = SPLIT_MAPPING.get(
                    data.get("split")
                )

                if split is None:
                    print(
                        f"\nUnknown split for {filename}: "
                        f"{data.get('split')}"
                    )
                    skipped += 1
                    continue

                image_path = (
                    OUTPUT_DIR / "images" / split / filename
                )

                label_path = (
                    OUTPUT_DIR / "labels" / split /
                    f"{Path(filename).stem}.txt"
                )

                # Download image
                was_existing = (
                    image_path.exists()
                    and image_path.stat().st_size > 0
                )

                success = download_image(
                    session,
                    data["url"],
                    image_path
                )

                if not success:
                    failed += 1
                    continue

                if not was_existing:
                    downloaded += 1

                # Extract bounding boxes
                boxes = (
                    data.get("annotations", {})
                    .get("boxes", [])
                )

                convert_annotations(
                    boxes,
                    label_path
                )

                processed += 1

    finally:

        session.close()

    create_data_yaml()



    print("\n" + "=" * 45)
    print("EGOHANDS DATASET PREPARATION COMPLETE")
    print("=" * 45)

    print(f"Processed images : {processed}")
    print(f"New downloads    : {downloaded}")
    print(f"Failed downloads : {failed}")
    print(f"Skipped records  : {skipped}")

    print(f"\nDataset location : {OUTPUT_DIR}")
    print(f"YOLO config      : {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":

    process_dataset()
