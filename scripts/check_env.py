"""Smoke test for NSL Recognition environment."""

import sys
import torch
import torchvision
import cv2
import ultralytics
import timm


def main():
    print("=" * 60)
    print("NSL Recognition Environment Check")
    print("=" * 60)

    # 1. Version info
    print("\n1. Package Versions:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   torch: {torch.__version__}")
    print(f"   torchvision: {torchvision.__version__}")
    print(f"   cv2 (opencv): {cv2.__version__}")
    print(f"   ultralytics: {ultralytics.__version__}")
    print(f"   timm: {timm.__version__}")

    # 2. CUDA availability
    print("\n2. CUDA Status:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   Built with CUDA: {torch.version.cuda}")
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"   VRAM: {vram_gb} GB")
    else:
        print("   No GPU detected")

    # 3. Test dataset image loading
    print("\n3. Dataset Image Test:")
    import os
    test_img = "data/raw/NSL/Plain Background/0/0.jpg"
    if os.path.exists(test_img):
        img = cv2.imread(test_img)
        if img is not None:
            print(f"   Image loaded: {test_img}")
            print(f"   Shape: {img.shape} (H, W, C)")
            print(f"   dtype: {img.dtype}")
        else:
            print(f"   Failed to load: {test_img}")
    else:
        print(f"   Image not found: {test_img}")

    # 4. Webcam test
    print("\n4. Webcam Test:")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"   Webcam opened successfully")
            print(f"   Frame shape: {frame.shape}")
        else:
            print(f"   Webcam opened but failed to read frame")
        cap.release()
    else:
        print(f"   Webcam not accessible (index 0)")

    # 5. Ultralytics YOLO test
    print("\n5. Ultralytics YOLO Test:")
    try:
        from ultralytics import YOLO
        # Use yolo11n-seg.pt (nano segmentation model)
        model = YOLO("yolo11n-seg.pt")
        # Run inference on a test image
        results = model(test_img, device=0 if cuda_available else "cpu", verbose=False)
        print(f"   Model loaded: yolo11n-seg.pt")
        print(f"   Inference successful on test image")
        print(f"   Results: {len(results)} result(s)")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Environment check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()