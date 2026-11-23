## NSL Character Classifier using trained CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


class PadToSquare:
    """Pad the shorter dimension to make image square, preserving aspect ratio."""
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            # Tensor: (C, H, W)
            h, w = img.shape[-2:]

        if w == h:
            return img

        max_dim = max(w, h)
        padding = [0, 0, 0, 0]  # left, top, right, bottom

        if w < max_dim:
            diff = max_dim - w
            padding[0] = diff // 2
            padding[2] = diff - padding[0]
        elif h < max_dim:
            diff = max_dim - h
            padding[1] = diff // 2
            padding[3] = diff - padding[1]

        return transforms.functional.pad(img, padding, fill=self.fill)

# Model architecture (must match training)
class CNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Class names mapping (0-35 to Devanagari characters)
# From dataset: Birat-Poudel/Nepali-Sign-Language-Character-Dataset
CLASS_NAMES = {
    0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ', 5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
    10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण', 15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
    20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म', 25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
    30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ'
}


class NSLClassifier:
    def __init__(self, model_path="notebooks/best_model.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = CNN(num_classes=36).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Preprocessing transform (matches training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            PadToSquare(fill=0),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        print(f"NSL Classifier loaded on {self.device}")

    def preprocess_crop(self, crop):
        """Preprocess hand crop for model input."""
        # crop is BGR numpy array from OpenCV
        # Convert to RGB for PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb)
        return tensor.unsqueeze(0).to(self.device)  # Add batch dimension

    def predict(self, crop):
        """Predict NSL character from hand crop."""
        with torch.no_grad():
            input_tensor = self.preprocess_crop(crop)
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            class_idx = predicted.item()
            conf = confidence.item()
            char = CLASS_NAMES.get(class_idx, str(class_idx))

            return class_idx, char, conf, probabilities.squeeze().cpu().numpy()

    def get_hand_crop(self, frame, hand_landmarks, margin_percent=0.2):
        """Extract hand crop from frame using landmarks with percentage-based margin."""
        h, w = frame.shape[:2]

        # Get bounding box from landmarks (normalized 0-1)
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]

        x_min_norm = min(x_coords)
        x_max_norm = max(x_coords)
        y_min_norm = min(y_coords)
        y_max_norm = max(y_coords)

        # Calculate box width/height in normalized coordinates
        box_w_norm = x_max_norm - x_min_norm
        box_h_norm = y_max_norm - y_min_norm

        # Expand by margin_percent (e.g., 0.2 = 20%) on each side
        margin_x = box_w_norm * margin_percent
        margin_y = box_h_norm * margin_percent

        x_min = int((x_min_norm - margin_x) * w)
        x_max = int((x_max_norm + margin_x) * w)
        y_min = int((y_min_norm - margin_y) * h)
        y_max = int((y_max_norm + margin_y) * h)

        # Clamp to frame boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Crop
        crop = frame[y_min:y_max, x_min:x_max]

        return crop, (x_min, y_min, x_max, y_max)