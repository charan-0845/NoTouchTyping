"""
ASL Alphabet - MediaPipe Landmark Extraction Script
====================================================
Extracts 21 hand landmarks (x, y, z) from each image in the ASL Alphabet dataset
and saves them to a CSV file for model training.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Expected folder structure after download:
    asl_alphabet_train/
        A/  (image files)
        B/
        C/
        ...
        Z/
        space/
        del/
        nothing/

Usage:
    pip install mediapipe opencv-python pandas tqdm
    python extract_landmarks.py --data_dir ./asl_alphabet_train --output landmarks.csv
"""

import os
import csv
import argparse
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # treat each image independently
    max_num_hands=1,              # we only need one hand per image
    min_detection_confidence=0.3, # lower = detect more, but noisier
)

# ── Constants ────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 21
AXES = ["x", "y", "z"]

# Column names: lm0_x, lm0_y, lm0_z, lm1_x, ... lm20_z, label
COLUMNS = [f"lm{i}_{ax}" for i in range(NUM_LANDMARKS) for ax in AXES] + ["label"]


def extract_landmarks_from_image(image_path: str):
    """
    Run MediaPipe Hands on a single image.

    Returns:
        list of 63 floats [lm0_x, lm0_y, lm0_z, ..., lm20_z]
        or None if no hand is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # MediaPipe expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None  # no hand detected

    # Take the first (and only) hand
    hand_landmarks = results.multi_hand_landmarks[0]

    # Flatten: [lm0_x, lm0_y, lm0_z, lm1_x, ...]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    return coords  # length = 63


def normalize_landmarks(coords: list):
    """
    Optional: translate landmarks so wrist (landmark 0) is at origin,
    then scale so the hand fits in a unit bounding box.
    This makes the model robust to hand position and size.
    """
    # Reshape to (21, 3)
    lms = [[coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]]
           for i in range(NUM_LANDMARKS)]

    # Translate: subtract wrist position
    wx, wy, wz = lms[0]
    lms = [[x - wx, y - wy, z - wz] for x, y, z in lms]

    # Scale: divide by the max absolute value so coords are in [-1, 1]
    flat = [v for pt in lms for v in pt]
    max_val = max(abs(v) for v in flat) or 1.0
    flat = [v / max_val for v in flat]

    return flat


def process_dataset(data_dir: str, output_csv: str, normalize: bool = True):
    """
    Walk through every class subfolder, extract landmarks, and save to CSV.
    """
    rows = []
    skipped = 0
    total = 0

    # Each subdirectory is one class (A, B, ..., Z, space, del, nothing)
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    print(f"Found {len(class_dirs)} classes: {class_dirs}\n")

    for label in class_dirs:
        class_path = os.path.join(data_dir, label)
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"Processing class '{label}' — {len(image_files)} images")

        for fname in tqdm(image_files, desc=f"  {label}", leave=False):
            img_path = os.path.join(class_path, fname)
            total += 1

            coords = extract_landmarks_from_image(img_path)

            if coords is None:
                skipped += 1
                continue

            if normalize:
                coords = normalize_landmarks(coords)

            rows.append(coords + [label])

    # Save to CSV
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Total images processed : {total}")
    print(f"  Landmarks extracted    : {len(rows)}")
    print(f"  Skipped (no hand)      : {skipped}")
    print(f"  Detection rate         : {len(rows)/total*100:.1f}%")
    print(f"  Output saved to        : {output_csv}")
    print(f"  CSV shape              : {df.shape}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")

    return df


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from ASL Alphabet dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./asl_alphabet_train",
        help="Path to the asl_alphabet_train folder"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="landmarks.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip landmark normalization (not recommended)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data_dir '{args.data_dir}' not found.")
        print("Download the dataset from Kaggle and unzip it first.")
        exit(1)

    df = process_dataset(
        data_dir=args.data_dir,
        output_csv=args.output,
        normalize=not args.no_normalize,
    )
