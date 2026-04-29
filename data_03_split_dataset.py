"""
============================================================
Step 1b: Split Available Data into Train / Val / Test
============================================================

Since the original VATEX validation and test videos could not be
downloaded (YouTube 403 errors), this script creates custom splits
from the available training data.

WHAT IT DOES:
    1. Finds all videos that have BOTH:
       - Extracted CLIP features (.npy files)
       - Annotations with English + Chinese captions
    2. Splits them into train/val/test (80%/10%/10%)
    3. Saves the split assignments as a JSON file

The split is done BY VIDEO ID (not by annotation), so all
captions for the same video stay in the same split.

OUTPUT:
    data_splits.json — contains:
    {
        "train": ["videoID1", "videoID2", ...],
        "val":   ["videoID3", ...],
        "test":  ["videoID4", ...],
        "metadata": { ... stats ... }
    }

USAGE:
    python step1b_split_dataset.py
    python step1b_split_dataset.py --seed 42
    python step1b_split_dataset.py --train-ratio 0.8 --val-ratio 0.1

============================================================
"""

import os
import json
import argparse
import numpy as np
from datasets import load_from_disk


# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_FEATURES_DIR = "features"
DEFAULT_DATASET_DIR = "vatex"
DEFAULT_OUTPUT_FILE = "data_splits.json"

# Split ratios (must sum to 1.0)
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO = 0.10
# Test ratio = 1.0 - train - val = 0.10

# Random seed for reproducibility
DEFAULT_SEED = 42


def find_usable_videos(features_dir, dataset_dir):
    """
    Find all videos that have BOTH:
    1. An extracted .npy feature file
    2. Annotations with non-empty English AND Chinese captions

    Returns:
        usable_ids: List of videoIDs that are ready for training
        stats: Dictionary with counts for reporting
    """
    print("\n[1/3] Finding usable videos...")

    # -----------------------------------------------------------
    # Step A: Get all videoIDs that have .npy feature files
    # -----------------------------------------------------------
    train_features_dir = os.path.join(features_dir, "train")
    feature_files = [f for f in os.listdir(train_features_dir) if f.endswith(".npy")]

    # Extract videoID from filename (remove .npy extension)
    feature_ids = set()
    for f in feature_files:
        video_id = f.replace(".npy", "")
        feature_ids.add(video_id)

    print(f"  Feature files found: {len(feature_ids)}")

    # -----------------------------------------------------------
    # Step B: Get all videoIDs from annotations that have captions
    # -----------------------------------------------------------
    annotations_dir = os.path.join(dataset_dir, "json")
    dataset = load_from_disk(annotations_dir)
    train_data = dataset["train"]

    annotation_ids = {}  # videoID -> annotation entry index
    for i in range(len(train_data)):
        entry = train_data[i]
        video_id = entry["videoID"]
        en_caps = entry["enCap"]
        ch_caps = entry["chCap"]

        # Only include if we have both English and Chinese captions
        if en_caps and ch_caps and len(en_caps) > 0 and len(ch_caps) > 0:
            annotation_ids[video_id] = i

    print(f"  Annotated videos (with captions): {len(annotation_ids)}")

    # -----------------------------------------------------------
    # Step C: Find the intersection — videos with BOTH
    # -----------------------------------------------------------
    usable_ids = sorted(list(feature_ids & set(annotation_ids.keys())))

    print(f"  Videos with BOTH features + captions: {len(usable_ids)}")

    stats = {
        "total_feature_files": len(feature_ids),
        "total_annotated": len(annotation_ids),
        "usable": len(usable_ids),
        "features_only": len(feature_ids - set(annotation_ids.keys())),
        "annotations_only": len(set(annotation_ids.keys()) - feature_ids),
    }

    return usable_ids, stats


def create_splits(usable_ids, train_ratio, val_ratio, seed):
    """
    Randomly split the usable video IDs into train/val/test.

    Args:
        usable_ids:   List of videoIDs to split
        train_ratio:  Fraction for training (e.g., 0.80)
        val_ratio:    Fraction for validation (e.g., 0.10)
        seed:         Random seed for reproducibility

    Returns:
        splits: dict with "train", "val", "test" lists
    """
    print(f"\n[2/3] Creating splits (seed={seed})...")

    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"  Ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(seed)
    shuffled = usable_ids.copy()
    rng.shuffle(shuffled)

    # Calculate split boundaries
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Split
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val :])

    print(f"  Train: {len(train_ids)} videos")
    print(f"  Val:   {len(val_ids)} videos")
    print(f"  Test:  {len(test_ids)} videos")

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def verify_splits(splits, features_dir, dataset_dir):
    """
    Verify that the splits are valid:
    - No overlap between splits
    - All videos have features and annotations
    - Feature shapes are correct
    """
    print(f"\n[3/3] Verifying splits...")

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Check no overlap
    assert len(train_set & val_set) == 0, "Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Val/Test overlap!"
    print("  ✅ No overlap between splits")

    # Check all videos have features
    features_train_dir = os.path.join(features_dir, "train")
    all_ids = list(train_set | val_set | test_set)
    missing_features = 0
    bad_shapes = 0

    for vid in all_ids:
        npy_path = os.path.join(features_train_dir, f"{vid}.npy")
        if not os.path.exists(npy_path):
            missing_features += 1
        else:
            features = np.load(npy_path)
            if len(features.shape) != 2 or features.shape[1] != 512:
                bad_shapes += 1

    print(f"  ✅ All {len(all_ids)} videos have feature files" if missing_features == 0
          else f"  ❌ {missing_features} videos missing feature files!")
    print(f"  ✅ All feature shapes are valid (N, 512)" if bad_shapes == 0
          else f"  ❌ {bad_shapes} videos with bad feature shapes!")

    # Check all videos have annotations
    annotations_dir = os.path.join(dataset_dir, "json")
    dataset = load_from_disk(annotations_dir)
    train_data = dataset["train"]

    annotated_ids = set()
    caption_counts = {"en": [], "ch": []}
    for i in range(len(train_data)):
        entry = train_data[i]
        vid = entry["videoID"]
        if vid in (train_set | val_set | test_set):
            annotated_ids.add(vid)
            caption_counts["en"].append(len(entry["enCap"]))
            caption_counts["ch"].append(len(entry["chCap"]))

    missing_annotations = len(all_ids) - len(annotated_ids)
    print(f"  ✅ All {len(all_ids)} videos have annotations" if missing_annotations == 0
          else f"  ❌ {missing_annotations} videos missing annotations!")

    if caption_counts["en"]:
        print(f"  ✅ English captions per video: avg={np.mean(caption_counts['en']):.1f}")
        print(f"  ✅ Chinese captions per video: avg={np.mean(caption_counts['ch']):.1f}")

    return missing_features == 0 and bad_shapes == 0 and missing_annotations == 0


def main():
    parser = argparse.ArgumentParser(
        description="Split available VATEX data into train/val/test"
    )
    parser.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CVT Thesis — Step 1b: Dataset Splitting")
    print("=" * 60)

    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print("[Error] Train + Val ratios exceed 1.0!")
        return

    # Find usable videos
    usable_ids, stats = find_usable_videos(args.features_dir, args.dataset_dir)

    if len(usable_ids) == 0:
        print("\n[Error] No usable videos found!")
        return

    # Create splits
    splits = create_splits(usable_ids, args.train_ratio, args.val_ratio, args.seed)

    # Verify splits
    all_good = verify_splits(splits, args.features_dir, args.dataset_dir)

    # Save the splits
    output_data = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": round(test_ratio, 2),
            "total_usable_videos": len(usable_ids),
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "source": "VATEX original train split (re-split due to missing val/test videos)",
            "features_model": "CLIP ViT-B/32",
            "feature_dim": 512,
            **stats,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total usable videos: {len(usable_ids)}")
    print(f"  Train: {len(splits['train'])} ({args.train_ratio:.0%})")
    print(f"  Val:   {len(splits['val'])} ({args.val_ratio:.0%})")
    print(f"  Test:  {len(splits['test'])} ({round(test_ratio, 2):.0%})")
    print(f"  Saved to: {args.output}")
    if all_good:
        print(f"\n  ✅ All checks passed! Ready for Step 2.")
    else:
        print(f"\n  ⚠️  Some checks failed — review the output above.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
