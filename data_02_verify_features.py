"""
============================================================
Step 1: Verify Extracted Visual Features
============================================================

Run this AFTER step1_extract_features.py to confirm everything worked.

This script checks:
    1. How many .npy feature files were created per split
    2. That all feature shapes are valid (N_frames, 512)
    3. Coverage: what % of annotation entries have matching features
    4. Feature statistics: norms, dimensions, and value ranges
    5. Saves a few sample frames for visual inspection

USAGE:
    python step1_verify_features.py
    python step1_verify_features.py --features-dir features --dataset-dir vatex

============================================================
"""

import os
import argparse
import numpy as np
from datasets import load_from_disk


# Default paths (same as the extraction script)
DEFAULT_FEATURES_DIR = "features"
DEFAULT_DATASET_DIR = "vatex"

# Which splits to verify
SPLITS = ["train", "validation", "public_test"]


def verify_split(split_name, features_dir, dataset_dir):
    """
    Verify the extracted features for a single split.

    Checks:
    - File count and shapes
    - Coverage against annotations
    - Feature value statistics

    Args:
        split_name:    Name of the split (e.g., "train")
        features_dir:  Path to the features root directory
        dataset_dir:   Path to the VATEX dataset root

    Returns:
        dict with verification results
    """
    print(f"\n{'='*50}")
    print(f"  Verifying: {split_name}")
    print(f"{'='*50}")

    split_features_dir = os.path.join(features_dir, split_name)

    # -----------------------------------------------------------
    # 1. Check if the features directory exists
    # -----------------------------------------------------------
    if not os.path.exists(split_features_dir):
        print(f"  [ERROR] Features directory not found: {split_features_dir}")
        print(f"  Run step1_extract_features.py first!")
        return None

    # -----------------------------------------------------------
    # 2. Count .npy files
    # -----------------------------------------------------------
    npy_files = [f for f in os.listdir(split_features_dir) if f.endswith(".npy")]
    print(f"\n  Feature files found: {len(npy_files)}")

    if len(npy_files) == 0:
        print(f"  [ERROR] No .npy files found! Extraction may have failed.")
        return None

    # -----------------------------------------------------------
    # 3. Check shapes and collect statistics
    # -----------------------------------------------------------
    print(f"\n  Checking feature shapes and statistics...")

    shapes = []           # List of (num_frames, feature_dim) for each file
    frame_counts = []     # Number of frames per video
    all_norms = []        # L2 norms of feature vectors
    invalid_files = []    # Files with unexpected shapes

    for npy_file in npy_files:
        filepath = os.path.join(split_features_dir, npy_file)
        try:
            # Load the feature array
            features = np.load(filepath)
            shapes.append(features.shape)

            # Check that the shape is valid: (N, 512)
            if len(features.shape) != 2 or features.shape[1] != 512:
                invalid_files.append((npy_file, features.shape))
                continue

            # Track frame counts
            frame_counts.append(features.shape[0])

            # Compute L2 norms of feature vectors
            # CLIP features should have norms close to 1.0 (they're normalized)
            norms = np.linalg.norm(features, axis=1)
            all_norms.extend(norms.tolist())

        except Exception as e:
            invalid_files.append((npy_file, str(e)))

    # --- Print shape statistics ---
    if frame_counts:
        print(f"\n  Shape Statistics:")
        print(f"    Feature dimension:   512 (all files)")
        print(f"    Min frames/video:    {min(frame_counts)}")
        print(f"    Max frames/video:    {max(frame_counts)}")
        print(f"    Mean frames/video:   {np.mean(frame_counts):.1f}")
        print(f"    Median frames/video: {np.median(frame_counts):.1f}")

    # --- Print norm statistics ---
    if all_norms:
        norms_array = np.array(all_norms)
        print(f"\n  Feature Norm Statistics:")
        print(f"    Mean norm:  {norms_array.mean():.4f}")
        print(f"    Std norm:   {norms_array.std():.4f}")
        print(f"    Min norm:   {norms_array.min():.4f}")
        print(f"    Max norm:   {norms_array.max():.4f}")

        # CLIP features are typically L2-normalized, so norms should be ~1.0
        # If they're not, the features may still be valid (unnormalized)
        if norms_array.mean() > 0.5:
            print(f"    ✅ Norms look healthy")
        else:
            print(f"    ⚠️  Norms seem low — features may need normalization")

    # --- Report invalid files ---
    if invalid_files:
        print(f"\n  ⚠️  Invalid files: {len(invalid_files)}")
        for filename, issue in invalid_files[:5]:  # Show first 5
            print(f"    - {filename}: {issue}")

    # -----------------------------------------------------------
    # 4. Check coverage against annotations
    # -----------------------------------------------------------
    print(f"\n  Checking coverage against annotations...")

    annotations_dir = os.path.join(dataset_dir, "json")
    if os.path.exists(annotations_dir):
        try:
            dataset = load_from_disk(annotations_dir)
            if split_name in dataset:
                split_data = dataset[split_name]
                total_annotations = len(split_data)

                # Count how many annotations have matching feature files
                matched = 0
                for entry in split_data:
                    video_id = entry["videoID"]
                    feature_file = os.path.join(
                        split_features_dir, f"{video_id}.npy"
                    )
                    if os.path.exists(feature_file):
                        matched += 1

                coverage = (matched / total_annotations) * 100

                print(f"    Total annotations:    {total_annotations}")
                print(f"    Matched features:     {matched}")
                print(f"    Coverage:             {coverage:.1f}%")

                if coverage >= 90:
                    print(f"    ✅ Coverage looks good!")
                elif coverage >= 50:
                    print(f"    ⚠️  Coverage is moderate — some videos may be missing")
                else:
                    print(f"    ❌ Coverage is low — check your video downloads")
            else:
                print(f"    [Skip] Split '{split_name}' not found in annotations")
        except Exception as e:
            print(f"    [Error] Could not load annotations: {e}")
    else:
        print(f"    [Skip] Annotations directory not found: {annotations_dir}")

    # -----------------------------------------------------------
    # 5. Check missing videos log
    # -----------------------------------------------------------
    missing_log = os.path.join(split_features_dir, "missing_videos.txt")
    if os.path.exists(missing_log):
        with open(missing_log, "r") as f:
            missing_count = sum(1 for _ in f)
        print(f"\n  Missing videos log: {missing_count} entries")
    else:
        print(f"\n  No missing videos log found (good — all videos were found!)")

    # -----------------------------------------------------------
    # 6. Show a sample
    # -----------------------------------------------------------
    if npy_files:
        sample_file = npy_files[0]
        sample_path = os.path.join(split_features_dir, sample_file)
        sample_features = np.load(sample_path)
        print(f"\n  Sample file: {sample_file}")
        print(f"    Shape: {sample_features.shape}")
        print(f"    Dtype: {sample_features.dtype}")
        print(f"    First vector (first 5 dims): {sample_features[0, :5]}")

    return {
        "split": split_name,
        "num_files": len(npy_files),
        "invalid_files": len(invalid_files),
        "mean_frames": np.mean(frame_counts) if frame_counts else 0,
    }


def main():
    """
    Main entry point. Verifies features for all splits.
    """
    parser = argparse.ArgumentParser(
        description="Verify extracted CLIP features for VATEX"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=DEFAULT_FEATURES_DIR,
        help=f"Path to features directory (default: {DEFAULT_FEATURES_DIR})",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to VATEX dataset (default: {DEFAULT_DATASET_DIR})",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  CVT Thesis — Step 1 Verification")
    print("=" * 50)

    # Verify each split
    results = []
    for split_name in SPLITS:
        result = verify_split(split_name, args.features_dir, args.dataset_dir)
        if result:
            results.append(result)

    # -----------------------------------------------------------
    # Print overall summary
    # -----------------------------------------------------------
    if results:
        print(f"\n{'='*50}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*50}")
        print(f"\n  {'Split':<15} {'Files':<10} {'Avg Frames':<12} {'Issues':<10}")
        print(f"  {'-'*47}")
        for r in results:
            print(
                f"  {r['split']:<15} {r['num_files']:<10} "
                f"{r['mean_frames']:<12.1f} {r['invalid_files']:<10}"
            )
        total_files = sum(r["num_files"] for r in results)
        print(f"\n  Total feature files: {total_files}")
        print(f"\n  ✅ Verification complete!")
    else:
        print(f"\n  ❌ No features found to verify.")
        print(f"     Run step1_extract_features.py first!")

    print()


if __name__ == "__main__":
    main()
