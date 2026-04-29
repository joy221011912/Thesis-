"""
============================================================
Step 1: Extract Visual Features from VATEX Videos using CLIP
============================================================

This script is part of the Contextual-Vision Transformer (CVT) thesis.

WHAT IT DOES:
    For every video in the VATEX dataset, this script:
    1. Opens the video file (.mp4)
    2. Samples frames at 1 frame-per-second (FPS)
    3. Passes each frame through a frozen CLIP ViT-B/32 model
    4. Saves the resulting feature vectors as a .npy file

WHY:
    These visual features will later be used in Step 3 (Cross-Attention),
    where the NMT text decoder "looks at" the video to disambiguate
    translations. For example, "bat" could mean 🦇 or 🏏 — the visual
    features help the model choose the correct translation.

OUTPUT:
    features/
    ├── train/
    │   ├── {videoID}.npy          # shape: (num_frames, 512)
    │   └── missing_videos.txt     # log of videos that couldn't be found
    ├── validation/
    │   └── ...
    └── public_test/
        └── ...

USAGE:
    # Extract all splits:
    python step1_extract_features.py

    # Extract only a specific split:
    python step1_extract_features.py --split train

    # Use CPU instead of GPU:
    python step1_extract_features.py --device cpu

    # Limit to N videos (for testing):
    python step1_extract_features.py --limit 10

============================================================
"""

import os
import argparse
import time
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk

# ============================================================
# CLIP is imported here. It must be installed separately:
#   pip install git+https://github.com/openai/CLIP.git
# ============================================================
import clip


# ============================================================
# CONFIGURATION — Default values (can be overridden via CLI)
# ============================================================

# Path to the VATEX dataset folder (contains 'json/' and 'videos/')
# This is relative to where you run the script (your thesis folder).
DEFAULT_DATASET_DIR = "vatex"

# Path where extracted features will be saved.
# Stored OUTSIDE the dataset folder to keep it clean.
DEFAULT_OUTPUT_DIR = "features"

# Which CLIP model to use.
# "ViT-B/32" = Vision Transformer, Base size, 32x32 patch size
# Output dimension: 512 (each frame becomes a 512-dim vector)
CLIP_MODEL_NAME = "ViT-B/32"

# How many frames to extract per second of video.
# 1 FPS is standard for multimodal translation research.
# VATEX clips are ~10 seconds, so this gives ~10 frames per video.
FRAMES_PER_SECOND = 1

# How many frames to process at once through CLIP.
# Higher = faster but uses more memory. 32 is safe for most GPUs.
BATCH_SIZE = 32

# Which dataset splits to process.
SPLITS = ["train", "validation", "public_test"]


def select_device(preferred_device=None):
    """
    Select the best available compute device for running CLIP.

    Priority order:
        1. User's explicit choice (if provided)
        2. CUDA (NVIDIA GPU)
        3. MPS (Apple Silicon GPU — your Mac)
        4. CPU (slowest, always available)

    Args:
        preferred_device: Optional string like "cpu", "cuda", or "mps"

    Returns:
        torch.device object
    """
    if preferred_device:
        # User explicitly chose a device
        device = torch.device(preferred_device)
        print(f"[Device] Using user-specified device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple Silicon GPU (MPS) detected")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU found, using CPU (this will be slower)")

    return device


def load_clip_model(device):
    """
    Load the pre-trained CLIP model and its image preprocessor.

    The model is downloaded automatically on first run (~340MB) and
    cached at ~/.cache/clip/ for future use.

    IMPORTANT: The model is set to eval() mode and we never compute
    gradients. We're using it purely as a feature extractor, NOT
    training it.

    Args:
        device: torch.device to load the model onto

    Returns:
        model: The CLIP model (frozen, eval mode)
        preprocess: The image transform function that CLIP expects
                    (resizes to 224x224, normalizes with specific mean/std)
    """
    print(f"\n[CLIP] Loading {CLIP_MODEL_NAME} model...")
    print(f"[CLIP] First run will download ~340MB to ~/.cache/clip/")

    # clip.load() returns both the model and the preprocessing transform.
    # The preprocess function handles:
    #   - Resize to 224x224 pixels
    #   - Center crop
    #   - Convert to tensor
    #   - Normalize with CLIP's training mean/std values
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

    # Set to evaluation mode — disables dropout, batch norm updates, etc.
    # This is critical: we want deterministic, consistent features.
    model.eval()

    print(f"[CLIP] Model loaded successfully on {device}")
    print(f"[CLIP] Output feature dimension: {model.visual.output_dim}")

    return model, preprocess


def extract_frames_from_video(video_path, fps_target=1):
    """
    Open a video file and sample frames at a target frame rate.

    This function reads a video using OpenCV and extracts frames
    at evenly-spaced intervals to achieve the target FPS.

    For example, if the video is 30 FPS and we want 1 FPS,
    we take every 30th frame.

    Args:
        video_path: Absolute path to the .mp4 file
        fps_target: How many frames per second to extract (default: 1)

    Returns:
        frames: List of PIL Image objects (RGB format)
                Returns empty list if video can't be opened
    """
    frames = []

    # Open the video file using OpenCV's VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"  [Warning] Cannot open video: {video_path}")
        return frames

    # Get the video's native frame rate (e.g., 24, 25, or 30 FPS)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle edge case: invalid FPS metadata
    if video_fps <= 0 or total_frames <= 0:
        print(f"  [Warning] Invalid video metadata: fps={video_fps}, frames={total_frames}")
        cap.release()
        return frames

    # Calculate how many native frames to skip between each sample.
    # Example: if video is 30 FPS and we want 1 FPS → skip every 30 frames
    frame_interval = max(1, int(video_fps / fps_target))

    # Calculate which frame indices to extract
    # Example: for a 300-frame video at 30 FPS, with 1 FPS target:
    #   frame_indices = [0, 30, 60, 90, 120, ..., 270]
    frame_indices = list(range(0, total_frames, frame_interval))

    for frame_idx in frame_indices:
        # Seek to the target frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()

        # ret is False if the frame couldn't be read (e.g., end of file)
        if not ret:
            break

        # OpenCV reads frames in BGR color format, but CLIP/PIL expects RGB.
        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL Image (CLIP's preprocessor expects PIL)
        pil_image = Image.fromarray(frame_rgb)

        frames.append(pil_image)

    # Release the video file handle
    cap.release()

    return frames


def extract_features_from_frames(frames, model, preprocess, device, batch_size=32):
    """
    Run a list of PIL Image frames through CLIP's image encoder
    to get feature vectors.

    Each frame is:
    1. Preprocessed (resize, crop, normalize) using CLIP's transform
    2. Passed through the frozen ViT-B/32 image encoder
    3. Converted to a 512-dimensional feature vector

    We process frames in batches the same way you'd batch inputs
    for any neural network — to maximize GPU utilization.

    Args:
        frames:     List of PIL Image objects
        model:      The loaded CLIP model
        preprocess: CLIP's image preprocessing transform
        device:     torch.device (cpu/cuda/mps)
        batch_size: Number of frames to process at once

    Returns:
        features: NumPy array of shape (num_frames, 512)
                  Each row is the CLIP feature vector for one frame
    """
    all_features = []

    # Process frames in batches to avoid running out of memory.
    # For example, if we have 10 frames and batch_size=32,
    # we process all 10 at once (one batch).
    for i in range(0, len(frames), batch_size):
        # Get the current batch of frames
        batch_frames = frames[i : i + batch_size]

        # Apply CLIP's preprocessing to each frame and stack into a tensor.
        # preprocess() converts each PIL Image to a tensor of shape (3, 224, 224)
        # torch.stack() combines them into shape (batch_size, 3, 224, 224)
        batch_tensor = torch.stack([preprocess(frame) for frame in batch_frames])

        # Move the tensor to the compute device (GPU/MPS/CPU)
        batch_tensor = batch_tensor.to(device)

        # Run the batch through CLIP's image encoder.
        # torch.no_grad() tells PyTorch we don't need gradients — we're
        # not training, just extracting features. This saves memory and time.
        with torch.no_grad():
            # model.encode_image() passes the images through the Vision Transformer
            # and returns the CLS token embedding for each image.
            # Output shape: (batch_size, 512)
            batch_features = model.encode_image(batch_tensor)

        # Move features back to CPU and convert to NumPy.
        # We store features as NumPy arrays because:
        # 1. They're framework-agnostic (can be loaded without PyTorch)
        # 2. .npy files are compact and fast to load
        # 3. The NMT model in Step 2-3 will convert them back to tensors
        all_features.append(batch_features.cpu().numpy())

    # Concatenate all batch results into a single array
    # Final shape: (total_num_frames, 512)
    features = np.concatenate(all_features, axis=0)

    return features


def resolve_video_path(video_dir, video_id, path_field):
    """
    Find the actual .mp4 file on disk for a given VATEX annotation entry.

    VATEX annotations have:
    - videoID: "YouTubeID_StartTime_EndTime" (e.g., "xIEG3R-L1PA_42_52")
    - path: Usually just the YouTube ID (e.g., "xIEG3R-L1PA")

    The actual video files on disk are named: {YouTubeID}.mp4

    This function tries multiple naming conventions to find the file.

    Args:
        video_dir:  Path to the directory containing .mp4 files
        video_id:   The videoID field from the annotation
        path_field: The path field from the annotation

    Returns:
        Path to the .mp4 file, or None if not found
    """
    # Strategy 1: Use the 'path' field directly (most reliable)
    # The path field usually contains just the YouTube ID
    candidate = os.path.join(video_dir, f"{path_field}.mp4")
    if os.path.exists(candidate):
        return candidate

    # Strategy 2: The videoID might itself be a valid filename
    candidate = os.path.join(video_dir, f"{video_id}.mp4")
    if os.path.exists(candidate):
        return candidate

    # Strategy 3: Parse YouTube ID from videoID by removing _start_end suffix
    # videoID format: "YouTubeID_StartTime_EndTime"
    # We need to be careful: YouTube IDs can contain underscores and hyphens!
    # So we try removing the last two _number segments
    parts = video_id.rsplit("_", 2)
    if len(parts) == 3:
        youtube_id = parts[0]
        candidate = os.path.join(video_dir, f"{youtube_id}.mp4")
        if os.path.exists(candidate):
            return candidate

    # Could not find the video file
    return None


def process_split(
    split_name, dataset_dir, output_dir, model, preprocess, device, limit=None
):
    """
    Process all videos in a single dataset split (train/validation/public_test).

    This is the main processing loop. For each annotation entry:
    1. Find the corresponding video file
    2. Extract frames at 1 FPS
    3. Run frames through CLIP
    4. Save features as .npy

    Args:
        split_name:  Name of the split ("train", "validation", "public_test")
        dataset_dir: Path to the VATEX dataset root
        output_dir:  Path to save features
        model:       Loaded CLIP model
        preprocess:  CLIP's image preprocessor
        device:      Compute device
        limit:       Optional max number of videos to process (for testing)
    """
    print(f"\n{'='*60}")
    print(f"  Processing split: {split_name}")
    print(f"{'='*60}")

    # -----------------------------------------------------------
    # 1. Load annotations from the HuggingFace Arrow files
    # -----------------------------------------------------------
    annotations_dir = os.path.join(dataset_dir, "json")
    print(f"\n[1/4] Loading annotations from: {annotations_dir}")

    # load_from_disk() reads the HuggingFace Dataset that was saved
    # in Arrow format. This gives us a DatasetDict with our splits.
    dataset = load_from_disk(annotations_dir)

    # Get just the split we want (e.g., dataset["train"])
    if split_name not in dataset:
        print(f"  [Error] Split '{split_name}' not found in dataset!")
        print(f"  Available splits: {list(dataset.keys())}")
        return
    split_data = dataset[split_name]

    print(f"  Found {len(split_data)} annotation entries")

    # -----------------------------------------------------------
    # 2. Create output directory for this split
    # -----------------------------------------------------------
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    print(f"\n[2/4] Output directory: {split_output_dir}")

    # -----------------------------------------------------------
    # 3. Set up the video directory path
    # -----------------------------------------------------------
    video_dir = os.path.join(dataset_dir, "videos")
    print(f"[3/4] Video directory: {video_dir}")

    if not os.path.exists(video_dir):
        print(f"  [Error] Video directory not found!")
        return

    # -----------------------------------------------------------
    # 4. Process each video
    # -----------------------------------------------------------
    print(f"\n[4/4] Extracting features...\n")

    # Apply the limit if specified (useful for testing with --limit 10)
    entries = split_data
    if limit:
        entries = split_data.select(range(min(limit, len(split_data))))
        print(f"  [Limit] Processing only {len(entries)} videos (--limit flag)\n")

    # Counters for tracking progress
    processed = 0       # Successfully processed videos
    skipped_exists = 0  # Skipped because .npy already exists (resume support)
    skipped_missing = 0 # Skipped because video file not found
    skipped_error = 0   # Skipped due to errors (corrupt video, no frames, etc.)
    missing_videos = [] # List of videoIDs we couldn't find

    # Create a progress bar using tqdm
    pbar = tqdm(range(len(entries)), desc=f"  {split_name}", unit="video")

    for idx in pbar:
        # Get the annotation entry for this video
        entry = entries[idx]
        video_id = entry["videoID"]
        path_field = entry["path"]

        # ----- RESUME SUPPORT -----
        # Check if features already exist for this video.
        # This lets you stop and restart the script without
        # re-processing videos that are already done.
        output_path = os.path.join(split_output_dir, f"{video_id}.npy")
        if os.path.exists(output_path):
            skipped_exists += 1
            continue

        # ----- FIND THE VIDEO FILE -----
        video_path = resolve_video_path(video_dir, video_id, path_field)

        if video_path is None:
            # Video file not on disk (YouTube video may have been deleted)
            skipped_missing += 1
            missing_videos.append(video_id)
            continue

        # ----- EXTRACT FRAMES -----
        # Sample frames from the video at 1 FPS
        frames = extract_frames_from_video(video_path, fps_target=FRAMES_PER_SECOND)

        if len(frames) == 0:
            # No frames could be extracted (corrupt or empty video)
            skipped_error += 1
            missing_videos.append(f"{video_id} (no frames)")
            continue

        # ----- EXTRACT CLIP FEATURES -----
        # Pass all sampled frames through CLIP's image encoder
        features = extract_features_from_frames(
            frames, model, preprocess, device, batch_size=BATCH_SIZE
        )

        # ----- SAVE FEATURES -----
        # Save as a NumPy .npy file
        # Shape: (num_frames, 512) — e.g., (10, 512) for a 10-second video
        np.save(output_path, features)

        processed += 1

        # Update progress bar with current stats
        pbar.set_postfix(
            done=processed,
            skip=skipped_exists,
            miss=skipped_missing,
            err=skipped_error,
        )

    # -----------------------------------------------------------
    # 5. Save the missing videos log
    # -----------------------------------------------------------
    if missing_videos:
        missing_log_path = os.path.join(split_output_dir, "missing_videos.txt")
        with open(missing_log_path, "w") as f:
            for vid in missing_videos:
                f.write(f"{vid}\n")
        print(f"\n  [Log] Missing videos saved to: {missing_log_path}")

    # -----------------------------------------------------------
    # 6. Print summary for this split
    # -----------------------------------------------------------
    print(f"\n  --- {split_name} Summary ---")
    print(f"  Total annotations:    {len(entries)}")
    print(f"  Successfully processed: {processed}")
    print(f"  Already existed (skip): {skipped_exists}")
    print(f"  Missing video files:    {skipped_missing}")
    print(f"  Errors (no frames):     {skipped_error}")
    print(f"  Feature files saved to: {split_output_dir}")


def main():
    """
    Main entry point. Parses command-line arguments and runs the pipeline.
    """
    # -----------------------------------------------------------
    # Parse command-line arguments
    # -----------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Extract CLIP visual features from VATEX videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step1_extract_features.py                        # Process all splits
  python step1_extract_features.py --split train          # Only train split
  python step1_extract_features.py --limit 10             # Test with 10 videos
  python step1_extract_features.py --device cpu           # Force CPU
  python step1_extract_features.py --dataset-dir vatex    # Custom dataset path
        """,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to VATEX dataset folder (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to save extracted features (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=SPLITS,
        help="Process only a specific split (default: all splits)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force a specific compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to N videos per split (for testing)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------
    # Print banner
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CVT Thesis — Step 1: Visual Feature Extraction")
    print("  Model: CLIP ViT-B/32 (frozen, feature extraction only)")
    print("  Sampling: 1 frame per second")
    print("  Output: .npy files with shape (num_frames, 512)")
    print("=" * 60)

    # -----------------------------------------------------------
    # Validate dataset directory exists
    # -----------------------------------------------------------
    if not os.path.exists(args.dataset_dir):
        print(f"\n[Error] Dataset directory not found: {args.dataset_dir}")
        print(f"  Make sure 'vatex/' folder is in the same directory as this script.")
        return

    # -----------------------------------------------------------
    # Select compute device (GPU/MPS/CPU)
    # -----------------------------------------------------------
    device = select_device(args.device)

    # -----------------------------------------------------------
    # Load CLIP model
    # -----------------------------------------------------------
    model, preprocess = load_clip_model(device)

    # -----------------------------------------------------------
    # Create the main output directory
    # -----------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[Output] Features will be saved to: {args.output_dir}/")

    # -----------------------------------------------------------
    # Determine which splits to process
    # -----------------------------------------------------------
    if args.split:
        # User specified a single split
        splits_to_process = [args.split]
    else:
        # Process all splits
        splits_to_process = SPLITS

    print(f"[Splits] Will process: {splits_to_process}")

    # -----------------------------------------------------------
    # Start the extraction!
    # -----------------------------------------------------------
    start_time = time.time()

    for split_name in splits_to_process:
        process_split(
            split_name=split_name,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            model=model,
            preprocess=preprocess,
            device=device,
            limit=args.limit,
        )

    # -----------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"  DONE! Total time: {minutes}m {seconds}s")
    print(f"  Features saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


# This block runs only when you execute the script directly:
#   python step1_extract_features.py
# It does NOT run when the file is imported as a module.
if __name__ == "__main__":
    main()
