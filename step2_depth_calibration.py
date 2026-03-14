import numpy as np
import os
import argparse
import sys

# --- CONFIGURATION ---
# Get the absolute path where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# Input: The raw depth you generated in Step 1
INPUT_NPY_PATH = os.path.join(OUTPUT_DIR, "step1_depth", "raw_depth.npy")

# Output: The calibrated depth we will create now
OUTPUT_NPY_PATH = os.path.join(OUTPUT_DIR, "step1_depth", "calibrated_depth.npy")

def calibrate_depth(known_distance):
    print(f"[Init] Script Location: {SCRIPT_DIR}")
    print(f"[Init] Loading Raw Depth from: {INPUT_NPY_PATH}")

    # 1. Check if Step 1 was actually done
    if not os.path.exists(INPUT_NPY_PATH):
        print(f"\n[CRITICAL ERROR] Raw depth file not found!")
        print(f"   Missing: {INPUT_NPY_PATH}")
        print("   Please run 'step1_depth_estimation.py' first.")
        return

    # 2. Load the Raw Data
    try:
        raw_depth_stack = np.load(INPUT_NPY_PATH) # Shape: (Frames, Height, Width)
        print(f"[Info] Loaded depth sequence. Shape: {raw_depth_stack.shape}")
    except Exception as e:
        print(f"[Error] Failed to load .npy file: {e}")
        return

    # 3. Calculate Scale Factor
    # We grab the first frame (Frame 0) to establish our baseline
    first_frame = raw_depth_stack[0]
    
    # Compute Median of the estimated depth
    estimated_median = np.median(first_frame)
    
    # Avoid division by zero
    if estimated_median == 0:
        print("[Error] Estimated median is 0. Cannot calibrate.")
        return

    # The Magic Formula: Scale = Real_Distance / Estimated_Median
    scale_factor = known_distance / estimated_median
    
    print("-" * 40)
    print(f"CALIBRATION STATISTICS:")
    print(f" > Target Real Dist:   {known_distance:.4f} meters")
    print(f" > Model Est. Median:  {estimated_median:.4f} units")
    print(f" > COMPUTED SCALE:     {scale_factor:.6f}")
    print("-" * 40)

    # 4. Apply Scale to the ENTIRE Video
    # This aligns every frame to the real world metric
    calibrated_depth_stack = raw_depth_stack * scale_factor

    # 5. Save Output
    print(f"[Info] Saving calibrated data to: {OUTPUT_NPY_PATH}")
    np.save(OUTPUT_NPY_PATH, calibrated_depth_stack)

    # 6. Verification
    new_median = np.median(calibrated_depth_stack[0])
    print(f"[Verify] New Median of Frame 0 is now: {new_median:.4f} m")
    print("\nSTEP 2 COMPLETE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default is 0.5m as you requested
    parser.add_argument("--dist", type=float, default=0.5, help="Known distance to object in first frame (meters)")
    args = parser.parse_args()
    
    calibrate_depth(args.dist)