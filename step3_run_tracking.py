import os
import sys
import numpy as np
import cv2
import subprocess

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAPIP3D_PATH = os.path.join(SCRIPT_DIR, "TAPIP3D")
CHECKPOINT_PATH = os.path.join(TAPIP3D_PATH, "checkpoints", "tapip3d_final.pth")

# Inputs
VIDEO_PATH = os.path.join(SCRIPT_DIR, "inputs", "my_video.mp4")
DEPTH_PATH = os.path.join(SCRIPT_DIR, "outputs", "step1_depth", "calibrated_depth.npy")

# Outputs
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "step3_tracking")
INPUT_NPZ_PATH = os.path.join(OUTPUT_DIR, "inference_input.npz")

# Default Intrinsics
DEFAULT_INTRINSICS = np.array([
    [1000.0, 0.0,    640.0], 
    [0.0,    1000.0, 360.0], 
    [0.0,    0.0,    1.0]
], dtype=np.float32)

def prepare_data_packet():
    print("-" * 50)
    print(f"[Prep] Preparing Data for Tracker...")
    
    if not os.path.exists(VIDEO_PATH) or not os.path.exists(DEPTH_PATH):
        print(f"[Error] Inputs missing. Check inputs/my_video.mp4 or run Step 2.")
        return False

    # 1. Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    video_arr = np.array(frames)
    T, H, W, _ = video_arr.shape

    # 2. Load Depth
    depth_arr = np.load(DEPTH_PATH)
    
    # --- FIX 1: RESIZE ---
    if depth_arr.shape[1:3] != (H, W) or depth_arr.shape[0] != T:
        print(f"[Info] Resizing depth to match video...")
        min_t = min(T, depth_arr.shape[0])
        video_arr = video_arr[:min_t]
        depth_arr = depth_arr[:min_t]
        
        new_depths = []
        for d in depth_arr:
            new_depths.append(cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST))
        depth_arr = np.array(new_depths)
        T = min_t

    # --- FIX 2: SANITIZE DEPTH (CRITICAL) ---
    print("[Info] Sanitizing depth map (removing NaNs and Infs)...")
    
    # Replace NaNs with 0
    depth_arr = np.nan_to_num(depth_arr, nan=0.0, posinf=10.0, neginf=0.0)
    
    # Clip values to a safe physical range (e.g., 0.1m to 10m)
    # The tracker crashes if depth > 1e9 or depth < 0
    depth_arr = np.clip(depth_arr, 0.1, 10.0)

    # 3. Save Bundle
    intrinsics_arr = np.tile(DEFAULT_INTRINSICS[None], (T, 1, 1))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(INPUT_NPZ_PATH, video=video_arr, depths=depth_arr, intrinsics=intrinsics_arr)
    print(f"[Prep] Data bundled to: {INPUT_NPZ_PATH}")
    return True

def run_tapip3d_inference():
    print("-" * 50)
    print("[Exec] Launching TAPIP3D Tracker...")
    
    inference_script = os.path.join(TAPIP3D_PATH, "inference.py")
    
    if not os.path.exists(inference_script):
        print(f"[Error] Cannot find {inference_script}")
        return

    cmd = [
        sys.executable, inference_script,
        "--input_path", INPUT_NPZ_PATH,
        "--output_dir", OUTPUT_DIR,
        "--checkpoint", CHECKPOINT_PATH,
        "--resolution_factor", "1"
    ]
    
    print(f"[Exec] Running Command: {' '.join(cmd)}")
    print("       (Please wait... this takes 2-5 mins on GPU)")
    print("-" * 50)

    try:
        subprocess.check_call(cmd)
        print("-" * 50)
        print("STEP 3 COMPLETE!")
        print(f"Check outputs in: {OUTPUT_DIR}")
        print("-" * 50)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Tracker crashed with exit code {e.returncode}")

if __name__ == "__main__":
    if prepare_data_packet():
        run_tapip3d_inference()