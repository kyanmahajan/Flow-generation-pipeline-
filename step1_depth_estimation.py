import os
import sys
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Get the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Add current directory to path so Python finds the 'moge' folder
sys.path.append(SCRIPT_DIR)

# 3. Define Exact Paths
INPUT_VIDEO_PATH = os.path.join(SCRIPT_DIR, "inputs", "my_video.mp4")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# 4. Model Config (Using Large model for your 24GB GPU)
MODEL_NAME = "Ruicheng/moge-2-vitl"

def process_step_1():
    # --- PATH CHECKS ---
    print(f"[Init] Script Location: {SCRIPT_DIR}")
    print(f"[Init] Looking for video at: {INPUT_VIDEO_PATH}")

    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"\n[CRITICAL ERROR] Video file not found!")
        print(f"Please check that '{INPUT_VIDEO_PATH}' exists.")
        return

    # Create Output Directories
    step1_out_dir = os.path.join(OUTPUT_DIR, "step1_depth")
    os.makedirs(step1_out_dir, exist_ok=True)
    
    save_npy_path = os.path.join(step1_out_dir, "raw_depth.npy")
    save_vis_path = os.path.join(step1_out_dir, "depth_visualization.mp4")

    # --- MODEL LOADING ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Hardware: {torch.cuda.get_device_name(0)}")
    print(f"[Info] Loading MoGe Model: {MODEL_NAME}...")
    
    try:
        from moge.model.v2 import MoGeModel
        model = MoGeModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        print("[Success] Model loaded.")
    except ImportError:
        print("[Error] Could not import 'moge'.") 
        print(f"Ensure the 'moge' folder is inside: {SCRIPT_DIR}")
        return
    except Exception as e:
        print(f"[Error] Failed to load weights: {e}")
        return

    # --- VIDEO PROCESSING ---
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[Info] Processing Video: {width}x{height} @ {fps} FPS ({total_frames} frames)")

    # Output Video Writer (Side-by-Side: RGB + Depth)
    out_vis = cv2.VideoWriter(save_vis_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    depth_data = []

    print("[Info] Running Inference...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        # 1. Preprocess (BGR -> RGB -> Tensor)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # 2. Inference
        with torch.no_grad():
            results = model.infer(img_tensor)
            depth_map = results['depth'].squeeze().cpu().numpy()

        # 3. Store Data
        depth_data.append(depth_map)

        # 4. Visualize
        # Clip at 2.5m for contrast (Adjust this if your scene is larger)
        depth_vis = np.clip(depth_map, 0, 2.5) 
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        
        # Resize safe-guard
        if depth_color.shape[:2] != frame.shape[:2]:
            depth_color = cv2.resize(depth_color, (width, height))

        # Stack Side-by-Side
        combined = np.hstack((frame, depth_color))
        out_vis.write(combined)

    cap.release()
    out_vis.release()

    # --- SAVE FINAL DATA ---
    final_depth_stack = np.array(depth_data)
    np.save(save_npy_path, final_depth_stack)

    print("-" * 50)
    print("STEP 1 COMPLETE")
    print(f"1. Raw Data: {save_npy_path}")
    print(f"2. Video:    {save_vis_path}")
    print("-" * 50)

if __name__ == "__main__":
    process_step_1()