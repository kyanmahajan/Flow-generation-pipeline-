import numpy as np
import cv2
import os

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "outputs", "step1_depth", "calibrated_depth.npy")
OUTPUT_VIDEO_PATH = os.path.join(SCRIPT_DIR, "outputs", "step1_depth", "calibrated_depth_viz.mp4")

def create_depth_video():
    print(f"[Init] Loading: {INPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        print("[Error] File not found. Run Step 2 first.")
        return

    # 1. Load Data
    depth_stack = np.load(INPUT_PATH) # (Frames, Height, Width)
    num_frames, height, width = depth_stack.shape

    # 2. Setup Video Writer
    # We generally assume 30 FPS, or you can try to read it from original video if strictly needed
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    print(f"[Info] Rendering {num_frames} frames...")

    for i in range(num_frames):
        d_map = depth_stack[i]
        
        # 3. Visualization Logic
        # Clip max range to 2.0 meters so 0.5m looks bright and distinct
        viz = np.clip(d_map, 0, 2.0)
        
        # Normalize 0-255
        viz = cv2.normalize(viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Color Map
        color_frame = cv2.applyColorMap(viz, cv2.COLORMAP_MAGMA)

        # 4. ADD TEXT OVERLAY (The Proof)
        # We pick the center pixel to show its value in METERS
        cy, cx = height // 2, width // 2
        center_depth_m = d_map[cy, cx]
        
        text = f"Center Depth: {center_depth_m:.3f} m"
        cv2.putText(color_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(color_frame)

    out.release()
    print("-" * 50)
    print(f"VIDEO SAVED: {OUTPUT_VIDEO_PATH}")
    print("Open it and verify the text overlay shows values around 0.5m")
    print("-" * 50)

if __name__ == "__main__":
    create_depth_video()