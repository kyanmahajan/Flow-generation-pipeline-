import numpy as np
import cv2
import os
import matplotlib.cm as cm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "inputs", "my_video.mp4")
FLOW_PATH = os.path.join(SCRIPT_DIR, "outputs", "step5_final_flow", "actionable_flow.npy")
OUTPUT_VIDEO = os.path.join(SCRIPT_DIR, "outputs", "step5_final_flow", "final_actionable_flow.mp4")

# Intrinsics (Must match Step 3/4)
FX, FY = 1000.0, 1000.0
CX, CY = 640.0, 360.0

def visualize_final_flow():
    print("-" * 50)
    print("[Init] Visualizing Final Actionable Flow")
    
    if not os.path.exists(FLOW_PATH):
        print(f"[Error] Flow file not found: {FLOW_PATH}")
        print("Did Step 4 finish successfully?")
        return

    # 1. Load Data
    # Shape is (Points, Frames, 3)
    tracks_3d = np.load(FLOW_PATH)
    print(f"[Info] Loaded {tracks_3d.shape[0]} filtered object points.")

    if tracks_3d.shape[0] == 0:
        print("[Warning] 0 points remained! The mask might have been too strict or the object wasn't detected.")
        return

    # 2. Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 3. Colors (Green for success)
    # We use a gradient from Green to Blue to show structure
    num_tracks = tracks_3d.shape[0]
    colors = cm.winter(np.linspace(0, 1, num_tracks))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    print("[Exec] Rendering video...")
    for t in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if t >= tracks_3d.shape[1]: break

        for i in range(num_tracks):
            x, y, z = tracks_3d[i, t]
            
            if z <= 0.1: continue

            u = int((x / z) * FX + CX)
            v = int((y / z) * FY + CY)

            if 0 <= u < width and 0 <= v < height:
                # BGR Color
                c = (int(colors[i][2]), int(colors[i][1]), int(colors[i][0]))
                cv2.circle(frame, (u, v), 3, c, -1)

        out.write(frame)
        if t % 10 == 0: print(f"       Frame {t}", end='\r')

    cap.release()
    out.release()
    print("\n" + "-" * 50)
    print(f"DONE! Saved to: {OUTPUT_VIDEO}")
    print("-" * 50)

if __name__ == "__main__":
    visualize_final_flow()