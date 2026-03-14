import numpy as np
import cv2
import os
import glob
import matplotlib.cm as cm

# ==========================================
#      USER CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your original video
VIDEO_PATH = os.path.join(SCRIPT_DIR, "inputs", "my_video.mp4")

# Base Tracking Output Folder
TRACKING_BASE_DIR = os.path.join(SCRIPT_DIR, "outputs", "step3_tracking")

# Output Visualization File
OUTPUT_VIDEO_PATH = os.path.join(TRACKING_BASE_DIR, "tracking_result_fixed.mp4")

# Intrinsics (Approximate)
FX, FY = 1000.0, 1000.0
CX, CY = 640.0, 360.0
# ==========================================

def find_latest_tracks_file():
    if not os.path.exists(TRACKING_BASE_DIR):
        print(f"[Error] Base folder not found: {TRACKING_BASE_DIR}")
        return None
    
    # Check for direct file first
    direct_file = os.path.join(TRACKING_BASE_DIR, "inference_input.result.npz")
    if os.path.exists(direct_file):
        return direct_file

    # Check subfolders
    subdirs = [os.path.join(TRACKING_BASE_DIR, d) for d in os.listdir(TRACKING_BASE_DIR) 
               if os.path.isdir(os.path.join(TRACKING_BASE_DIR, d))]
    
    if subdirs:
        latest_folder = max(subdirs, key=os.path.getmtime)
        target_file = os.path.join(latest_folder, "inference_input.result.npz")
        if os.path.exists(target_file):
            return target_file
            
    print("[Error] Could not find 'inference_input.result.npz'")
    return None

def create_visualization():
    print("-" * 50)
    print(f"[Init] Visualization Script (Shape Auto-Fix)")
    
    # 1. Load Tracks
    tracks_file = find_latest_tracks_file()
    if not tracks_file: return
    print(f"[Load] Loading: {tracks_file}")

    try:
        data = np.load(tracks_file)
        if 'coords' in data:
            raw_tracks = data['coords']
        elif 'tracks' in data:
            raw_tracks = data['tracks']
        else:
            print(f"[Error] Keys found: {list(data.keys())}")
            return
            
        # 2. AUTO-DETECT SHAPE
        # We expect (Points, Time, 3) or (Time, Points, 3)
        # T is usually small (e.g. 40-100), N is large (e.g. 256-1024)
        shape = raw_tracks.shape
        print(f"[Info] Raw Data Shape: {shape}")
        
        # If 4D (Batch, T, N, 3), squeeze batch
        if len(shape) == 4:
            raw_tracks = raw_tracks[0]
            shape = raw_tracks.shape

        # Heuristic: The smaller dimension (that isn't 3) is Time
        dim0, dim1 = shape[0], shape[1]
        
        if dim0 < dim1:
            # Shape is (Time, Points, 3) -> Transpose to (Points, Time, 3)
            print("[Info] Detected (Time, Points, 3). Transposing...")
            tracks_3d = np.transpose(raw_tracks, (1, 0, 2))
        else:
            # Shape is (Points, Time, 3) -> Keep as is
            print("[Info] Detected (Points, Time, 3). Keeping as is.")
            tracks_3d = raw_tracks

    except Exception as e:
        print(f"[Error] Failed to load NPZ: {e}")
        return

    # 3. Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Setup Writer
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 5. Colors
    num_tracks = tracks_3d.shape[0]
    # Reduce number of points drawn if huge (optimization)
    step = 1 if num_tracks < 2000 else num_tracks // 1000
    
    colors = cm.rainbow(np.linspace(0, 1, num_tracks))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    print(f"[Exec] Rendering {num_tracks} points over {tracks_3d.shape[1]} frames...")
    
    for t in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if t >= tracks_3d.shape[1]: break

        # Draw Tracks
        for i in range(0, num_tracks, step):
            x, y, z = tracks_3d[i, t]
            
            # --- PROJECTION FIXES ---
            # If Z is negative, the point is behind camera (ignore)
            if z <= 0.1: continue

            # Standard Pinhole Projection
            u = int((x / z) * FX + CX)
            v = int((y / z) * FY + CY)

            # Draw
            if 0 <= u < width and 0 <= v < height:
                # BGR Color
                c = (int(colors[i][2]), int(colors[i][1]), int(colors[i][0]))
                cv2.circle(frame, (u, v), 2, c, -1)

        out.write(frame)
        if t % 10 == 0:
            print(f"       Frame {t}/{total_frames}", end='\r')

    cap.release()
    out.release()
    print("\n" + "-" * 50)
    print(f"DONE! Video saved to:\n{OUTPUT_VIDEO_PATH}")
    print("-" * 50)

if __name__ == "__main__":
    create_visualization()