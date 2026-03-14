import numpy as np
import cv2
import os
import matplotlib.cm as cm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "inputs", "my_video.mp4")
FLOW_PATH = os.path.join(SCRIPT_DIR, "outputs", "step5_final_flow", "actionable_flow.npy")
OUTPUT_VIDEO = os.path.join(SCRIPT_DIR, "outputs", "step5_final_flow", "final_flow_trails.mp4")

# Intrinsics
FX, FY = 1000.0, 1000.0
CX, CY = 640.0, 360.0

def select_key_points(tracks_3d, num_points=5):
    """
    Selects the 'num_points' indices that are closest to the object's centroid
    in the first frame. These are usually the most stable points.
    """
    # Get all points at frame 0
    points_t0 = tracks_3d[:, 0, :] # (N, 3)
    
    # Calculate Centroid
    centroid = np.mean(points_t0, axis=0)
    
    # Calculate distance of every point to the centroid
    dists = np.linalg.norm(points_t0 - centroid, axis=1)
    
    # Get indices of the closest points
    # argsort returns indices that would sort the array, we take the first 'num_points'
    key_indices = np.argsort(dists)[:num_points]
    return key_indices

def visualize_trails():
    print("-" * 50)
    print("[Init] Visualizing Trajectories (Trails)...")
    
    if not os.path.exists(FLOW_PATH):
        print(f"[Error] Flow file not found: {FLOW_PATH}")
        return

    # 1. Load Data
    tracks_3d = np.load(FLOW_PATH) # (Points, Frames, 3)
    num_tracks, num_frames, _ = tracks_3d.shape
    print(f"[Info] Loaded {num_tracks} points.")

    # 2. Select the 3 Best Points to Draw Lines For
    key_indices = select_key_points(tracks_3d, num_points=5)
    print(f"[Info] Selected Key Point Indices: {key_indices}")

    # 3. Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Pre-compute distinct colors for the 3 main lines (Red, Green, Blue/Yellow)
    # Format: BGR
    line_colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (0, 255, 255)   # Yellow
    ]

    print("[Exec] Rendering video with trails...")
    
    # Buffer to store 2D history of the key points
    # List of lists: [ [(u,v), (u,v)...],  [(u,v)...] ]
    history_2d = [[] for _ in range(len(key_indices))]

    for t in range(num_frames):
        ret, frame = cap.read()
        if not ret: break

        # A. Draw Faint Dots for ALL points (Context)
        for i in range(0, num_tracks, 2): # stride 2 to save time
            x, y, z = tracks_3d[i, t]
            if z <= 0.1: continue
            u = int((x / z) * FX + CX)
            v = int((y / z) * FY + CY)
            if 0 <= u < width and 0 <= v < height:
                # Faint white/gray dots
                cv2.circle(frame, (u, v), 1, (200, 200, 200), -1)

        # B. Draw Bold Lines for KEY points
        for k_idx, real_idx in enumerate(key_indices):
            x, y, z = tracks_3d[real_idx, t]
            
            if z > 0.1:
                u = int((x / z) * FX + CX)
                v = int((y / z) * FY + CY)
                
                # Add to history
                history_2d[k_idx].append((u, v))

            # Draw the path so far
            path_points = history_2d[k_idx]
            if len(path_points) > 1:
                # Convert list of points to numpy array for polylines
                pts = np.array(path_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Draw the line (thick)
                cv2.polylines(frame, [pts], False, line_colors[k_idx % 3], 3, lineType=cv2.LINE_AA)
                
                # Draw the "head" (current position) as a big circle
                if len(path_points) > 0:
                    curr = path_points[-1]
                    cv2.circle(frame, curr, 6, line_colors[k_idx % 3], -1)

        out.write(frame)
        if t % 10 == 0: print(f"       Frame {t}", end='\r')

    cap.release()
    out.release()
    print("\n" + "-" * 50)
    print(f"DONE! Video saved to: {OUTPUT_VIDEO}")
    print("-" * 50)

if __name__ == "__main__":
    visualize_trails()