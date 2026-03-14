import os
import sys
import numpy as np
import torch
import cv2
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GSAM2_PATH = os.path.join(SCRIPT_DIR, "Grounded-SAM-2")
sys.path.append(GSAM2_PATH) # Add to path

# Inputs
VIDEO_PATH = os.path.join(SCRIPT_DIR, "inputs", "my_video.mp4")
TRACKING_DIR = os.path.join(SCRIPT_DIR, "outputs", "step3_tracking")
TEXT_PROMPT = "Green Handle" # <--- CHANGE THIS TO YOUR OBJECT NAME

# Model Paths
GDINO_CHECKPOINT = os.path.join(GSAM2_PATH, "checkpoints", "groundingdino_swint_ogc.pth")
SAM2_CHECKPOINT = os.path.join(GSAM2_PATH, "checkpoints", "sam2_hiera_large.pt")
SAM2_CONFIG = "sam2_hiera_l.yaml" # Config name inside the repo

# Output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "step4_grounding")
FINAL_FLOW_PATH = os.path.join(SCRIPT_DIR, "outputs", "step5_final_flow", "actionable_flow.npy")

# Intrinsics (Must match Step 3)
FX, FY = 1000.0, 1000.0
CX, CY = 640.0, 360.0

def find_latest_tracks():
    # Logic to find the .npz file from Step 3
    if not os.path.exists(TRACKING_DIR): return None
    
    # Check subfolders
    subdirs = [os.path.join(TRACKING_DIR, d) for d in os.listdir(TRACKING_DIR) if os.path.isdir(os.path.join(TRACKING_DIR, d))]
    if subdirs:
        latest = max(subdirs, key=os.path.getmtime)
        target = os.path.join(latest, "inference_input.result.npz")
        if os.path.exists(target): return target
        
    return None

def run_grounded_sam2():
    print("-" * 50)
    print(f"[Init] Step 4: Object Grounding for '{TEXT_PROMPT}'")
    
    # 1. Imports (Delayed to ensure path is added)
    try:
        from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        import grounding_dino.groundingdino.datasets.transforms as T
    except ImportError as e:
        print(f"[Error] Failed to import Grounded-SAM2: {e}")
        print("Did you run 'pip install -e .' inside the Grounded-SAM-2 folder?")
        return None

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    # 3. Load Grounding DINO (Detector)
    print(f"[Info] Loading Grounding DINO...")
    # The config file for GDINO is usually inside the repo
    gdino_config = os.path.join(GSAM2_PATH, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    gdino_model = load_model(gdino_config, GDINO_CHECKPOINT, device=device)

    # 4. Load SAM 2 (Segmenter)
    print(f"[Info] Loading SAM 2...")
    sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)

    # 5. Process Frame 0 for Detection
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame0 = cap.read()
    cap.release()
    if not ret: return None
    
    # GDINO expects image transformed
    # We save a temp frame 0 to load it cleanly via their util
    cv2.imwrite("temp_frame0.jpg", frame0)
    image_source, image = load_image("temp_frame0.jpg")
    
    # Detect Box
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    boxes, logits, phrases = predict(
        model=gdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    if len(boxes) == 0:
        print(f"[Error] No object found for prompt: '{TEXT_PROMPT}'")
        return None

    # Pick best box (highest logit)
    # GDINO returns normalized boxes (cx, cy, w, h). SAM2 needs (x1, y1, x2, y2) absolute.
    h, w, _ = frame0.shape
    best_box = boxes[0] * torch.Tensor([w, h, w, h])
    xyxy = best_box.numpy()
    xyxy[0] -= xyxy[2] / 2
    xyxy[1] -= xyxy[3] / 2
    xyxy[2] += xyxy[0]
    xyxy[3] += xyxy[1]
    
    print(f"[Info] Detected '{TEXT_PROMPT}' at: {xyxy}")

    # 6. Propagate Mask with SAM 2
    print("[Info] Propagating mask across video (SAM 2)...")
    inference_state = sam2_predictor.init_state(video_path=VIDEO_PATH)
    sam2_predictor.reset_state(inference_state)
    
    _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=xyxy
    )

    # Collect masks for all frames
    video_masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
        # mask shape: (1, H, W) -> squeeze to (H, W)
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        video_masks.append(mask)

    print(f"[Info] Generated {len(video_masks)} masks.")
    return np.array(video_masks) # (T, H, W)

def filter_tracks(masks):
    print("[Info] Filtering 3D Tracks...")
    tracks_file = find_latest_tracks()
    if not tracks_file:
        print("[Error] No tracks file found from Step 3.")
        return

    data = np.load(tracks_file)
    if 'coords' in data: raw_tracks = data['coords']
    elif 'tracks' in data: raw_tracks = data['tracks']
    else: return

    # Transpose if needed (Time, Points, 3) -> (Points, Time, 3)
    if raw_tracks.shape[0] < raw_tracks.shape[1]:
        raw_tracks = np.transpose(raw_tracks, (1, 0, 2))

    num_points, num_frames, _ = raw_tracks.shape
    valid_indices = []

    print(f"[Info] Checking {num_points} trajectories...")
    
    for i in range(num_points):
        is_valid = True
        # Check every frame (or strided for speed)
        for t in range(0, num_frames, 2): 
            if t >= len(masks): break
            
            x, y, z = raw_tracks[i, t]
            
            # Project to 2D
            if z <= 0.1: 
                is_valid = False
                break
            
            u = int((x / z) * FX + CX)
            v = int((y / z) * FY + CY)
            
            # Check Mask
            if u < 0 or v < 0 or u >= masks.shape[2] or v >= masks.shape[1]:
                is_valid = False; break
            
            # If mask is False (0), reject
            if not masks[t, v, u]:
                is_valid = False; break
        
        if is_valid:
            valid_indices.append(i)

    # Save Result
    filtered_tracks = raw_tracks[valid_indices]
    print(f"[Result] Retained {len(valid_indices)} / {num_points} points.")
    
    os.makedirs(os.path.dirname(FINAL_FLOW_PATH), exist_ok=True)
    np.save(FINAL_FLOW_PATH, filtered_tracks)
    print(f"[Success] Actionable Flow saved to: {FINAL_FLOW_PATH}")

if __name__ == "__main__":
    masks = run_grounded_sam2()
    if masks is not None:
        filter_tracks(masks)