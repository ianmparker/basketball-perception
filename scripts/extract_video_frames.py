"""
Extract sample frames from Tarek model output videos to show ball detection results
"""
import cv2
from pathlib import Path
import os

output_dir = Path("outputs")
frame_output_dir = Path("outputs/sample_frames_for_viewing")
frame_output_dir.mkdir(exist_ok=True)

def extract_frames(video_path, output_prefix, num_frames=5):
    """Extract evenly-spaced frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"❌ Could not read video: {video_path}")
        return
    
    print(f"\n📹 {video_path.name}")
    print(f"   Total frames: {total_frames}")
    
    indices = [int(total_frames * i / (num_frames - 1)) for i in range(num_frames)]
    
    for idx, frame_num in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Resize for web viewing (max 1280 wide)
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            
            out_path = frame_output_dir / f"{output_prefix}_{idx:02d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"   ✓ Frame {frame_num:3d} → {out_path.name}")
    
    cap.release()

# Extract from key videos
print("🎬 EXTRACTING FRAMES FROM TAREK MODEL OUTPUTS")
print("=" * 70)

videos_to_extract = [
    ("outputs/phase7b_v5_broadcast_74.mp4", "broadcast_74_tarek"),
    ("outputs/phase7b_v5_broadcast_104.mp4", "broadcast_104_tarek"),
    ("outputs/pipeline_broadcast_74.mp4", "broadcast_74_pipeline"),
]

for video_path, prefix in videos_to_extract:
    video_full = Path(video_path)
    if video_full.exists():
        extract_frames(video_full, prefix, num_frames=4)
    else:
        print(f"\n❌ Video not found: {video_path}")

print("\n" + "=" * 70)
print("✓ Frames extracted to: outputs/sample_frames_for_viewing/")
print("  You can view these PNG files directly in any image viewer")
