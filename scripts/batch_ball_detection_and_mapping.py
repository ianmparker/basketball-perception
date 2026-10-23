import os
import glob
import subprocess
from pathlib import Path

VIDEOS_DIR = 'data/sample/videos/'
CALIBRATION_DIR = 'data/calibration/'
RESULTS_DIR = 'data/results/'
BALL_DETECT_SCRIPT = 'scripts/detect_ball_positions.py'
MAP_SCRIPT = 'scripts/map_ball_to_court.py'
MODEL_PATH = 'models/best.pt'
CONF = '0.25'

# Find all mp4 videos in the videos directory
video_files = glob.glob(os.path.join(VIDEOS_DIR, '*.mp4'))

for video_path in video_files:
    video_name = Path(video_path).stem
    # Try to find the matching calibration JSON
    calib_pattern = f"homography_matrix_{video_name}.json"
    calib_path = os.path.join(CALIBRATION_DIR, calib_pattern)
    if not os.path.exists(calib_path):
        print(f"❌ No calibration found for {video_name}, skipping.")
        continue

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n=== Processing {video_name} ===")
    # Step 1: Ball detection
    ball_json = os.path.join(RESULTS_DIR, f"{video_name}_ball_positions.json")
    if not os.path.exists(ball_json):
        print(f"Running ball detection...")
        subprocess.run([
            'python', BALL_DETECT_SCRIPT,
            '--video', video_path,
            '--model', MODEL_PATH,
            '--conf', CONF,
            '--output', ball_json
        ], check=True)
    else:
        print(f"Ball detection already exists: {ball_json}")

    # Step 2: Map to court coordinates
    court_json = os.path.join(RESULTS_DIR, f"{video_name}_ball_positions_court_coords.json")
    if not os.path.exists(court_json):
        print(f"Mapping to court coordinates...")
        subprocess.run([
            'python', MAP_SCRIPT,
            '--ball-positions', ball_json,
            '--calibration', calib_path
        ], check=True)
    else:
        print(f"Court mapping already exists: {court_json}")

print("\n✅ Batch processing complete.")
