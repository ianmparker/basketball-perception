"""
Inference Script for Learned Homography

Automatically predicts homography matrix for new basketball videos.
No manual clicking required!

Usage:
    python predict_homography.py --video game.mp4 --model best.pt
    
Output:
    homography_matrix_game.json (same format as manual calibration)
"""

import torch
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import LearnedHomography


def get_inference_transform():
    """Transform pipeline for inference (no augmentation)."""
    return A.Compose([
        A.Resize(height=720, width=1280),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def predict_homography(video_path: str,
                       model_path: str,
                       device: str = 'auto',
                       frame_idx: int = None) -> dict:
    """
    Predict homography matrix for a video.
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model checkpoint
        device: Device to use ('cuda', 'cpu', or 'auto')
        frame_idx: Frame index to use (None = middle frame)
    
    Returns:
        result: Dictionary containing:
            - homography_matrix: 3x3 matrix as list
            - video_path: Input video path
            - frame_number: Frame used for prediction
            - model_path: Model used for prediction
    """
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = LearnedHomography(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation RMSE: {checkpoint.get('rmse', 'N/A'):.2f} ft")
    
    # Load video
    print(f"\nLoading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    
    # Select frame to use
    if frame_idx is None:
        frame_idx = total_frames // 2  # Use middle frame
    
    print(f"  Using frame: {frame_idx}")
    
    # Extract frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from video")
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Prepare for model
    transform = get_inference_transform()
    transformed = transform(image=frame_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict homography
    print("\nPredicting homography matrix...")
    with torch.no_grad():
        H = model(image_tensor)
        H = H.squeeze(0).cpu().numpy()
    
    print("✓ Prediction complete!")
    print(f"\nHomography matrix:")
    print(H)
    
    # Prepare result
    result = {
        'homography_matrix': H.tolist(),
        'video_path': str(video_path),
        'frame_number': frame_idx,
        'model_path': str(model_path),
        'model_rmse': float(checkpoint.get('rmse', 0.0)),
        'video_info': {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height
        }
    }
    
    return result


def visualize_homography(video_path: str,
                        homography_matrix: np.ndarray,
                        output_path: str = None) -> None:
    """
    Visualize homography by overlaying court lines on video frame.
    
    Args:
        video_path: Path to video file
        homography_matrix: 3x3 homography matrix
        output_path: Path to save visualization (None = display only)
    """
    from visualization.draw_court import draw_court_lines
    
    # Load video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("WARNING: Cannot visualize - frame read failed")
        return
    
    # Draw court overlay
    H_inv = np.linalg.inv(homography_matrix)
    overlay = draw_court_lines(frame.copy(), H_inv)
    
    # Display or save
    if output_path:
        cv2.imwrite(output_path, overlay)
        print(f"✓ Visualization saved to: {output_path}")
    else:
        cv2.imshow('Homography Visualization', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict homography for basketball video")
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (best.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: homography_matrix_<video_name>.json)')
    parser.add_argument('--frame', type=int, default=None,
                       help='Frame index to use (default: middle frame)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of court overlay')
    
    args = parser.parse_args()
    
    print("=== Automatic Homography Prediction ===\n")
    
    # Predict homography
    result = predict_homography(
        video_path=args.video,
        model_path=args.model,
        device=args.device,
        frame_idx=args.frame
    )
    
    # Determine output filename
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"homography_matrix_{video_name}.json"
    
    # Save result
    print(f"\nSaving homography to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("✓ Saved successfully!")
    
    # Visualize if requested
    if args.visualize:
        viz_output = Path(args.output).parent / f"{Path(args.output).stem}_viz.jpg"
        H = np.array(result['homography_matrix'])
        visualize_homography(args.video, H, str(viz_output))
    
    print("\n=== Prediction Complete ===")
    print(f"Video: {args.video}")
    print(f"Homography: {args.output}")
    print(f"Model RMSE: {result['model_rmse']:.2f} ft")
    print("\nYou can now use this homography with your existing pipeline:")
    print(f"  python map_ball_to_court.py --calibration {args.output}")
