# hentaivid/video.py 
import cv2
import numpy as np
from PIL import Image
from typing import Generator, Optional, Tuple
from loguru import logger

def get_video_frames(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    A generator that yields frames from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    
    cap.release()

def get_video_properties(video_path: str) -> Optional[Tuple[int, int, float]]:
    """
    Returns the width, height, and fps of a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps 

def create_video_writer(output_path: str, width: int, height: int, fps: float) -> Optional[cv2.VideoWriter]:
    """
    Creates a VideoWriter object to save a video file.
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return writer
    except Exception as e:
        logger.error(f"Error creating video writer: {e}")
        return None

def overlay_qr_code(frame: np.ndarray, qr_image: Image.Image, region: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Overlays a QR code onto a frame at the specified region.
    """
    x, y, w, h = region
    
    # Resize QR code to fit the region
    qr_resized = qr_image.resize((w, h), Image.Resampling.LANCZOS)
    
    # Convert PIL image to numpy array
    qr_np = np.array(qr_resized)
    
    # Place the QR code on the frame
    frame[y:y+h, x:x+w] = qr_np
    
    return frame 