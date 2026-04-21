import os
import cv2
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm
from PIL import Image

input_video = r"C:\Users\kunch\Desktop\GAN\input.mp4"
output_video = r"C:\Users\kunch\Desktop\GAN\enhanced_video.mp4"
output_frames_folder = r"C:\Users\kunch\Desktop\GAN\enhanced_frames"
os.makedirs(output_frames_folder, exist_ok=True)

# Force GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load RealESRGAN model
model_path = r"C:\Users\kunch\Desktop\GAN\RealESRGAN_x4plus.pth"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

half = device.type == 'cuda'  # Only enable FP16 if GPU exists

upscaler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=200,
    tile_pad=10,
    pre_pad=10,
    half=half,
    device=device
)

# Step 1: Extract frames
print("📌 Extracting frames from video...")
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)  # <-- Move this here
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for i in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

# Step 2: Enhance frames
torch.cuda.empty_cache()  # Clear GPU memory before enhancement

print("⚙️ Enhancing frames using GPU...")
enhanced_frames = []
for i, frame in tqdm(list(enumerate(frames))):
    output, _ = upscaler.enhance(frame, outscale=4)
    enhanced_frames.append(output)

# Step 3: Save enhanced video (no audio yet)
print("🎬 Rebuilding video...")
height, width = enhanced_frames[0].shape[:2]


video_writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

for frame in enhanced_frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

video_writer.release()

print("✨ Video enhancement complete:", output_video)
