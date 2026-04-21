import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import os

# ---------------------- CONFIG ----------------------
model_path = r"C:\Users\kunch\Desktop\GAN\RealESRGAN_x4plus.pth"
input_image = r"C:\Users\kunch\Desktop\GAN\input.jpg"
output_image = r"C:\Users\kunch\Desktop\GAN\output_x4.png"
# ----------------------------------------------------

# Load RRDBNet model structure
model = RRDBNet(
    num_in_ch=3, 
    num_out_ch=3, 
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

# Initialize RealESRGAN engine
upscaler = RealESRGANer(
    scale=4,
    model_path=model_path,
    dni_weight=None,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,  # CPU mode -> must be False
    device=torch.device('cpu')
)

# Load image
if not os.path.exists(input_image):
    raise FileNotFoundError("Input image not found! Check path.")

img = cv2.imread(input_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Enhance image
output, _ = upscaler.enhance(img, outscale=4)

# Save result
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # Back to BGR for saving
cv2.imwrite(output_image, output)

print(f"Enhanced image saved successfully: {output_image}")
