import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

#  paths GitHub use
image_path = "datapath"
label_path = "datapath"

# === Existance Check ===
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image file not found: {image_path}")

if not os.path.exists(label_path):
    raise FileNotFoundError(f"❌ Label file not found: {label_path}")

print("✅ Files found. Loading image and label data...")

# === Loading Image MRI ===
image_obj = nib.load(image_path)
image_data = image_obj.get_fdata()
print(f"🧠 MRI data shape: {image_data.shape}")  # (240, 240, 155, 4)

# ===  labels ===
label_obj = nib.load(label_path)
label_data = label_obj.get_fdata()
print(f"🏷️ Labels shape: {label_data.shape}")  # (240, 240, 155)

# ===  layer & channel for imaging ===
layer = 80
channel = 0

plt.figure(figsize=(12, 5))

# image MRI
plt.subplot(1, 2, 1)
plt.imshow(image_data[:, :, layer, channel], cmap='gray')
plt.title(f'MRI Image (Layer {layer}, Channel {channel})')
plt.axis('off')

# Labels
plt.subplot(1, 2, 2)
plt.imshow(label_data[:, :, layer], cmap='nipy_spectral')
plt.title(f'Labels (Layer {layer})')
plt.axis('off')

plt.tight_layout()
plt.show()

# === Category Printing ===
unique_labels = np.unique(label_data)
print("\n🧾 Unique label values:", unique_labels)
print("📘 Label classes:")
print("  0: Normal")
print("  1: Edema")
print("  2: Non-enhancing tumor")
print("  3: Enhancing tumor")
