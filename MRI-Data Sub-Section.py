import numpy as np
import keras
import pandas as pd
import random

# Define a simple one dimensional "image" to extract from
image = np.array([10, 11, 12, 13, 14, 15])
print("Original Image Array:", image)

# Compute the dimensions of your "image"
image_length = image.shape[0]
print("Length of the image:", image_length)

# Define a patch length, which will be the size of your extracted sub-section
patch_length = 3
print("Patch Length:", patch_length)

# Define your start index
start_i = 0
print(f"Start index: {start_i}")

# Define an end index given your start index and patch size
end_i = start_i + patch_length
print(f"End index: {end_i}")

# Extract a sub-section from your "image"
sub_section = image[start_i:end_i]
print("Extracted Sub-section:", sub_section)

# Add one to your start index
start_i += 1

# Handle case where we run into the edge of the image
if start_i + patch_length > image_length:
    print("Reached edge of the image, no valid sub-section to extract.")
else:
    sub_section = image[start_i:end_i]
    print(f"New Sub-section starting from index {start_i}: {sub_section}")

# Compute and print the largest valid value for start index
print(f"The largest valid start index is {image_length - patch_length}")

# Compute the range of valid start indices
valid_start_indices = [i for i in range(image_length - patch_length + 1)]
print("Valid start indices:", valid_start_indices)

# Randomly select a start index within the valid range
start_i = random.choice(valid_start_indices)
print(f"Randomly selected start index: {start_i}")

# Randomly select multiple start indices in a loop
print("Randomly selected start indices:")
for _ in range(10):
    start_i = random.choice(valid_start_indices)
    print(f"  {start_i}")

# Now, let's demonstrate the background ratio using label data

# Simulating a patch of labels (0: background, 1: edema, 2: non-enhancing tumor, 3: enhancing tumor)
patch_labels = np.random.randint(0, 4, (16))
print("Simulated Patch Labels:", patch_labels)

# Calculate the background ratio (i.e., count the number of 0's in the labels)
bgrd_ratio = np.count_nonzero(patch_labels == 0) / len(patch_labels)
print(f"Background ratio (using np.count_nonzero): {bgrd_ratio}")

# Convert labels to one-hot encoding
patch_labels_one_hot = keras.utils.to_categorical(patch_labels, num_classes=4)
print("One-Hot Encoded Patch Labels:")
print(pd.DataFrame(patch_labels_one_hot, columns=['background', 'edema', 'non-enhancing tumor', 'enhancing tumor']))

# Compute background ratio from the one-hot encoding
bgrd_ratio_one_hot = np.sum(patch_labels_one_hot[:, 0]) / len(patch_labels)
print(f"Background ratio (using one-hot column): {bgrd_ratio_one_hot}")
