import numpy as np
import time

# Class-to-color mapping
CLASS_TO_COLOR = {
    0: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
}

# Convert CLASS_TO_COLOR into arrays for compatibility
COLOR_ARRAY = np.array(list(CLASS_TO_COLOR.values()), dtype=np.uint8)
CLASS_ARRAY = np.array(list(CLASS_TO_COLOR.keys()), dtype=np.uint8)

# Your implementation

def mask_to_class_index(mask):
    """
    Convert an RGB mask to a class index mask using vectorized operations.
    """
    mask = np.array(mask)
    class_indices = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Define color-to-class mapping
    color_to_class = {
        (0, 0, 0): 0,
        (111, 74, 0): 1,
        (81, 0, 81): 2,
        (128, 64, 128): 3,
        (244, 35, 232): 4,
        (250, 170, 160): 5,
        (230, 150, 140): 6,
        (70, 70, 70): 7,
        (102, 102, 156): 8,
        (190, 153, 153): 9,
        (180, 165, 180): 10,
        (150, 100, 100): 11,
        (150, 120, 90): 12,
        (153, 153, 153): 13,
        (250, 170, 30): 14,
        (220, 220, 0): 15,
        (107, 142, 35): 16,
        (152, 251, 152): 17,
        (70, 130, 180): 18,
        (220, 20, 60): 19,
        (255, 0, 0): 20,
        (0, 0, 142): 21,
        (0, 0, 70): 22,
        (0, 60, 100): 23,
        (0, 0, 90): 24,
        (0, 0, 110): 25,
        (0, 80, 100): 26,
        (0, 0, 230): 27,
        (119, 11, 32): 28,
    }

    # Convert color-to-class mapping into numpy arrays
    colors = np.array(list(color_to_class.keys()), dtype=np.uint8)
    class_ids = np.array(list(color_to_class.values()), dtype=np.uint8)

    # Reshape for broadcasting
    reshaped_mask = mask.reshape(-1, 3)  # Flatten mask to [num_pixels, 3]
    reshaped_colors = colors.reshape(1, -1, 3)  # Shape [1, num_classes, 3]

    # Compare all pixels with all colors
    matches = np.all(reshaped_mask[:, None, :] == reshaped_colors, axis=2)

    # Assign class indices based on matches
    class_indices = class_ids[np.argmax(matches, axis=1)].reshape(mask.shape[:2])

    return class_indices

def class_index_to_mask(class_indices):
    """
    Convert a class index mask to an RGB mask using vectorized operations.
    """
    h, w = class_indices.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_TO_COLOR.items():
        rgb_mask[class_indices == class_id] = color

    return rgb_mask

# My implementation

def maskToClassIndex_numpy(mask):
    """
    Convert an RGB mask to a class index mask using NumPy.
    """
    h, w, _ = mask.shape
    flat_mask = mask.reshape(-1, 3)  # Flatten to (H*W, 3)
    class_indices = np.zeros(h * w, dtype=np.uint8)

    for i, color in enumerate(COLOR_ARRAY):
        matches = np.all(flat_mask == color, axis=1)
        class_indices[matches] = CLASS_ARRAY[i]

    return class_indices.reshape(h, w)

def classIndexToMask_numpy(class_indices):
    """
    Convert a class index mask to an RGB mask using NumPy.
    """
    h, w = class_indices.shape
    flat_class = class_indices.flatten()
    rgb_mask = np.zeros((h * w, 3), dtype=np.uint8)

    for i, class_id in enumerate(CLASS_ARRAY):
        matches = flat_class == class_id
        rgb_mask[matches] = COLOR_ARRAY[i]

    return rgb_mask.reshape(h, w, 3)

# Testing and benchmarking
if __name__ == "__main__":
    # Generate a random RGB mask
    image_size = (512, 512, 3)
    rgb_mask = np.random.randint(0, 255, size=image_size, dtype=np.uint8)

    # Test your implementation
    start = time.time()
    class_indices_yours = mask_to_class_index(rgb_mask)
    reconstructed_rgb_mask_yours = class_index_to_mask(class_indices_yours)
    your_time = time.time() - start
    print(f"Your Implementation Time (mask -> class -> mask): {your_time:.6f} seconds")

    # Test my implementation
    start = time.time()
    class_indices_mine = maskToClassIndex_numpy(rgb_mask)
    reconstructed_rgb_mask_mine = classIndexToMask_numpy(class_indices_mine)
    my_time = time.time() - start
    print(f"My Implementation Time (mask -> class -> mask): {my_time:.6f} seconds")

    # Validate outputs
    print("Outputs Match (mask -> class):",
          np.array_equal(class_indices_yours, class_indices_mine))
    print("Outputs Match (class -> mask):",
          np.array_equal(reconstructed_rgb_mask_yours, reconstructed_rgb_mask_mine))
