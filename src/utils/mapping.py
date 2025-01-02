import numpy as np

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

def maskToClassIndex(mask):
    """
    Convert an RGB mask to a class index mask using NumPy.
    """
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    h, w, _ = mask.shape
    flat_mask = mask.reshape(-1, 3)  # Flatten to (H*W, 3)
    class_indices = np.zeros(h * w, dtype=np.uint8)

    for i, color in enumerate(COLOR_ARRAY):
        matches = np.all(flat_mask == color, axis=1)
        class_indices[matches] = CLASS_ARRAY[i]

    return class_indices.reshape(h, w)

def classIndexToMask(class_indices):
    """
    Convert a class index mask to an RGB mask using NumPy.
    """
    if not isinstance(class_indices, np.ndarray):
        class_indices = np.array(class_indices)

    h, w = class_indices.shape
    flat_class = class_indices.flatten()
    rgb_mask = np.zeros((h * w, 3), dtype=np.uint8)

    for i, class_id in enumerate(CLASS_ARRAY):
        matches = flat_class == class_id
        rgb_mask[matches] = COLOR_ARRAY[i]

    return rgb_mask.reshape(h, w, 3)
