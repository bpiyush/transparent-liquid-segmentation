import PIL
from PIL import Image, ImageOps, ImageDraw
import numpy as np

# define predominanat colors
COLORS = {
    "pink": (242, 116, 223),
    "cyan": (46, 242, 203),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def get_predominant_color(color_key, mode="RGB", alpha=0):
    assert color_key in COLORS.keys(), f"Unknown color key: {color_key}"
    if mode == "RGB":
        return COLORS[color_key]
    elif mode == "RGBA":
        return COLORS[color_key] + (alpha,)
    

def concat_images(images):
    im1 = images[0]
    dst = PIL.Image.new('RGB', (sum([im.width for im in images]), im1.height))
    for i, im in enumerate(images):
        dst.paste(im, (i * im.width, 0))
    return dst


def add_mask_on_image(image: Image, mask: Image, color="green"):
    image = image.copy()
    mask = mask.copy()

    color = get_predominant_color(color)
    mask = ImageOps.colorize(mask, (0, 0, 0, 0), color)

    mask = mask.convert("RGB")
    assert (mask.size == image.size)
    assert (mask.mode == image.mode)

    # Blend the original image and the segmentation mask with a 50% weight
    blended_image = Image.blend(image, mask, 0.5)
    return blended_image


def add_bbox_on_image(image: Image, bbox, color="green", width=2):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    color = get_predominant_color(color)
    draw.rectangle(bbox, outline=color, width=width)
    return image


def binarize_mask(mask, threshold=0.5):
    """
    Binarize a 2D mask array by setting values below a specified threshold to 0.

    Args:
        mask (numpy.ndarray): 2D mask array with values between 0 and 1.
        threshold (float): Threshold value (default is 0.5).

    Returns:
        numpy.ndarray: Binarized mask where values below the threshold are set to 0.
    """
    binarized_mask = np.where(mask >= threshold, 1, 0)
    return binarized_mask


def alpha_mask_to_pil_image(amask, threshold=0.5):
    # amask = amask.copy().astype(np.float32)
    # import ipdb; ipdb.set_trace()
    # # Replace all pixels where value < threshold to be 0
    # amask[amask < threshold] = 0.
    amask = binarize_mask(amask, threshold=threshold)
    
    amask_pil = PIL.Image.fromarray(np.clip(amask * 255., 0., 255.).astype(np.uint8))
    return amask_pil



def mask_to_bounding_box(ground_truth_map, perturbation=10):
    """
    Ref: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb
    """
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    if perturbation > 0:
        x_min = max(0, x_min - np.random.randint(-perturbation, perturbation))
        x_max = min(W, x_max + np.random.randint(-perturbation, perturbation))
        y_min = max(0, y_min - np.random.randint(-perturbation, perturbation))
        y_max = min(H, y_max + np.random.randint(-perturbation, perturbation))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox