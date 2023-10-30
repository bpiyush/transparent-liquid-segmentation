import PIL
from PIL import Image, ImageOps

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
