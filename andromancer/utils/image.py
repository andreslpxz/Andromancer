import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
import re

def label_image(image_path: str, elements: List[Dict], output_path: str) -> Dict[int, Dict]:
    """
    Draws numbered labels on the image at the positions of the given elements.
    Returns a mapping of label number to element.
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    label_map = {}

    for i, element in enumerate(elements):
        bounds = element.get('bounds', '')
        nums = [int(n) for n in re.findall(r"-?\d+", bounds)]
        if len(nums) < 4:
            continue

        x1, y1, x2, y2 = nums
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        label_num = i + 1
        label_map[label_num] = element

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label circle/background
        label_text = str(label_num)
        # Using textbbox if available (Pillow 8+)
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((center_x, center_y), label_text, font=font)
            # Expand a bit
            bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
        else:
            w, h = draw.textsize(label_text, font=font)
            bbox = (center_x - w//2 - 5, center_y - h//2 - 5, center_x + w//2 + 5, center_y + h//2 + 5)

        draw.ellipse(bbox, fill="red")
        draw.text((center_x, center_y), label_text, fill="white", font=font, anchor="mm" if hasattr(draw, 'textbbox') else None)

    img.save(output_path)
    return label_map
