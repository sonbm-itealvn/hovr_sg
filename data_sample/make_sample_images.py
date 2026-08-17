from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).parent
for name, color in [("sample_000.jpg", (235, 245, 255)), ("sample_001.jpg", (255, 240, 230)), ("sample_002.jpg", (240, 255, 235))]:
    image = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 30, 130, 220), outline=(30, 80, 160), width=3)
    draw.rectangle((140, 120, 210, 210), outline=(160, 80, 30), width=3)
    draw.rectangle((30, 215, 220, 250), outline=(80, 50, 20), width=3)
    image.save(root / name, quality=90)
