from PIL import Image
import numpy as np

# Define the base colors and their corresponding hues
base_colors = {
    "red": 0,
    "orange": 30,
    "yellow": 60,
    "lime": 120,
    "green": 180,
    "aqua": 210,
    "blue": 240,
    "pink": 330,
    "purple": 300
}

# Iterate over each base color and generate the corresponding image
for color, hue in base_colors.items():
    # Create a blank image with the dimensions 16x16
    image = Image.new("HSV", (16, 16))

    # Convert the image to a numpy array for easy manipulation
    pixels = np.array(image)

    # Generate random hue offsets for each pixel
    hue_offsets = np.random.randint(-20, 21, size=(16, 16))

    # Iterate over each pixel and set its color
    for y in range(16):
        for x in range(16):
            # Calculate the new hue value based on the base color and offset
            h = (hue + hue_offsets[y, x]) % 360

            # Set the color of the pixel with constant saturation and brightness
            pixels[y, x] = (h, 220, 220)

    # Convert the numpy array back to an image and convert from HSV to RGB
    image = Image.fromarray(pixels.astype(np.uint8), mode='HSV').convert('RGB')

    # Save the image with the color name
    image.save(f"{color}_shades.png")

