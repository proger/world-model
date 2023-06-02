from PIL import Image
import numpy as np

# Set the dimensions of the image
width = 16
height = 16

# Create a blank image with the specified dimensions
image = Image.new("RGB", (width, height))

# Convert the image to a numpy array for easy manipulation
pixels = np.array(image)

# Iterate over each pixel and set its color
for y in range(height):
    for x in range(width):
        # Generate a random RGB color
        r, g, b = np.random.randint(0, 150, size=3)
        
        # Set the color of the pixel
        pixels[y, x] = (r, r, r)

# Convert the numpy array back to an image
image = Image.fromarray(pixels.astype(np.uint8))

# Save the image
image.save("gray.png")

