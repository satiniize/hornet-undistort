import cv2
import numpy as np
import OpenEXR
import Imath

def load_exr_uv(filename):
    """
    Load an EXR file and extract the R and G channels as UV coordinates.
    """
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the image dimensions from the header
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define pixel type for float32
    channel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    # Extract R and G channels (U and V coordinates)
    u_channel = np.frombuffer(exr_file.channel('R', channel_type), dtype=np.float32).reshape(height, width)
    v_channel = np.frombuffer(exr_file.channel('G', channel_type), dtype=np.float32).reshape(height, width)

    return u_channel, v_channel, width, height

def main():
    # Load distorted_image (the distorted image)
    distorted_image = cv2.imread("Blender/Distorted.png", cv2.IMREAD_UNCHANGED)  # Distorted input image
    height, width = distorted_image.shape[:2]

    # Load UV map (DistortedUV)
    u, v, uv_width, uv_height = load_exr_uv("Blender/DistortedUV.exr")

    # Ensure UV map dimensions match distorted_image dimensions
    if (uv_width, uv_height) != (width, height):
        print("Resizing UV map to match image dimensions...")
        u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

    # Flip v because UV map is from bottom-left, but OpenCV is top-left
    v = 1.0 - v

    # Scale UV coordinates to pixel space
    # Here, (u, v) now represent coordinates in the original image2 space
    u_pixel = np.clip(u * (width - 1), 0, width - 1).astype(np.float32)
    v_pixel = np.clip(v * (height - 1), 0, height - 1).astype(np.float32)

    # Now we know: distorted_image(x, y) = image2(v_pixel(y, x), u_pixel(y, x))
    # We want to invert this:
    # For each pixel in image2 at (U, V), find (X, Y) in distorted_image.
    # We'll build inverse maps:
    inverse_map_x = -np.ones_like(u_pixel, dtype=np.float32)
    inverse_map_y = -np.ones_like(v_pixel, dtype=np.float32)

    # Populate the inverse map by iterating over distorted_image coordinates
    for y in range(height):
        for x in range(width):
            # source coords in image2
            U = int(u_pixel[y, x])
            V = int(v_pixel[y, x])
            # Assign the inverse mapping
            inverse_map_x[V, U] = x
            inverse_map_y[V, U] = y

    # Some pixels in image2 might remain -1 if the mapping isn't one-to-one.
    # Here, we do a simple fix: set undefined mappings to 0
    # You could also try more advanced hole-filling techniques.
    mask = (inverse_map_x < 0) | (inverse_map_y < 0)
    inverse_map_x[mask] = 0
    inverse_map_y[mask] = 0

    # Now use cv2.remap to reconstruct image2
    # image2(U, V) = distorted_image(inverse_map_y(U, V), inverse_map_x(U, V))
    image2 = cv2.remap(distorted_image, inverse_map_x, inverse_map_y, interpolation=cv2.INTER_LINEAR)

    # Save the reconstructed image2
    cv2.imwrite("ReconstructedUndistorted.png", image2)

if __name__ == "__main__":
    main()
