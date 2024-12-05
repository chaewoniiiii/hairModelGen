import os
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Allow saving plots without GUI

# Supersample image
def supersample_image(img, grid_size=10):
    new_width = int(img.shape[1] * grid_size)
    new_height = int(img.shape[0] * grid_size)
    new_size = (new_width, new_height)
    high_res_image = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    return high_res_image

def average_superpixels(image, grid_size=10):
    h, w = image.shape
    assert h % grid_size == 0 and w % grid_size == 0, "Image dimensions must be divisible by grid_size."
    reshaped = image.reshape(h // grid_size, grid_size, w // grid_size, grid_size)
    averaged = reshaped.mean(axis=(1, 3))
    return averaged

# Save and display processed images
def showimage(myimage, save_path, figsize=[10, 10]):
    if myimage.ndim > 2:
        myimage = myimage[:, :, ::-1]  # Convert to RGB if needed
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(myimage, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Image saved as: {save_path}")

# HSV visualization for orientation and confidence
def save_hsv_image(orientations, confidence_map, save_path):
    hue = orientations / np.pi  # Normalize orientations to [0, 1]
    confidence = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min())
    
    hsv_image = np.zeros((*orientations.shape, 3), dtype=np.float32)
    hsv_image[..., 0] = hue  # Hue: orientation
    hsv_image[..., 1] = 1.0  # Saturation: fixed
    hsv_image[..., 2] = confidence  # Value: confidence
    
    rgb_image = cv2.cvtColor((hsv_image * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    cv2.imwrite(save_path, rgb_image)
    print(f"HSV image saved as: {save_path}")

# Create Gabor filters
def createFilter():
    filters = []
    sigma_u, sigma_v, lambd = 1.8, 2.4, 4
    for theta in np.linspace(0, np.pi, 32):  # 32 orientations
        kernel = cv2.getGaborKernel((9, 9), sigma_u, theta, lambd, sigma_u / sigma_v, 0, cv2.CV_32F)
        kernel /= kernel.sum()
        filters.append((theta, kernel))
    return filters

# Apply Gabor filters and compute responses
def applyFilter(img, filters):
    img /= 255.0  # Normalize to [0, 1]
    response_maps = []

    for theta, kernel in filters:
        response_map = cv2.filter2D(img, -1, kernel)
        #averaged_image = average_superpixels(response_map, grid_size=10)
        response_maps.append(response_map)

    response_maps = np.stack(response_maps, axis=-1)
    max_responses = np.max(response_maps, axis=-1)
    max_orientations = np.argmax(response_maps, axis=-1) * (np.pi / 32)  # Convert indices to angles
    confidence_map = calculate_confidence(response_maps, max_responses, max_orientations)

    return max_responses, confidence_map, max_orientations

# Calculate confidence map
def calculate_confidence(response_maps, max_responses, max_orientations):
    confidence_map = np.zeros_like(response_maps[..., 0])
    num_angles = response_maps.shape[-1]

    for i in range(num_angles):
        theta = i * (np.pi / 32)
        angular_distance = np.minimum(np.abs(theta - max_orientations), np.pi - np.abs(theta - max_orientations))
        delta_response = response_maps[..., i] - max_responses
        confidence_map += (angular_distance * (delta_response ** 2)) ** 0.5

    return confidence_map

# Iterative processing pipeline
def iterations(filters, input_dir, output_dir, max_iterations=3):
    iteration = 0
    while iteration < max_iterations:
        img_paths = glob.glob(os.path.join(input_dir, '*.png'))
        if not img_paths:
            print(f"No images found in {input_dir}.")
            return

        print(f"Iteration {iteration + 1}: Processing images from {input_dir}...")
        for img_path in img_paths:
            if iteration == 0:
                img = cv2.imread(img_path, 0).astype(np.float32)  # Load grayscale image
            else:
                img = cv2.imread(img_path, 0).astype(np.float32)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            #img = supersample_image(img)
            base_name = os.path.basename(img_path)
            max_responses, confidence_map, max_orientations = applyFilter(img, filters)

            if iteration == max_iterations - 1:
                # Save max responses
                save_path = os.path.join(output_dir, base_name)
                showimage(max_responses, save_path)
                
            else:
                # Save confidence map for intermediate iterations
                save_path = os.path.join(output_dir, base_name)
                showimage(confidence_map, save_path)

        input_dir = output_dir
        iteration += 1

# Main function
def main():
    input_dir = './images/results'
    output_dir = './images/results/confidences'
    os.makedirs(output_dir, exist_ok=True)

    filters = createFilter()
    iterations(filters, input_dir, output_dir)

if __name__ == "__main__":
    main()
