import cv2
import numpy as np
from glob import glob
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import subprocess


# Helper function to resize image and add borders
def resize_and_add_border(img, target_width=1920, target_height=1280, border_color=[0, 0, 0]):
    if img is None:
        raise ValueError("Input image is None, cannot resize and add border.")
    
    original_height, original_width = img.shape[:2]
    
    aspect_ratio_img = original_width / original_height
    aspect_ratio_target = target_width / target_height
    
    if aspect_ratio_img > aspect_ratio_target:
        new_width = target_width
        new_height = int(target_width / aspect_ratio_img)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio_img)
    
    resized_img = cv2.resize(img, (new_width, new_height))
    
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left
    
    bordered_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)
    
    return bordered_img


# Function to apply zoom and pan
def apply_zoom_and_pan(img, zoom_factor, move_x, move_y):
    height, width = img.shape[:2]

    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = int(width / (2 * zoom_factor)), int(height / (2 * zoom_factor))
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    cropped_img = img[min_y:max_y, min_x:max_x]
    resized_img = cv2.resize(cropped_img, (width, height))

    translation_matrix = np.float32([[1, 0, move_x], [0, 1, move_y]])
    translated_img = cv2.warpAffine(resized_img, translation_matrix, (width, height))

    return translated_img


# Function to generate videos with zoom and pan
def GenerateZoomPanVideo(input_imgs, output_video_path, num_frames, max_zoom, max_move_x):
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    
    for img_path in input_imgs:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Image at path {img_path} not found.")
            continue
        
        img = resize_and_add_border(img, target_width=1920, target_height=1280)

        filename = os.path.splitext(os.path.basename(img_path))[0]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(output_video_path, filename + '.mp4'), fourcc, 30, (1920, 1280)) 

        zoom_factor = 1.0
        move_x = 0
        move_y = 0
        zoom_direction = 1
        move_direction = 1

        for i in range(num_frames):
            frame = apply_zoom_and_pan(img, zoom_factor, move_x, move_y)
            video.write(frame)

            zoom_factor += zoom_direction * (max_zoom - 1) / (num_frames / 2.5)
            if zoom_factor >= max_zoom or zoom_factor <= 1.0:
                zoom_direction *= -1 

            move_x += move_direction * (2 * max_move_x) / (num_frames / 2.5)
            if abs(move_x) >= max_move_x:
                move_direction *= -1 

        video.release()
        print(f"Video created successfully: {os.path.join(output_video_path, filename + '.mp4')}")


# Function to run OpenFace feature extraction
def runOpenFace(video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    input_vid = glob(os.path.abspath(video_path) + '/*.mp4')
    
    for video in input_vid:
        try:
            subprocess.run(['/Users/coco/OpenFace/OpenFace/build/bin/FeatureExtraction', '-f', video, '-out_dir', output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"OpenFace failed for {video}: {e}")


# Function to compute residuals for optimization
def residuals(params, input_landmarks, template_landmarks, weights):
    s = params[0]  # scale
    t = np.array(params[1:4])  # translation
    r = np.array(params[4:])  # rotation vector
    
    rotation_matrix = R.from_rotvec(r).as_matrix()  # rotation to matrix

    transformed_landmarks = s * (np.dot(template_landmarks, rotation_matrix.T) - t) # apply transformation
    error = (transformed_landmarks - input_landmarks) * weights[:, np.newaxis]
    
    return error.flatten()


# Function to calculate optimized coordinates
def calculatingCoordinates(init_guess):
    input_landmarks = glob('./videos/results/input/*.csv')
    template_landmarks = glob('./videos/results/template/*.csv')
    
    if not input_landmarks or not template_landmarks:
        raise ValueError("Landmark files missing for input or template.")

    input_landmarks = [pd.read_csv(f).iloc[-1, 435:639].values for f in input_landmarks]
    template_landmarks = [pd.read_csv(f).iloc[-1, 435:639].values for f in template_landmarks]
    
    input_landmarks = np.array([np.column_stack((l[:68], l[68:136], l[136:204])) for l in input_landmarks])
    template_landmarks = np.array([np.column_stack((l[:68], l[68:136], l[136:204])) for l in template_landmarks])

    weights = np.ones(68)
    
    result = least_squares(residuals, init_guess, args=(input_landmarks[0], template_landmarks[0], weights))
    s_opt, t_opt, r_opt = result.x[0], result.x[1:4], result.x[4:]
    print(f"Optimized Scale: {s_opt}, Translation: {t_opt}, Rotation: {r_opt}")


# Main function
def main(input_imgs, input_image_temp_path, output_video_path, output_video_temp_path):
    init_guess = [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3]
    temp_out = './videos/results/template'
    input_out = './videos/results/input'
    
    #GenerateZoomPanVideo(input_imgs, output_video_path, num_frames=150, max_zoom=1.25, max_move_x=350)
    #GenerateZoomPanVideo(input_image_temp_path, output_video_temp_path, num_frames=250, max_zoom=1.25, max_move_x=350)
    #runOpenFace(output_video_path, input_out)
    #runOpenFace(output_video_temp_path, temp_out)
    calculatingCoordinates(init_guess)


# Paths for input images and videos
input_image_path = './images/input/*.png'
input_image_temp_path = './images/template/*.png'
input_imgs = glob(input_image_path)
temp_imgs = glob(input_image_temp_path)

output_video_path = './videos/input'
output_video_temp_path = './videos/template'

# Run the main function
main(input_imgs, temp_imgs, output_video_path, output_video_temp_path)
