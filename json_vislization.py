import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import colors as mcolors

import matplotlib
matplotlib.use('TkAgg')

# Define a set of colors for different labels
LABEL_COLORS = {
    "label1": "red",
    "label2": "blue",
    "label3": "green",
    "TIMBER": "brown",
    "SCREW": "purple",
    # Add more labels and corresponding colors here
}

def list_png_files(directory):
    """Returns a list of all .png files in the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith('.png')]

def filter_png_files(png_files):
    """Filters out png files with 'background', 'SCREW', or 'TIMBER' in their name."""
    return [f for f in png_files if not any(exclude in f for exclude in ['background', 'SCREW', 'TIMBER'])]

def load_json_file(json_path):
    """Loads the JSON file and returns the parsed data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_color_for_label(label):
    """Returns a color for the given label."""
    return LABEL_COLORS.get(label, "yellow")  # Default color is yellow if the label is not found

def visualize_labels_on_image(image_path, shapes, output_dir):
    """Visualizes the labels and points on the image using matplotlib and saves the result."""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    for shape in shapes:
        label = shape['label']
        points = shape['points']
        color = get_color_for_label(label)
        polygon = Polygon(points, closed=True, fill=None, edgecolor=color)
        plt.gca().add_patch(polygon)
        plt.text(points[0][0], points[0][1], label, color=color, fontsize=12, weight='bold')

    plt.axis('off')

    # 이미지 이름을 유지하면서 output_dir에 저장
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualized image to: {output_path}")

def main(image_dir, json_dir, output_dir):
    # Output directory 생성
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: List and filter PNG files
    all_png_files = list_png_files(image_dir)
    filtered_png_files = filter_png_files(all_png_files)

    # Step 2: Match PNG files with corresponding JSON files and visualize
    for png_file in filtered_png_files:
        image_path = os.path.join(image_dir, png_file)
        json_file = png_file.replace('.png', '.json')
        json_path = os.path.join(json_dir, json_file)

        if os.path.exists(json_path):
            json_data = load_json_file(json_path)
            shapes = json_data.get('shapes', [])
            visualize_labels_on_image(image_path, shapes, output_dir)
        else:
            print(f"JSON file not found for image: {image_path}")

if __name__ == "__main__":
    image_dir = "vis"  # 이미지 파일들이 위치한 경로
    json_dir = "labels"    # JSON 파일들이 위치한 경로
    output_dir = "output_images"  # 결과 이미지를 저장할 경로

    main(image_dir, json_dir, output_dir)
