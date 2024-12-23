import cv2
import numpy as np
import math
import os
import exifread

def get_exif_data(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
        
        # Extract Focal Length
        focal_length = tags.get('EXIF FocalLength')
        if focal_length:
            focal_length_mm = float(focal_length.values[0].num) / float(focal_length.values[0].den)
        else:
            raise ValueError("Focal Length not found in the image metadata")
        
        # Extract 35mm Equivalent Focal Length
        focal_length_35mm = tags.get('EXIF FocalLengthIn35mmFilm')
        if focal_length_35mm:
            focal_length_35mm = float(focal_length_35mm.values[0])
        else:
            raise ValueError("35mm equivalent focal length not found in the image metadata")
        
        return focal_length_mm, focal_length_35mm

def read_image(image_path):
    """이미지 파일 읽기"""
    with open(image_path, "rb") as stream:
        bytes_array = bytearray(stream.read())
    numpy_array = np.asarray(bytes_array, dtype=np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image

def warp_image():
    # file_name = "difficult_WNQKD_TRUE imageSrc (48).jpg"
    file_name = r"F:\000. PJT\06. LX\Img\ColorDetection\TEST\주방-합 (186).jpg"
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    
    # Get focal lengths from the image metadata
    focal_length_mm, focal_length_35mm = get_exif_data(file_path)
    print(f"Focal Length: {focal_length_mm} mm")
    print(f"35mm Equivalent Focal Length: {focal_length_35mm} mm")

    # Calculate the crop factor
    crop_factor = focal_length_35mm / focal_length_mm
    print(f"Crop Factor: {crop_factor}")

    img = read_image(file_name)

    # 4개 코너가
    # 0 3
    # 1 2
    # 순서로 들어감
    src_pts = np.array([
        # [67, 1190],
        # [58, 1265],
        # [3528, 1343],
        # [3523, 1263]
        
        # [356 - 200, 864 - 200],
        # [356 - 200, 935 + 200],
        # [2669 + 200, 915 + 200],
        # [2670 + 200, 833 - 200]
        
        # [425 - 200, 1026 - 200],
        # [423 - 200, 1110 + 200],
        # [3360 + 200, 1070 + 200],
        # [3356 + 200, 986 - 200]
        
        [563 - 200, 1597 - 200],
        [562 - 200, 1640 + 200],
        [2326 + 200, 1680 + 200],
        [2325 + 200, 1638 - 200]
        
    ], dtype=np.float64)

    left_length = math.sqrt((src_pts[0, 0] - src_pts[1, 0])**2 + (src_pts[0, 1] - src_pts[1, 1])**2)
    right_length = math.sqrt((src_pts[2, 0] - src_pts[3, 0])**2 + (src_pts[2, 1] - src_pts[3, 1])**2)
    top_length = math.sqrt((src_pts[0, 0] - src_pts[3, 0])**2 + (src_pts[0, 1] - src_pts[3, 1])**2)
    bottom_length = math.sqrt((src_pts[2, 0] - src_pts[1, 0])**2 + (src_pts[2, 1] - src_pts[1, 1])**2)

    real_height_mm = 56.0

    print(f"left length : {left_length}, right length : {right_length}")
    print(f"top length : {top_length}, bottom length : {bottom_length}")
    print("")

    w, h = img.shape[1], img.shape[0]
    diagonal_of_pixel = math.sqrt(w * w + h * h)
    
    # Now we calculate the diagonal of the 35mm sensor using the crop factor
    diagonal_of_35mm = 43.3  # This can be considered constant for full-frame sensors

    physical_diag = diagonal_of_35mm / crop_factor
    sensor_size_mm = physical_diag / diagonal_of_pixel
    focal_length_pixel = focal_length_mm / sensor_size_mm

    left_depth = focal_length_pixel * real_height_mm / left_length
    right_depth = focal_length_pixel * real_height_mm / right_length
    depth_diff = abs(right_depth - left_depth)

    cx1 = (src_pts[0, 0] + src_pts[1, 0]) / 2.0
    cx2 = (src_pts[2, 0] + src_pts[3, 0]) / 2.0
    cy1 = (src_pts[0, 1] + src_pts[1, 1]) / 2.0
    cy2 = (src_pts[2, 1] + src_pts[3, 1]) / 2.0

    x_diff = abs(cx2 - cx1)
    y_diff = abs(cy2 - cy1)
    projected_width_pixel = math.sqrt(x_diff * x_diff + y_diff * y_diff)
    projected_width_mm = projected_width_pixel * min(left_depth, right_depth) / focal_length_pixel
    real_width_mm = math.sqrt(depth_diff * depth_diff + projected_width_mm * projected_width_mm)

    virtual_img = np.zeros((int(real_height_mm), int(real_width_mm), 3), dtype=np.uint8)

    # 4개 코너
    # 0 3
    # 1 2
    dst_pts = np.array([
        [0, 0],
        [0, virtual_img.shape[0] - 1],
        [virtual_img.shape[1] - 1, virtual_img.shape[0] - 1],
        [virtual_img.shape[1] - 1, 0]
    ], dtype=np.float64)

    homography, _ = cv2.findHomography(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, homography, (virtual_img.shape[1], virtual_img.shape[0]))

    return warped_img

warped_image = warp_image()
cv2.imshow("warped_image", warped_image)
cv2.waitKey()
cv2.imwrite("./warped_image.jpg", warped_image) 