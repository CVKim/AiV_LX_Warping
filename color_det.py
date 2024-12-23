import cv2
import numpy as np
import os
import json
from tkinter import Tk, filedialog
import math
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

def warp_image():
    file_name = "difficult_WNQKD_TRUE imageSrc (48).jpg"
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    
    # Get focal lengths from the image metadata
    focal_length_mm, focal_length_35mm = get_exif_data(file_path)
    print(f"Focal Length: {focal_length_mm} mm")
    print(f"35mm Equivalent Focal Length: {focal_length_35mm} mm")

    # Calculate the crop factor
    crop_factor = focal_length_35mm / focal_length_mm
    print(f"Crop Factor: {crop_factor}")

    img = cv2.imread(file_path)

    # 4개 코너가
    # 0 3
    # 1 2
    # 순서로 들어감
    src_pts = np.array([
        [356 - 100, 864 - 100],
        [356 - 100, 935 + 100],
        [2669 + 100, 915 + 100],
        [2670 + 100, 833 - 100]
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

def read_image(image_path):
    """이미지 파일 읽기"""
    stream = open(image_path, "rb")
    bytes_array = bytearray(stream.read())
    numpy_array = np.asarray(bytes_array, dtype=np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image

low_H, low_S, low_V = 60, 20, 190  # Green의 기본값 (하한값)
high_H, high_S, high_V = 255, 255, 255  # Green의 기본값 (상한값)

def nothing(x):
    pass

def load_json(file_path):
    """JSON 파일을 읽어와 데이터 반환"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON 파일을 읽을 수 없습니다: {e}")
        return None

def get_resized_points(points, original_size, resized_size):
    """
    원본 좌표를 리사이즈된 이미지의 크기에 맞게 보정
    points: 원본 좌표 리스트
    original_size: 원본 이미지 크기 (width, height)
    resized_size: 리사이즈된 이미지 크기 (width, height)
    """
    scale_x = resized_size[0] / original_size[0]
    scale_y = resized_size[1] / original_size[1]

    resized_points = [
        [int(point[0] * scale_x), int(point[1] * scale_y)] for point in points
    ]
    return np.array(resized_points, dtype=np.int32)


def get_points_from_json(json_data, label_name, original_size, resized_size):
    """JSON 데이터에서 특정 label의 points를 추출하고 보정"""
    points_list = []
    if not json_data:
        return points_list

    for shape in json_data.get("shapes", []):
        if shape.get("label") == label_name:
            # 좌표를 리사이즈 크기에 맞게 보정
            resized_points = get_resized_points(
                shape.get("points"), original_size, resized_size
            )
            points_list.append(resized_points)
    return points_list


def fit_best_line_on_mask(mask, result_image):
    """
    마스크에서 가장 적합한 라인을 피팅하고 파라미터를 반환
    """
    # 마스크 침식
    kernel = np.ones((1, 1), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Contours 찾기
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 contour 찾기 (면적 기준)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    fit_line_params = None

    if largest_contour is not None and len(largest_contour) >= 2:  # 최소한 두 점 이상이 있어야 fitLine 적용 가능
        # 라인 피팅 수행
        fit_line_params = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # 결과 이미지에 직선 그리기
        rows, cols = mask.shape[:2]
        left_y = int((-fit_line_params[2] * fit_line_params[1] / fit_line_params[0]) + fit_line_params[3])
        right_y = int(((cols - fit_line_params[2]) * fit_line_params[1] / fit_line_params[0]) + fit_line_params[3])
        cv2.line(result_image, (cols - 1, right_y), (0, left_y), (0, 0, 255), 1)

    return result_image, fit_line_params


def calculate_laser_distances(bbox, line_params):
    """
    팀버 바운딩 박스 하단부와 레이저 라인의 거리 계산 및 로그 남기기
    bbox: Bounding Box 좌표 (x_min, y_min, x_max, y_max)
    line_params: cv2.fitLine에서 반환된 (vx, vy, x, y)
    """
    x_min, y_min, x_max, y_max = bbox
    vx, vy, x, y = line_params  # cv2.fitLine 반환값

    # 라인의 기울기와 절편 계산
    m = vy / vx  # 기울기
    b = y - m * x  # 절편

    distances = []

    # 로그 출력 준비
    print(f"\n[LOG] Bounding Box X Range: {x_min} to {x_max}")
    print("[LOG] POS (Bounding Box X, Bounding Box Y, Laser Y) | DIS (Distance)\n")

    for x_coord in range(x_min, x_max + 1):
        y_box = y_max  # Bounding Box 하단부의 y 좌표
        y_laser = int(m * x_coord + b)  # 라인의 y 좌표
        distance = abs(y_box - y_laser)  # 거리 계산
        distances.append(distance)

        # 로그 출력
        print(f"POS: ({x_coord}, {y_box}, {y_laser}) | DIS: {distance}")

    return distances


def evaluate_line_horizontal(distances, threshold=5):
    """
    레이저 라인의 수평성을 평가
    """
    mean_distance = np.mean(distances) if distances else 0
    return mean_distance, mean_distance <= threshold


def detect_color(image_path):
    global low_H, low_S, low_V, high_H, high_S, high_V

    image = read_image(image_path)
    if image is None:
        print("이미지 없음")
        return

    json_path = os.path.splitext(image_path)[0] + ".json"
    json_data = load_json(json_path)

    original_size = (image.shape[1], image.shape[0])
    # resized_size = (499, 56)
    resized_size = (315, 56)

    timber_points = get_points_from_json(json_data, "TIMBER", original_size, resized_size)
    if not timber_points:
        print("JSON에서 'label: TIMBER'에 해당하는 points가 없습니다.")
        return

    resized = cv2.resize(image, resized_size)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    mask_overlay = np.zeros(resized.shape[:2], dtype=np.uint8)
    for points in timber_points:
        cv2.fillPoly(mask_overlay, [points], 255)

    while True:
        lower_bound = np.array([low_H, low_S, low_V])
        upper_bound = np.array([high_H, high_S, high_V])
        mask_hsv = cv2.inRange(hsv, lower_bound, upper_bound)

        combined_mask = cv2.bitwise_and(mask_hsv, mask_hsv, mask=mask_hsv)

        result = cv2.bitwise_and(resized, resized, mask=combined_mask)
        result_with_lines, fit_line_params = fit_best_line_on_mask(combined_mask, resized.copy())

        cv2.imshow("Mask", combined_mask)
        cv2.imshow("Result with Best Line", result_with_lines)
        
        # image write function 
        cv2.imwrite("./combined_mask.png", combined_mask)
        cv2.imwrite("./result_with_lines.png", result_with_lines)

        key = cv2.waitKey(1)
        if key == 27:
            break
        
    if fit_line_params is not None and timber_points:
        all_points = np.vstack(timber_points)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        bbox = (x_min, y_min, x_max, y_max)

        distances = calculate_laser_distances(bbox, fit_line_params)
        mean_distance, is_horizontal = evaluate_line_horizontal(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)

        print(f"Bounding Box: {bbox}")
        print(f"Min Distance: {min_distance:.2f} mm")
        print(f"Max Distance: {max_distance:.2f} mm")
        print(f"Mean Distance: {mean_distance:.2f} mm")
        print(f"Std Deviation: {std_distance:.2f} mm")
        print(f"Is Horizontal: {'Yes' if is_horizontal else 'No'}")
        
        # 결과 이미지에 각 포인트에 대한 거리 라인 그리기
        for x_coord in range(x_min, x_max + 1):
            y_box = y_max
            y_laser = int((fit_line_params[1] / fit_line_params[0]) * (x_coord - fit_line_params[2]) + fit_line_params[3])
            cv2.line(result_with_lines, (x_coord, y_box), (x_coord, y_laser), (255, 0, 0), 1)  # 파란색 라인

        cv2.imshow("Result with Distance Lines", result_with_lines)
        cv2.imwrite("Result_Distance_Lines.jpg", result_with_lines)
        cv2.waitKey(0)  # 키 입력 대기
        cv2.destroyAllWindows()

        # 로그를 txt 파일로 저장
        log_file_path = os.path.splitext(image_path)[0] + "_log.txt"
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Bounding Box: {bbox}\n")
            log_file.write(f"Min Distance: {min_distance:.2f} mm\n")
            log_file.write(f"Max Distance: {max_distance:.2f} mm\n")
            log_file.write(f"Mean Distance: {mean_distance:.2f} mm\n")
            log_file.write(f"Std Deviation: {std_distance:.2f} mm\n")
            log_file.write(f"Is Horizontal: {'Yes' if is_horizontal else 'No'}\n\n")
            log_file.write("POS (Bounding Box X, Bounding Box Y, Laser Y) | DIS (Distance)\n")
            for x_coord, distance in zip(range(x_min, x_max + 1), distances):
                y_box = y_max
                y_laser = int((fit_line_params[1] / fit_line_params[0]) * (x_coord - fit_line_params[2]) + fit_line_params[3])
                log_file.write(f"POS: ({x_coord}, {y_box}, {y_laser}) | DIS: {distance}\n")

        print(f"로그가 {log_file_path}에 저장되었습니다.")

    cv2.destroyAllWindows()


def select_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.bmp")])
    if not file_path:
        print("파일 선택 X !!!")
        return
    if not file_path.lower().endswith(('.jpg', '.bmp')):
        print("Only jpg or bmp")
        return

    detect_color(file_path)


if __name__ == "__main__":
    select_image()
