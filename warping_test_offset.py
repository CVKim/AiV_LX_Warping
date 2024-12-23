import cv2
import numpy as np
import math
import os
import exifread
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image
import logging

def setup_logger(log_file_path):
    """로거 설정 함수"""
    logger = logging.getLogger('ScrewProcessor')
    logger.setLevel(logging.DEBUG)
    
    # 파일 핸들러 설정
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러 설정
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 포매터 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def get_exif_data(image_path, logger):
    """이미지의 EXIF 데이터를 추출하여 초점 거리 반환"""
    with image_path.open('rb') as f:
        tags = exifread.process_file(f, stop_tag="UNDEF", details=False)
        
        # Focal Length 추출
        focal_length = tags.get('EXIF FocalLength')
        if focal_length:
            focal_length_mm = float(focal_length.values[0].num) / float(focal_length.values[0].den)
        else:
            logger.error("Focal Length not found in the image metadata")
            raise ValueError("Focal Length not found in the image metadata")
        
        # 35mm Equivalent Focal Length 추출
        focal_length_35mm = tags.get('EXIF FocalLengthIn35mmFilm')
        if focal_length_35mm:
            focal_length_35mm = float(focal_length_35mm.values[0])
        else:
            logger.error("35mm equivalent focal length not found in the image metadata")
            raise ValueError("35mm equivalent focal length not found in the image metadata")
        
        return focal_length_mm, focal_length_35mm

def parse_json(json_path, logger):
    """JSON 파일을 파싱하여 레이블 딕셔너리 반환"""
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # utf-8 실패 시 cp949 시도
        with json_path.open('r', encoding='cp949') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"JSON 파일 파싱 실패: {e}")
        raise
    
    # 'shapes' 키가 있고 리스트인지 확인
    if 'shapes' in data and isinstance(data['shapes'], list):
        labels = data['shapes']
    elif isinstance(data, dict):
        labels = [data]
    elif isinstance(data, list):
        labels = data
    else:
        logger.error(f"Unsupported JSON structure in {json_path}")
        raise ValueError(f"Unsupported JSON structure in {json_path}")
    
    label_dict = {}
    for label in labels:
        label_name = label.get('label')
        points = label.get('points', [])
        bbox = label.get('bbox', {})
        if label_name and points:
            label_name_upper = label_name.upper()
            if label_name_upper not in label_dict:
                label_dict[label_name_upper] = []
            label_dict[label_name_upper].append({
                'points': points,
                'bbox': bbox
            })
    
    return label_dict

def get_src_points(label_dict, logger):
    """'TIMBER' 레이블에서 원본 포인트 추출"""
    # 'TIMBER' 레이블이 존재하는지 확인
    if 'TIMBER' not in label_dict:
        logger.error("'TIMBER' label not found in JSON")
        raise ValueError("'TIMBER' label not found in JSON")
    
    timber_entries = label_dict['TIMBER']
    if not timber_entries:
        logger.error("No 'TIMBER' entries found in JSON")
        raise ValueError("No 'TIMBER' entries found in JSON")
    
    # 여러 'TIMBER' 엔트리가 있는 경우, 첫 번째를 사용하도록 수정
    timber_entry = timber_entries[0]
    
    points = timber_entry.get('points', [])
    if not points or len(points) < 4:
        logger.error("'TIMBER' label does not have enough points to form a polygon")
        raise ValueError("'TIMBER' label does not have enough points to form a polygon")
    
    # NumPy 배열로 변환
    points_np = np.array(points, dtype=np.float32)
    
    # 좌표별 꼭짓점 계산
    top_left = points_np[np.argmin(points_np.sum(axis=1))]
    bottom_right = points_np[np.argmax(points_np.sum(axis=1))]
    top_right = points_np[np.argmin(points_np[:, 1] - points_np[:, 0])]
    bottom_left = points_np[np.argmax(points_np[:, 1] - points_np[:, 0])]
    
    ordered_box = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.float64)
    
    logger.info(f"Ordered Source Points: {ordered_box}")
    
    return ordered_box

def get_screw_centers_and_points(label_dict, logger):
    """
    'SCREW' 레이블의 각 개별 중심점, 너비, 높이와 첫 두 점을 계산하여 리스트로 반환합니다.
    
    Args:
        label_dict (dict): JSON 파일에서 파싱된 레이블 딕셔너리.
        logger (Logger): 로깅 객체.
    
    Returns:
        list of dict: 각 'SCREW' 레이블의 중심점, 너비, 높이 및 첫 두 점을 포함하는 딕셔너리의 리스트.
    """
    # 'SCREW' 레이블 존재 여부 확인
    if 'SCREW' not in label_dict:
        logger.warning("'SCREW' 레이블이 JSON 파일에 없습니다.")
        return []
    
    screw_entries = label_dict['SCREW']
    if not screw_entries:
        logger.warning("JSON 파일에 'SCREW' 레이블 항목이 없습니다.")
        return []
    
    screws = []
    for idx, entry in enumerate(screw_entries, start=1):
        screw_points = entry.get('points', [])
        bbox = entry.get('bbox', {})
        
        # 유효하지 않은 스크류 항목 건너뛰기
        if not screw_points or len(screw_points) < 2:
            logger.warning(f"'SCREW' 레이블 항목 {idx}에 유효하지 않은 'points' 데이터가 있습니다. 건너뜁니다.")
            continue
        # 'bbox'가 없을 경우, points를 이용하여 width와 height 계산
        if not bbox or len(bbox) < 4:
            logger.warning(f"'SCREW' 레이블 항목 {idx}에 'bbox' 데이터가 없거나 형식이 올바르지 않습니다. 'points'를 사용하여 width와 height를 계산합니다.")
            try:
                screw_points_np = np.array(screw_points, dtype=np.float64)
                x_min, y_min = screw_points_np.min(axis=0)
                x_max, y_max = screw_points_np.max(axis=0)
                width = x_max - x_min
                height = y_max - y_min
            except Exception as e:
                logger.error(f"'SCREW' 레이블 항목 {idx}의 'points' 데이터로 width와 height를 계산하는 데 실패했습니다: {e}. 건너뜁니다.")
                continue
        else:
            try:
                # 'bbox'로부터 너비와 높이 추출 (Assuming 'bbox' format: {'x_min': ..., 'y_min': ..., 'width': ..., 'height': ...})
                width = float(bbox.get('width', 0))
                height = float(bbox.get('height', 0))
            except Exception as e:
                logger.error(f"'SCREW' 레이블 항목 {idx}의 'bbox' 데이터로 width와 height를 추출하는 데 실패했습니다: {e}. 건너뜁니다.")
                continue
        
        try:
            # NumPy 배열로 변환
            screw_points_np = np.array(screw_points, dtype=np.float64)
            center_x = np.mean(screw_points_np[:, 0])
            center_y = np.mean(screw_points_np[:, 1])
            center = (center_x, center_y)
            
            # 첫 두 점 추출
            point1 = tuple(screw_points_np[0])
            point2 = tuple(screw_points_np[1])
            
            screws.append({
                'center': center,
                'width': width,
                'height': height,
                'point1': point1,
                'point2': point2
            })
            
            logger.info(f"Screw {idx} Center: {center}, Width: {width}, Height: {height}")
            logger.info(f"Screw {idx} Point1: {point1}, Point2: {point2}")
        
        except Exception as e:
            logger.error(f"'SCREW' 레이블 항목 {idx} 처리 중 오류 발생: {e}. 건너뜁니다.")
            continue  # 다음 스크류로 넘어가기
    
    return screws

def read_image(image_path):
    """이미지 파일 읽기"""
    with open(image_path, "rb") as stream:
        bytes_array = bytearray(stream.read())
    numpy_array = np.asarray(bytes_array, dtype=np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image

def distance_point_to_line(px, py, x1, y1, x2, y2):
    """점(px, py)이 선분(x1,y1)-(x2,y2)에서 얼마나 떨어져 있는지 계산"""
    if (x1, y1) == (x2, y2):
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    else:
        line_mag = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        u = ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / (line_mag**2)
        if u < 0.0:
            ix, iy = x1, y1
        elif u >1.0:
            ix, iy = x2, y2
        else:
            ix = x1 + u*(x2 - x1)
            iy = y1 + u*(y2 - y1)
        distance = math.sqrt((px - ix)**2 + (py - iy)**2)
        return distance

def compute_radius(width, height):
    """스크류의 width와 height를 기반으로 원의 반지름을 계산"""
    # width와 height의 평균을 사용하여 반지름 계산
    avg_size = (width + height) / 2.0
    scaling_factor = 2  # 이 값을 조정하여 원의 크기를 적절히 설정
    radius = int(avg_size / scaling_factor)
    return max(radius, 3)  # 최소 반지름을 3으로 설정

def adjust_src_pts(src_pts, offset, logger):
    """
    src_pts의 4개 꼭지점을 지정된 방향으로 offset만큼 확장합니다.
    
    Args:
        src_pts (np.ndarray): 원본 소스 포인트 (4x2 배열).
        offset (float): 확장할 오프셋 값.
        logger (Logger): 로깅 객체.
    
    Returns:
        np.ndarray: 확장된 소스 포인트 (4x2 배열).
    """
    expanded_pts = src_pts.copy()
    
    # 좌상단: x -= offset, y -= offset
    expanded_pts[0][0] -= offset
    expanded_pts[0][1] -= offset
    
    # 좌하단: x -= offset, y += offset
    expanded_pts[1][0] -= offset
    expanded_pts[1][1] += offset
    
    # 우하단: x += offset, y += offset
    expanded_pts[2][0] += offset
    expanded_pts[2][1] += offset
    
    # 우상단: x += offset, y -= offset
    expanded_pts[3][0] += offset
    expanded_pts[3][1] -= offset
    
    logger.info(f"Expanded Source Points: {expanded_pts}")
    
    return expanded_pts

def warp_image(image_path, src_pts, logger):
    """이미지 워핑 및 변환 수행"""
    # EXIF 데이터에서 초점 거리 추출
    focal_length_mm, focal_length_35mm = get_exif_data(image_path, logger)
    logger.info(f"Processing Image: {image_path.name}")
    logger.info(f"Focal Length: {focal_length_mm} mm")
    logger.info(f"35mm Equivalent Focal Length: {focal_length_35mm} mm")
    
    # Crop Factor 계산
    crop_factor = focal_length_35mm / focal_length_mm
    logger.info(f"Crop Factor: {crop_factor}")
    
    # 이미지 읽기
    img = read_image(image_path)
    
    # Optional: 길이 계산 (필요 시 제거 가능)
    left_length = math.sqrt((src_pts[0, 0] - src_pts[1, 0])**2 + (src_pts[0, 1] - src_pts[1, 1])**2)
    right_length = math.sqrt((src_pts[2, 0] - src_pts[3, 0])**2 + (src_pts[2, 1] - src_pts[3, 1])**2)
    top_length = math.sqrt((src_pts[0, 0] - src_pts[3, 0])**2 + (src_pts[0, 1] - src_pts[3, 1])**2)
    bottom_length = math.sqrt((src_pts[2, 0] - src_pts[1, 0])**2 + (src_pts[2, 1] - src_pts[1, 1])**2)

    real_height_mm = 56.0  # 실제 높이 (필요에 따라 조정)
    
    logger.info(f"left length : {left_length}, right length : {right_length}")
    logger.info(f"top length : {top_length}, bottom length : {bottom_length}")
    logger.info("")
    
    h, w = img.shape[0], img.shape[1]
    diagonal_of_pixel = math.sqrt(w * w + h * h)
    
    # 35mm 센서의 대각선 계산 (crop factor 사용)
    diagonal_of_35mm = 43.3  # 풀프레임 센서 기준 상수
    
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

    # 목적지 포인트 설정 (좌상단, 좌하단, 우하단, 우상단)
    dst_pts = np.array([
        [0, 0],
        [0, virtual_img.shape[0] - 1],
        [virtual_img.shape[1] - 1, virtual_img.shape[0] - 1],
        [virtual_img.shape[1] - 1, 0]
    ], dtype=np.float64)

    homography, status = cv2.findHomography(src_pts, dst_pts)
    if homography is None:
        logger.error(f"Homography computation failed for image: {image_path}")
        raise ValueError(f"Homography computation failed for image: {image_path}")
    
    warped_img = cv2.warpPerspective(img, homography, (virtual_img.shape[1], virtual_img.shape[0]))
    cv2.imshow("warped_image", warped_img)
    cv2.waitKey(1)  # 창을 빠르게 열고 닫기 위해 1ms 대기
    return warped_img, homography, dst_pts, virtual_img.shape

def save_image(warped_path, warped_image, logger):
    """Pillow를 사용하여 이미지 저장"""
    try:
        # OpenCV 이미지를 RGB로 변환
        warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        # Pillow 이미지를 생성
        image_pil = Image.fromarray(warped_image_rgb)
        # Pillow로 이미지 저장 (파일 경로를 문자열로 변환)
        image_pil.save(str(warped_path))
        logger.info(f"Image saved successfully: {warped_path}")
    except Exception as e:
        logger.error(f"Failed to save image {warped_path}: {e}")

def process_images(source_directory, result_directory, logger):
    """이미지 파일들을 처리하는 함수"""
    # RESULT 폴더 내에 OFFSET 서브 폴더 생성
    offset_directory = result_directory / 'OFFSET'
    offset_directory.mkdir(exist_ok=True)
    
    # 디렉토리 내의 모든 png/jpg/jpeg 파일 찾기
    for file in source_directory.iterdir():
        if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_path = file
            json_filename = file.stem + '.json'
            json_path = source_directory / json_filename
            
            if not json_path.exists():
                logger.warning(f"JSON file not found for image: {file.name}, skipping...")
                continue
            
            try:
                # JSON 파일 파싱
                label_dict = parse_json(json_path, logger)
                
                # 'TIMBER' 레이블의 src_pts 추출
                src_pts = get_src_points(label_dict, logger)
                
                # 이미지 워핑
                warped_image, homography, dst_pts, virtual_shape = warp_image(image_path, src_pts, logger)
                
                # 원본 워핑 이미지 저장
                org_warped_filename = file.stem + '_origin_warped.jpg'
                org_warped_path = result_directory / org_warped_filename
                save_image(org_warped_path, warped_image, logger)
                
                # 추가: src_pts를 확장하여 워핑된 이미지 저장
                offset = 25  # 오프셋 값 설정
                expanded_src_pts = adjust_src_pts(src_pts, offset, logger)
                
                # 가상 이미지 크기 확장
                expanded_virtual_height = virtual_shape[0] + 2 * offset
                expanded_virtual_width = virtual_shape[1] + 2 * offset
                
                # 확장된 목적지 포인트 설정 (좌상단, 좌하단, 우하단, 우상단)
                # 기존 dst_pts는 [0,0], [0, height-1], [width-1, height-1], [width-1,0]
                # 확장된 목적지 포인트는 [offset, offset], [offset, height-1 + offset], [width-1 + offset, height-1 + offset], [width-1 + offset, offset]
                expanded_dst_pts = dst_pts + np.array([offset, offset])
                
                # 호모그래피 계산 (확장된 src_pts -> 확장된 dst_pts)
                homography_expanded, status_expanded = cv2.findHomography(expanded_src_pts, expanded_dst_pts)
                if homography_expanded is None:
                    logger.error(f"Homography computation failed for expanded src_pts in image: {image_path}")
                else:
                    # 이미지 워핑
                    warped_image_expanded = cv2.warpPerspective(read_image(image_path), homography_expanded, (expanded_virtual_width, expanded_virtual_height))
                    
                    # 확장된 워핑 이미지 저장
                    expanded_warped_filename = file.stem + '_warped_expanded.jpg'
                    expanded_warped_path = offset_directory / expanded_warped_filename
                    save_image(expanded_warped_path, warped_image_expanded, logger)
                    
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}\n")
                continue  # 다음 파일로 넘어가기
            
            try:
                # 'SCREW' 레이블의 중심점, 너비, 높이 및 포인트 추출
                screws = get_screw_centers_and_points(label_dict, logger)
                
                if not screws:
                    logger.warning(f"No valid 'SCREW' entries found for image: {file.name}, skipping connection process.")
                    continue
                
                # 호모그래피 변환된 모든 SCREW 중심점과 크기를 리스트에 저장
                transformed_screws = []
                
                # 각 스크류에 대해 호모그래피 변환 후 리스트에 추가
                for idx, screw in enumerate(screws, start=1):
                    # 중심점 변환
                    screw_center_homogeneous = np.array([screw['center'][0], screw['center'][1], 1.0])
                    transformed_center = homography @ screw_center_homogeneous
                    if transformed_center[2] == 0:
                        logger.error(f"Homography transformation resulted in zero w-component for screw {idx}, skipping marker.")
                        continue
                    transformed_center /= transformed_center[2]
                    transformed_center = transformed_center[:2].astype(int)
                    
                    # 너비와 높이 변환
                    # 기준 벡터를 사용하여 너비와 높이를 변환
                    width_vector = np.array([screw['width'], 0, 0])
                    height_vector = np.array([0, screw['height'], 0])
                    
                    transformed_width_vector = homography @ width_vector
                    transformed_height_vector = homography @ height_vector
                    
                    # 변환된 너비와 높이 계산
                    transformed_width = np.linalg.norm(transformed_width_vector[:2])
                    transformed_height = np.linalg.norm(transformed_height_vector[:2])
                    
                    # 원의 반지름 계산
                    radius = compute_radius(transformed_width, transformed_height)
                    
                    # 중심점에 마커 그리기 (빨간색 원)
                    cv2.circle(warped_image, tuple(transformed_center), radius=radius, color=(0, 0, 255), thickness=1)
                    logger.info(f"Drew marker for Screw {idx} at: {transformed_center}, Width: {transformed_width:.2f}, Height: {transformed_height:.2f}, Radius: {radius}")
                    
                    # 두 점 변환
                    point1_homogeneous = np.array([screw['point1'][0], screw['point1'][1], 1.0])
                    transformed_point1 = homography @ point1_homogeneous
                    point2_homogeneous = np.array([screw['point2'][0], screw['point2'][1], 1.0])
                    transformed_point2 = homography @ point2_homogeneous
                    
                    if transformed_point1[2] == 0 or transformed_point2[2] == 0:
                        logger.error(f"Homography transformation resulted in zero w-component for screw {idx} points, skipping line.")
                        continue
                    
                    transformed_point1 /= transformed_point1[2]
                    transformed_point2 /= transformed_point2[2]
                    
                    transformed_point1 = transformed_point1[:2].astype(int)
                    transformed_point2 = transformed_point2[:2].astype(int)
                    
                    # 두 점을 연결하는 선 그리기 (파란색 선)
                    cv2.line(warped_image, tuple(transformed_point1), tuple(transformed_point2), color=(255, 0, 0), thickness=1)
                    logger.info(f"Drew line for Screw {idx} from {transformed_point1} to {transformed_point2}")
                    
                    # 거리 계산 (픽셀 단위)
                    distance_pixels = math.sqrt(
                        (transformed_point2[0] - transformed_point1[0]) ** 2 +
                        (transformed_point2[1] - transformed_point1[1]) ** 2
                    )
                    
                    # 변환된 중심점과 크기를 리스트에 추가
                    transformed_screws.append({
                        'center': tuple(transformed_center),
                        'width': transformed_width,
                        'height': transformed_height
                    })

            except Exception as e:
                logger.error(f"Error during SCREW processing for {file.name}: {e}\n")
                continue  # 다음 파일로 넘어가기
            
            try:
                # SCREW 간 거리 측정 및 선 그리기
                if len(transformed_screws) >= 2:
                    # 너비와 높이가 5 이상인 스크류만 필터링
                    filtered_screws = [s for s in transformed_screws if s['width'] >= 5.0 and s['height'] >= 5.0]
                    
                    if not filtered_screws:
                        logger.warning("No screws with width and height >= 5.0 found.")
                    else:
                        # x 좌표 기준으로 좌측에서 우측으로 정렬
                        sorted_screws = sorted(filtered_screws, key=lambda x: x['center'][0])
                        logger.info("Filtered and Sorted Screws (Left to Right):")
                        for idx, screw in enumerate(sorted_screws, start=1):
                            logger.info(f"Screw {idx}: Center={screw['center']}, Width={screw['width']:.2f}, Height={screw['height']:.2f}")
                        
                        # 연결 로직 수정: 각 SCREW에서 가장 가까운 SCREW(150 픽셀 이상)를 찾아 연결
                        # 시작 스크류는 첫 번째 스크류
                        current_screw = sorted_screws[0]
                        connected_indices = set()
                        connected_indices.add(0)  # 첫 번째 스크류는 연결됨
                        
                        # 첫 번째 스크류에 노란색 원 표시
                        radius = compute_radius(current_screw['width'], current_screw['height'])
                        cv2.circle(warped_image, current_screw['center'], radius=5, color=(0, 255, 255), thickness=-1)
                        logger.info(f"Drew yellow circle for Screw 1 at: {current_screw['center']}, Radius: {radius}")
                        
                        while True:
                            current_index = sorted_screws.index(current_screw)
                            next_screw = None
                            
                            # 현재 스크류로부터 150 픽셀 이상 떨어진 SCREW 중 가장 가까운 SCREW 찾기
                            min_distance = float('inf')
                            for idx in range(current_index + 1, len(sorted_screws)):
                                screw = sorted_screws[idx]
                                distance = math.sqrt(
                                    (screw['center'][0] - current_screw['center'][0]) ** 2 +
                                    (screw['center'][1] - current_screw['center'][1]) ** 2
                                )
                                if distance >= 150:
                                    if distance < min_distance:
                                        min_distance = distance
                                        next_screw = screw
                                    break  # 첫 번째 150 이상 거리의 스크류를 찾으면 루프 종료
                            
                            if next_screw is None:
                                break  # 더 이상 연결할 SCREW가 없음
                            
                            next_index = sorted_screws.index(next_screw)
                            
                            # 선 그리기 (녹색 선)
                            cv2.line(warped_image, current_screw['center'], next_screw['center'], color=(0, 255, 0), thickness=1)
                            logger.info(f"Drew connecting line from Screw {current_index +1} to Screw {next_index +1}")
                            
                            # 거리 텍스트 계산
                            distance_text = f"{min_distance:.2f}px"
                            
                            # 선의 중간 지점 계산
                            mid_x = (current_screw['center'][0] + next_screw['center'][0]) // 2
                            mid_y = (current_screw['center'][1] + next_screw['center'][1]) // 2
                            
                            # 거리 텍스트 표시 (텍스트 배경 제거)
                            cv2.putText(warped_image, distance_text, (mid_x, mid_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            logger.info(f"Drew connecting distance between Screw {current_index +1} and Screw {next_index +1} at: ({mid_x}, {mid_y}) with distance {distance_text}")
                            
                            # 연결된 SCREW에 노란색 원 그리기
                            radius = compute_radius(next_screw['width'], next_screw['height'])
                            cv2.circle(warped_image, next_screw['center'], radius=radius, color=(0, 255, 255), thickness=1)
                            logger.info(f"Drew yellow circle for Screw {next_index +1} at: {next_screw['center']}, Radius: {radius}")
                            
                            # 연결된 스크류로 업데이트
                            current_screw = next_screw
                            connected_indices.add(next_index)
                            
                else:
                    logger.warning("Not enough screws to perform connecting distance calculations.")
                
            except Exception as e:
                logger.error(f"Error during SCREW distance calculations for {file.name}: {e}\n")
                continue  # 다음 파일로 넘어가기
            
            try:
                # 변경된 warped_image (마킹과 선이 그려진 이미지) 저장
                annotated_warped_filename = file.stem + '_annotated_warped.jpg'
                annotated_warped_path = result_directory / annotated_warped_filename
                save_image(annotated_warped_path, warped_image, logger)
                logger.info(f"Annotated image saved successfully: {annotated_warped_path}")
            except Exception as e:
                logger.error(f"Failed to save annotated image for {file.name}: {e}")
                continue  # 다음 파일로 넘어가기

def save_image(warped_path, warped_image, logger):
    """Pillow를 사용하여 이미지 저장"""
    try:
        # OpenCV 이미지를 RGB로 변환
        warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        # Pillow 이미지를 생성
        image_pil = Image.fromarray(warped_image_rgb)
        # Pillow로 이미지 저장 (파일 경로를 문자열로 변환)
        image_pil.save(str(warped_path))
        logger.info(f"Image saved successfully: {warped_path}")
    except Exception as e:
        logger.error(f"Failed to save image {warped_path}: {e}")

def select_directory():
    """Tkinter를 사용하여 디렉토리 선택"""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="이미지 및 JSON 파일이 있는 폴더를 선택하세요")
    return Path(directory) if directory else None

def main():
    source_directory = select_directory()
    if not source_directory:
        messagebox.showinfo("Info", "폴더가 선택되지 않았습니다. 프로그램을 종료합니다.")
        return
    
    result_directory = source_directory / 'RESULT_241216_offset_test'
    result_directory.mkdir(exist_ok=True)
    
    # OFFSET 서브 폴더는 process_images 함수 내에서 생성됩니다.
    
    # 로거 설정
    log_file_path = result_directory / 'process_log.txt'
    logger = setup_logger(log_file_path)
    
    logger.info(f"Processing started for directory: {source_directory}")
    
    process_images(source_directory, result_directory, logger)
    
    logger.info(f"Processing completed. Results saved in '{result_directory}' folder.")
    messagebox.showinfo("완료", f"이미지 처리가 완료되었습니다.\n결과는 '{result_directory}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
