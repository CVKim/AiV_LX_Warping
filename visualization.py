import glob
import json
import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.validation import explain_validity

# 입력 및 출력 경로 설정
input_dir = r"F:\000. PJT\06. LX\Img\Exp12\test"
output_dir = r"F:\000. PJT\06. LX\Img\Exp12\test\result"

# 출력 디렉토리 생성
if not osp.exists(output_dir):
    os.makedirs(output_dir)

# 폴리곤 유효성 검사 및 자동 수정
def validate_polygon(polygon):
    if not polygon.is_valid:
        print(f"Invalid polygon detected: {explain_validity(polygon)}")
        polygon = polygon.buffer(0)  # 유효하지 않은 폴리곤을 수정
        if not polygon.is_valid:
            print("Polygon could not be fixed.")
            return None
    return polygon

# 레이블별로 병합된 폴리곤 얻기
def get_merged_polygons(shapes):
    polygons = []
    for ann in shapes:
        shape_type = ann['shape_type']
        points = ann['points']
        if shape_type == 'rectangle':
            # Rectangle을 Polygon으로 변환
            rect = cv2.minAreaRect(np.array(points, dtype=np.float32))
            box = cv2.boxPoints(rect)
            poly = Polygon(box)
        elif shape_type == 'polygon':
            poly = Polygon(points)
        else:
            raise NotImplementedError(f"There is no such shape-type({shape_type}) considered")
        
        poly = validate_polygon(poly)
        if poly is not None and not poly.is_empty:
            polygons.append(poly)
    
    if not polygons:
        return []
    
    # 모든 폴리곤을 병합
    merged = polygons[0]
    for poly in polygons[1:]:
        merged = merged.union(poly)
    
    # 만약 병합 결과가 MultiPolygon이면 리스트로 반환
    if merged.geom_type == 'Polygon':
        return [merged]
    elif merged.geom_type == 'MultiPolygon':
        return list(merged.geoms)
    else:
        return []

# 컨투어가 직선 형태인지 아닌지 검사하는 함수
def is_contour_straight(contour, epsilon=0.02):
    # 컨투어의 경로를 직선으로 근사화
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    return len(approx) <= 2  # 직선은 두 점으로 근사 가능

# 컨투어의 너비를 계산하는 함수
def get_contour_width(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w

# OBB 꼭지점 정렬 함수 (안정성을 위해 예외 처리 추가)
def get_ordered_box(contour):
    try:
        # 각 점의 (x, y) 합과 차이를 계산하여 꼭지점 정렬
        contour = contour.reshape(-1, 2)
        s = contour.sum(axis=1)
        diff = np.diff(contour, axis=1).reshape(-1)
        
        top_left = contour[np.argmin(s)]
        bottom_right = contour[np.argmax(s)]
        top_right = contour[np.argmin(diff)]
        bottom_left = contour[np.argmax(diff)]
        
        # Check for NaN values
        if np.isnan(top_left).any() or np.isnan(bottom_right).any() or np.isnan(top_right).any() or np.isnan(bottom_left).any():
            return None
        
        return [tuple(top_left), tuple(bottom_left), tuple(bottom_right), tuple(top_right)]
    except Exception as e:
        print(f"Error in get_ordered_box: {e}")
        return None

# OBB(Oriented Bounding Box) 계산 및 그리기
def draw_obb_from_contour(img, contour, label):
    # OBB 꼭지점 정렬
    ordered_box = get_ordered_box(contour)
    
    if ordered_box is None:
        print(f"Ordered box is invalid for label '{label}'. Skipping OBB drawing.")
        return
    
    # 사각형 그리기 (빨간색)
    cv2.polylines(img, [np.array(ordered_box, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # 꼭지점 표시 (파란색 원)
    for point in ordered_box:
        cv2.circle(img, point, 5, (255, 0, 0), -1)
    
    # 레이블 텍스트 그리기 (박스의 중심 위치에)
    center_x = np.mean([p[0] for p in ordered_box])
    center_y = np.mean([p[1] for p in ordered_box])
    
    # Check for NaN in center coordinates
    if np.isnan(center_x) or np.isnan(center_y):
        print(f"Center coordinates are NaN for label '{label}'. Skipping label text.")
    else:
        center_x = int(center_x)
        center_y = int(center_y)
        cv2.putText(
            img,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

# 레이블의 포인트를 그리기
def draw_label_points(img, shapes):
    for ann in shapes:
        label = ann['label']
        shape_type = ann['shape_type']
        points = ann['points']

        # 포인트 그리기 (녹색 원)
        for point in points:
            cv2.circle(img, (int(point[0]), int(point[1])), radius=3, color=(0, 255, 0), thickness=-1)

        # 레이블 텍스트 그리기 (첫 번째 포인트 위치에 표시, 녹색 텍스트)
        if len(points) > 0:
            cv2.putText(
                img,
                label,
                (int(points[0][0]), int(points[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

# Progress bar 추가 및 이미지 처리
img_files = glob.glob(osp.join(input_dir, '*.png'))
for img_file in tqdm(img_files, desc="Processing Images", unit="file"):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    json_file = osp.splitext(img_file)[0] + '.json'

    if osp.exists(json_file):
        # 원본 이미지 읽기 (한글 파일명 지원)
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        # JSON 파일 읽기
        with open(json_file, encoding='utf-8') as jf:
            anns = json.load(jf)['shapes']

        # 원본 이미지에 shapes 그리기
        for ann in anns:
            label = ann['label']
            shape_type = ann['shape_type']
            points = ann['points']

            if shape_type == 'rectangle':
                cv2.rectangle(
                    img,
                    (int(points[0][0]), int(points[0][1])),
                    (int(points[1][0]), int(points[1][1])),
                    (0, 0, 255), 2  # 두께를 2로 줄였습니다.
                )
            elif shape_type == 'polygon':
                cv2.fillPoly(
                    img,
                    [np.array(points, dtype=np.int32)],
                    color=(255, 255, 0)
                )
            else:
                raise NotImplementedError(f"There is no such shape-type({shape_type}) considered")

        # 레이블별로 그룹화
        label_to_shapes = {}
        for ann in anns:
            label = ann['label']
            if label not in label_to_shapes:
                label_to_shapes[label] = []
            label_to_shapes[label].append(ann)

        # 레이블별로 병합된 폴리곤 얻기 및 OBB 그리기
        for label, shapes in label_to_shapes.items():
            merged_polygons = get_merged_polygons(shapes)
            for polygon in merged_polygons:
                # 병합된 폴리곤의 컨투어 추출
                contour = np.array(polygon.exterior.coords, dtype=np.int32)
                contour = contour.reshape(-1, 1, 2)

                # 컨투어가 직선인지 아닌지 체크
                if is_contour_straight(contour):
                    print(f"Label '{label}': Contour is a straight line. Skipping OBB drawing.")
                    # 직선인 경우, 컨투어만 그리기
                    cv2.drawContours(img, [contour], 0, (0, 255, 255), 2)  # 노란색으로 그리기
                    draw_obb_from_contour(img, contour, label)
                else:
                    # 컨투어의 너비가 100 픽셀보다 작은지 체크
                    width_contour = get_contour_width(contour)
                    if width_contour < 100:
                        contour_color = (0, 0, 255)  # 빨간색
                        print(f"Label '{label}': Contour width {width_contour}px is less than 100px. Drawing in red.")
                    else:
                        contour_color = (255, 0, 0)  # 파란색

                    # 컨투어 그리기
                    cv2.drawContours(img, [contour], 0, contour_color, 2)

                    # OBB 계산 및 그리기
                    draw_obb_from_contour(img, contour, label)

        # 레이블의 포인트 그리기
        draw_label_points(img, anns)

        # 결과 이미지를 BMP로 저장 (한글 파일명 지원)
        save_path = osp.join(output_dir, filename + '.bmp')
        cv2.imencode('.bmp', img)[1].tofile(save_path)


        
# # 출력 디렉토리 생성
# if not osp.exists(output_dir):
#     os.makedirs(output_dir)

# # 랜덤 색상 생성 함수
# def generate_random_color():
#     return tuple(random.randint(0, 255) for _ in range(3))

# def validate_polygon(polygon):
#     if not polygon.is_valid:
#         print(f"Invalid polygon: {explain_validity(polygon)}")
#         return polygon.buffer(0)  # 자동 수정
#     return polygon

# def merge_shapes_by_label(shapes):
#     merged_shapes = {}
#     for ann in shapes:
#         label = ann['label']
#         points = ann['points']

#         # 폴리곤 객체 생성
#         polygon = Polygon(points)
#         polygon = validate_polygon(polygon)  # 유효성 검사 및 수정

#         if label not in merged_shapes:
#             merged_shapes[label] = [polygon]
#         else:
#             # 기존 폴리곤들과 병합 검사
#             new_polygons = []
#             merged = False
#             for existing_polygon in merged_shapes[label]:
#                 if polygon.intersects(existing_polygon):  # 교차 여부 확인
#                     polygon = polygon.union(existing_polygon)
#                     merged = True
#                 else:
#                     new_polygons.append(existing_polygon)
#             new_polygons.append(polygon)
#             merged_shapes[label] = new_polygons
#     return merged_shapes

# # 이미지 파일 검색 (한글 파일명 지원)
# img_files = glob.glob(osp.join(input_dir, '*.png'))

# # Progress bar 추가
# for img_file in tqdm(img_files, desc="Processing Images", unit="file"):
#     filename = osp.split(osp.splitext(img_file)[0])[-1]
#     json_file = osp.splitext(img_file)[0] + '.json'

#     # JSON 파일 존재 여부 확인
#     if osp.exists(json_file):
#         # 이미지 읽기
#         img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
#         height, width, _ = img.shape

#         # JSON 파일 읽기
#         with open(json_file, encoding='utf-8') as jf:
#             anns = json.load(jf)['shapes']

#         # 레이블별로 병합
#         merged_shapes = merge_shapes_by_label(anns)

#         # 병합 결과를 이미지에 그리기
#         for label, polygons in merged_shapes.items():
#             color = generate_random_color()
#             for polygon in polygons:
#                 # 폴리곤의 좌표를 가져와서 그리기
#                 points = np.array(list(polygon.exterior.coords), dtype=np.int32)
#                 cv2.fillPoly(img, [points], color=color)

#         # 결과 이미지를 BMP로 저장 (한글 파일명 지원)
#         save_path = osp.join(output_dir, filename + '.bmp')
#         cv2.imencode('.bmp', img)[1].tofile(save_path)
