import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# 이미지를 불러옵니다.
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 이미지를 BGR로 불러오기 때문에 RGB로 변환합니다.

# 원본 이미지의 크기
original_image_height, original_image_width, _ = image_rgb.shape

# src_points는 주어진 4개의 포인트 (좌상단, 우상단, 우하단, 좌하단)입니다.
src_points = np.array([[331, 297], [1079, 321], [1081, 337], [330, 320]], dtype=np.float32)

# 가상 공간의 크기를 원본 이미지 크기와 동일하게 설정
virtual_map_size = (original_image_width, original_image_height)
virtual_map = np.zeros(virtual_map_size, dtype=np.uint8)

# 좌상단 좌표
top_left = src_points[0]
top_right = src_points[1]
bottom_right = src_points[2]
bottom_left = src_points[3]

# x축 이동 간격 (임의로 설정한 10픽셀)
step = 10

# 가상 공간에서의 좌상단 좌표 (원본 이미지의 좌상단 좌표와 일치하게 설정)
virtual_top_left = top_left.copy()  # 원본 이미지 좌상단 좌표를 그대로 사용

# 초기 우상단 좌표
virtual_top_right_x = virtual_top_left[0]
virtual_top_right_y = virtual_top_left[1]

# 좌상단과 좌하단 사이의 초기 y 거리
initial_y_dist = np.linalg.norm(top_left - bottom_left)

# 가상 공간에서의 우상단 x 좌표 계산
total_steps = 0

for x in range(int(top_left[0]), int(top_right[0]), step):
    # 현재 x에서의 상단 y값 계산
    current_y_top = top_left[1] + (x - top_left[0]) * (top_right[1] - top_left[1]) / (top_right[0] - top_left[0])
    # 현재 x에서의 하단 y값 계산
    current_y_bottom = bottom_left[1] + (x - bottom_left[0]) * (bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0])
    
    # 이전 y 거리와 현재 y 거리 비율
    current_y_dist = current_y_bottom - current_y_top
    y_ratio = initial_y_dist / current_y_dist if current_y_dist != 0 else 1
    
    # 가상 공간에서의 우상단 x 좌표 업데이트
    virtual_top_right_x += step * y_ratio

    # 총 이동한 스텝 계산
    total_steps += step

# 최종적으로 우상단 좌표 계산
virtual_top_right = np.array([virtual_top_right_x, virtual_top_right_y])

# 가상 공간에서 좌상단 ~ 우상단을 y 방향으로 56픽셀만큼 확장한 결과
virtual_bottom_left = virtual_top_left + np.array([0, 56])
virtual_bottom_right = virtual_top_right + np.array([0, 56])

# 호모그래피 행렬 계산
dst_points = np.array([virtual_top_left, virtual_top_right, virtual_bottom_right, virtual_bottom_left], dtype=np.float32)
H, _ = cv2.findHomography(src_points, dst_points)

# 가상 공간에 이미지 삽입
warped_image = cv2.warpPerspective(image_rgb, H, (original_image_width, original_image_height))

# warped_image를 다시 BGR로 변환하여 저장
warped_image_bgr = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("./tmp.jpg", warped_image_bgr)

# 원본 이미지의 너비와 높이 계산
w_src = np.linalg.norm(top_right - top_left)
h_src = np.linalg.norm(bottom_left - top_left)

# 변환된 이미지의 너비와 높이 계산
w_dst = np.linalg.norm(virtual_top_right - virtual_top_left)
h_dst = np.linalg.norm(virtual_bottom_left - virtual_top_left)

# 비율 계산
width_ratio = w_dst / w_src
height_ratio = h_dst / h_src

# 결과 출력
print(f"Original Image Width (w_src): {w_src}")
print(f"Original Image Height (h_src): {h_src}")
print(f"Transformed Image Width (w_dst): {w_dst}")
print(f"Transformed Image Height (h_dst): {h_dst}")
print(f"Width Ratio: {width_ratio}")
print(f"Height Ratio: {height_ratio}")

# 결과를 보여줍니다.
plt.figure(figsize=(10, 10))

# 원본 이미지 표시
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.scatter(src_points[:, 0], src_points[:, 1], c='r', marker='o')
for i, txt in enumerate(src_points):
    plt.text(txt[0], txt[1], f'({txt[0]:.0f}, {txt[1]:.0f})', color='white', fontsize=12)
plt.xlim(0, original_image_width)
plt.ylim(original_image_height, 0)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 가상 공간에서 보정된 이미지 표시
plt.subplot(1, 2, 2)
plt.title("Corrected Image in Virtual Space")
plt.imshow(warped_image)
plt.scatter(dst_points[:, 0], dst_points[:, 1], c='b', marker='x')
for i, txt in enumerate(dst_points):
    plt.text(txt[0], txt[1], f'({txt[0]:.0f}, {txt[1]:.0f})', color='white', fontsize=12)
plt.xlim(0, original_image_width)
plt.ylim(original_image_height, 0)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()
