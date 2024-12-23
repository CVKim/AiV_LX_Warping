import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# 이미지를 불러옵니다.
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 이미지를 BGR로 불러오기 때문에 RGB로 변환합니다.

# 사용자 지정 4 포인트 (srcPoints)
src_points = np.array([[331, 297], [1079, 321], [1081, 337], [330, 320]], dtype=np.float32)

# src_points에서 width 계산
src_width = np.linalg.norm(src_points[1] - src_points[0])  # 상단 두 점의 거리로 너비 계산
src_height = 56  # 원래 고정된 높이

# width와 height에 3을 곱한 값
scaled_width = src_width * 3
scaled_height = src_height * 3

# dst_points는 확대된 width와 height를 사용하여 설정
dst_points = np.array([
    [0, 0],                # 좌상단
    [scaled_width, 0],     # 우상단
    [scaled_width, scaled_height],  # 우하단
    [0, scaled_height]     # 좌하단
], dtype=np.float32)

# 호모그래피 계산
H, status = cv2.findHomography(src_points, dst_points)

# 원본 이미지에 호모그래피 적용
# 변환 후 이미지의 크기는 계산된 확대된 너비와 높이를 사용
warped_image = cv2.warpPerspective(image_rgb, H, (int(scaled_width), int(scaled_height)))

# 결과를 보여줍니다.
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Warped Image")
plt.imshow(warped_image)
plt.axis('off')

plt.show()