import math

# 주어진 데이터
focal_length_mm = 2.20  # 초점 거리 in mm
image_width_px = 4000  # 이미지 가로 해상도 in pixels
fov_deg = 75  # 가정: 대략적인 FOV in degrees

# FOV를 라디안으로 변환
fov_rad = math.radians(fov_deg)

# 가로 FOV 계산 (센서 크기는 focal_length_mm을 사용해 가정)
sensor_width_mm = 2 * focal_length_mm * math.tan(fov_rad / 2)

# 한 픽셀당 mm 계산
pixel_size_mm = sensor_width_mm / image_width_px

# 56mm가 몇 픽셀인지 계산
target_distance_mm = 56
target_distance_px = target_distance_mm / pixel_size_mm

pixel_size_mm, target_distance_px


print (f"pixel_size_mm : {pixel_size_mm}, {target_distance_px}")