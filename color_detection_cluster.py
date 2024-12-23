import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

def imread_unicode(filepath):
    """한글 경로를 포함한 파일을 읽기 위해 바이너리 모드로 파일을 읽고 imdecode를 사용하여 이미지를 로드합니다."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")
        return img
    except Exception as e:
        print(f"이미지를 로드하는 동안 오류가 발생했습니다: {e}")
        exit()

def resize_image(image, max_size=1024):
    """
    이미지의 종횡비를 유지하면서 최대 크기를 max_size로 리사이즈합니다.
    """
    height, width = image.shape[:2]
    if max(height, width) <= max_size:
        print("이미지 리사이즈가 필요하지 않습니다.")
        return image
    if height > width:
        new_height = max_size
        new_width = int((width / height) * max_size)
    else:
        new_width = max_size
        new_height = int((height / width) * max_size)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print(f"이미지를 {new_width}x{new_height}로 리사이즈했습니다.")
    return resized_image

def load_image(resize=False, max_size=1024):
    """파일 다이얼로그를 통해 이미지를 선택하고 로드합니다. 필요 시 리사이즈."""
    Tk().withdraw()  # Tkinter 창을 숨깁니다.
    file_path = filedialog.askopenfilename(
        title="이미지 파일 선택",
        filetypes=[("JPEG 파일", "*.jpg;*.jpeg"), ("PNG 파일", "*.png"), ("모든 파일", "*.*")]
    )
    if not file_path:
        print("파일이 선택되지 않았습니다.")
        exit()
    image = imread_unicode(file_path)
    
    if resize:
        image = resize_image(image, max_size=max_size)
    
    return image

def select_roi(image):
    """사용자가 ROI를 선택할 수 있도록 합니다."""
    print("이미지 창에서 ROI를 선택한 후 Enter 키를 누르세요.")
    roi = cv2.selectROI("이미지", image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("이미지")
    if roi == (0,0,0,0):
        print("ROI가 선택되지 않았습니다.")
        exit()
    x, y, w, h = roi
    return image[y:y+h, x:x+w], roi

def pad_image(image, patch_size=256):
    """패치 단위로 분할할 수 있도록 이미지를 제로 패딩합니다."""
    height, width = image.shape[:2]
    pad_height = (patch_size - height % patch_size) if height % patch_size != 0 else 0
    pad_width = (patch_size - width % patch_size) if width % patch_size != 0 else 0
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0,0,0])
    return padded_image, pad_width, pad_height

def remove_padding(padded_image, original_height, original_width):
    """제로 패딩을 제거하여 원래의 ROI 크기로 되돌립니다."""
    return padded_image[0:original_height, 0:original_width]

def divide_into_patches(image, patch_size=256):
    """이미지를 patch_size x patch_size 크기의 패치로 분할합니다."""
    patches = []
    positions = []  # 패치의 시작 좌표 (x, y)
    height, width = image.shape[:2]
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((x, y))
    
    return patches, positions

def process_patches_threshold(image, patch_size=256, g_threshold=200, hsv_lower=(35, 100, 100), hsv_upper=(85, 255, 255)):
    """
    이미지를 패치 단위로 처리하여 그린 픽셀을 마스크로 생성하고, 이를 원본 이미지에 적용합니다.
    HSV 색상 공간을 사용하여 그린 픽셀을 정의합니다.
    """
    patches, positions = divide_into_patches(image, patch_size)
    result_image = image.copy()
    
    # HSV 색상 공간으로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    for idx, (patch, (x, y)) in enumerate(zip(patches, positions)):
        print(f"Processing patch {idx + 1}/{len(patches)} at position x:{x}, y:{y}")
        
        # 패치의 HSV 추출
        hsv_patch = hsv_image[y:y+patch.shape[0], x:x+patch.shape[1]]
        
        # 그린 색상 범위에 해당하는 마스크 생성
        mask = cv2.inRange(hsv_patch, hsv_lower, hsv_upper)
        
        # 마스크에 대한 제로 패딩 처리 필요 없음 (이미 패치 단위로 처리)
        
        # G 값 기준 추가 필터링 (필요 시)
        # 예: G > 200
        g_channel = patch[:, :, 1]
        mask_g = g_channel > g_threshold
        mask = cv2.bitwise_and(mask, mask, mask=mask_g.astype(np.uint8)*255)
        
        if not np.any(mask):
            print(f"Patch {idx + 1}: 그린 픽셀이 없어 건너뜁니다.")
            continue
        
        # 모폴로지 연산을 적용하여 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 마스크를 원본 이미지에 적용하여 그린 픽셀을 강조
        result_image[y:y+patch.shape[0], x:x+patch.shape[1]][mask == 255] = [0, 255, 0]  # BGR에서 순수 그린
    
    return result_image

def apply_mask_direct(original_image, mask, roi_coordinates):
    """
    마스크를 원본 이미지에 직접 적용하여 그린 영역을 강조합니다.
    """
    x, y, w, h = roi_coordinates
    mask_full = np.zeros(original_image.shape[:2], dtype=np.uint8)
    mask_full[y:y+h, x:x+w] = mask
    
    # 원본 이미지에 마스크를 적용하여 그린 영역을 강조
    result = original_image.copy()
    
    # 마스크가 255인 곳을 그린으로 변경
    result[mask_full == 255] = [0, 255, 0]  # BGR에서 순수 그린
    
    return result

def main():
    # 1. 이미지 로드 및 리사이즈 (종횡비 유지)
    image = load_image(resize=True, max_size=1024)
    if image is None:
        print("이미지를 로드하는 데 실패했습니다.")
        exit()

    # 2. ROI 선택
    roi, roi_coords = select_roi(image)
    original_roi_height, original_roi_width = roi.shape[:2]

    # 3. ROI 패딩
    patch_size = 256
    padded_roi, pad_w, pad_h = pad_image(roi, patch_size=patch_size)
    print(f"Padded ROI size: {padded_roi.shape[1]}x{padded_roi.shape[0]} (width x height)")

    # 4. 그린 픽셀 식별 및 마스크 생성
    # HSV 색상 범위 정의 (그린 색상)
    # Hue: 약 35-85 (0-179 스케일)
    # Saturation: 약 100 이상
    # Value: 약 100 이상
    hsv_lower = (35, 100, 100)
    hsv_upper = (85, 255, 255)
    
    processed_padded_roi = process_patches_threshold(
        padded_roi, 
        patch_size=patch_size, 
        g_threshold=200, 
        hsv_lower=hsv_lower, 
        hsv_upper=hsv_upper
    )

    # 5. 패딩 제거
    processed_roi = remove_padding(processed_padded_roi, original_roi_height, original_roi_width)

    # 6. 원본 이미지에 처리된 ROI 적용
    result_image = image.copy()
    x, y, w, h = roi_coords
    result_image[y:y+h, x:x+w] = processed_roi

    # 7. 결과 출력
    cv2.imshow("원본 이미지", image)
    cv2.imshow("선택된 ROI", roi)
    cv2.imshow("처리된 ROI (그린 픽셀 강조)", processed_roi)
    cv2.imshow("최종 결과 이미지", result_image)
    print("결과 이미지를 확인한 후 아무 키나 누르세요.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 8. 결과 이미지 저장
    Tk().withdraw()  # Tkinter 창을 숨깁니다.
    save_path = filedialog.asksaveasfilename(
        title="결과 이미지 저장",
        defaultextension=".jpg",
        filetypes=[("JPEG 파일", "*.jpg;*.jpeg"), ("PNG 파일", "*.png"), ("모든 파일", "*.*")]
    )
    if save_path:
        cv2.imwrite(save_path, result_image)
        print(f"결과 이미지를 저장했습니다: {save_path}")
    else:
        print("결과 이미지가 저장되지 않았습니다.")

if __name__ == "__main__":
    main()