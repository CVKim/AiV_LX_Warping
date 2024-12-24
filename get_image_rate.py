import os
from math import gcd
from PIL import Image

def get_aspect_ratio(width, height):
    """주어진 가로 및 세로 픽셀 수를 이용하여 비율을 계산"""
    common_divisor = gcd(width, height)
    simplified_width = width // common_divisor
    simplified_height = height // common_divisor
    return f"{simplified_width}:{simplified_height}"

def list_jpg_images(directory):
    """디렉토리 내의 모든 .jpg 파일을 리스트업"""
    jpg_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.jpg'):
            jpg_files.append(file)
    return jpg_files

def get_image_dimensions(image_path):
    """이미지의 가로 및 세로 픽셀 수를 반환"""
    with Image.open(image_path) as img:
        return img.width, img.height

def write_aspect_ratios_to_txt(directory, output_file):
    """이미지의 비율을 계산하여 텍스트 파일로 저장"""
    jpg_files = list_jpg_images(directory)
    
    if not jpg_files:
        print("지정된 디렉토리에 .jpg 파일이 없습니다.")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, file_name in enumerate(jpg_files, start=1):
            image_path = os.path.join(directory, file_name)
            try:
                width, height = get_image_dimensions(image_path)
                ratio = get_aspect_ratio(width, height)
                f.write(f"{idx}. {file_name} → {ratio}\n")
                f.write("---\n")
            except Exception as e:
                f.write(f"{idx}. {file_name} → 오류 발생: {e}\n")
                f.write("---\n")
    
    print(f"비율 정보가 '{output_file}' 파일에 저장되었습니다.")

def main():
    # 고정된 디렉토리 경로 설정 (Raw string 사용하여 백슬래시 문제 방지)
    directory = r"F:\000. PJT\06. LX\Img\241212\data"
    
    # 디렉토리가 존재하는지 확인
    if not os.path.isdir(directory):
        print("유효하지 않은 디렉토리 경로입니다.")
        return
    
    # 출력 파일 경로 설정
    output_file = os.path.join(directory, "output.txt")
    
    # 비율 계산 및 텍스트 파일로 저장
    write_aspect_ratios_to_txt(directory, output_file)

if __name__ == "__main__":
    main()
