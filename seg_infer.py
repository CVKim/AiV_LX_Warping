import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# 입력 이미지 전처리
def preprocess_image(image_path):
    input_image = Image.open(image_path)
    input_image = input_image.resize((512, 512))  # 모델 입력 크기에 맞게 조정
    input_array = np.array(input_image) / 255.0  # 정규화
    input_array = np.expand_dims(input_array, axis=0)  # 배치 차원 추가
    return input_array, input_image

# 모델을 사용한 추론
def inference(model, input_array):
    output = model.predict(input_array)
    output = np.argmax(output, axis=-1)[0]  # 예측 클래스 인덱스 추출
    return output

# 결과 시각화
def visualize_segmentation(output, original_image):
    # Cityscapes 색상 팔레트 사용
    palette = np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ])

    r = Image.fromarray(output.astype(np.uint8)).resize(original_image.size)
    r.putpalette(palette)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(r)
    plt.title('Segmented Image')

    plt.show()

# 모델 경로와 이미지 경로 설정
model_path = "./lx_deeplabv3_seg.h5"
image_path = "./easy/230913-3-1.jpg"

# 모델 로드
model = load_model(model_path)

# 이미지 전처리
input_array, original_image = preprocess_image(image_path)

# 추론 수행
output = inference(model, input_array)

# 결과 시각화
visualize_segmentation(output, original_image)
