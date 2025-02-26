from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import math

def image_to_pixelart(input_path, output_path, target_size=(64, 64), colors=16, scale_factor=4):
    # 打开图像并转换为RGB
    img = Image.open(input_path).convert('RGB')
    np_img = np.array(img)
    
    # 1. 下采样：调整到目标尺寸，使用平均颜色
    small_img = img.resize(target_size, Image.Resampling.BOX)
    
    # 2. 颜色量化
    pixels = np.array(small_img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=colors).fit(pixels)
    labels = kmeans.labels_
    palette = kmeans.cluster_centers_.astype('uint8')
    quantized_img = palette[labels].reshape(target_size[1], target_size[0], 3)
    
    # 转换为PIL图像并放大
    pixelart = Image.fromarray(quantized_img).resize(
        (target_size[0] * scale_factor, target_size[1] * scale_factor),
        Image.Resampling.NEAREST
    )
    pixelart.save(output_path)

if __name__ == '__main__':
    scale = 1.2
    width = math.ceil(83//scale)
    height = math.ceil(125//scale)
    colors = 500
    imageScale = 4 # math.ceil(20*scale)
    image_to_pixelart('input.png', 'output.png', target_size=(width, height), colors=colors, scale_factor=imageScale)