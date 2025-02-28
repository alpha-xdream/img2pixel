import colorsys
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000 # 需要 numpy-1.22.4
import math

def rgb_to_lab(rgb):
    """ 将RGB颜色（0-255）转换为Lab对象 """
    # 将RGB归一化到0-1
    rgb = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    return convert_color(rgb, LabColor)
def rgbs_to_labs(rgb_array):
    """ 将RGB数组（0-255范围）转换为Lab数组 """
    lab_array = []
    for rgb in rgb_array:
        lab_color = rgb_to_lab(rgb)
        # lab_array.append([lab_color.lab_l, lab_color.lab_a, lab_color.lab_b]) # cKDTree使用
        lab_array.append(lab_color)
    # return np.array(lab_array) # cKDTree使用
    return lab_array

def find_nearest_color(pixel_rgb, palette_lab):
    """ 使用Delta E 2000找到调色板中最接近的颜色 """
    pixel_lab = rgb_to_lab(pixel_rgb)
    min_delta = float('inf')
    nearest_color = None
    for color_lab in palette_lab:
        delta = delta_e_cie2000(pixel_lab, color_lab)
        if delta < min_delta:
            min_delta = delta
            nearest_color = color_lab.get_value_tuple()
    return nearest_color

class ColorTheoryPalette:
    """ 基于色彩理论的程序化调色板生成器 """
    def __init__(self, base_hue, saturation=0.7, value_range=(0.3, 0.8)):
        """
        :param base_hue: 基色色相 (0-1)
        :param saturation: 饱和度 (0-1)
        :param value_range: 明度范围 (min, max)
        """
        self.base_hue = base_hue
        self.saturation = saturation
        self.value_min, self.value_max = value_range

    def generate(self, scheme_type="analogous", num_colors=6):
        """ 生成调色板 """
        hues = self._get_hues(scheme_type)
        colors = []
        for h in hues:
            for v in np.linspace(self.value_min, self.value_max, num_colors//len(hues)+1):
                rgb = colorsys.hsv_to_rgb(h, self.saturation, v)
                colors.append(tuple(int(x*255) for x in rgb))
        return list(set(colors))[:num_colors]  # 去重并截断

    def _get_hues(self, scheme_type):
        """ 根据配色方案生成色相列表 """
        if scheme_type == "analogous":    # 类似色
            return [self.base_hue, (self.base_hue + 0.08) % 1, (self.base_hue - 0.08) % 1]
        elif scheme_type == "complementary":  # 互补色
            return [self.base_hue, (self.base_hue + 0.5) % 1]
        elif scheme_type == "triadic":    # 三分色
            return [self.base_hue, (self.base_hue + 0.33) % 1, (self.base_hue + 0.66) % 1]
        elif scheme_type == "monochromatic":  # 单色
            return [self.base_hue]
        else:
            raise ValueError("Unsupported scheme type")

def auto_detect_base_hue(image):
    """ 从图像中自动检测主色色相 """
    np_img = np.array(image.convert("RGB"))
    pixels = np_img.reshape(-1, 3)
    dominant_color = pixels.mean(axis=0)  # 使用平均色作为基色
    h, _, _ = colorsys.rgb_to_hsv(*(dominant_color / 255))
    return h

# 根据图像对比度自动调整饱和度
def auto_adjust_saturation(image):
    np_img = np.array(image.convert("HSV"))
    return np_img[:, :, 1].mean() / 255
def generate_pixelart(
    input_path, output_path,
    target_size=(64, 64), scale_factor=8,
    palette_size=6, scheme_type="triadic"
):
    # 读取图像并检测基色
    img = Image.open(input_path)
    base_hue = auto_detect_base_hue(img)
    
    # 生成程序化调色板
    palette_gen = ColorTheoryPalette(
        base_hue=base_hue,
        saturation=auto_adjust_saturation(img),
        value_range=(0.0, 0.9)
    )
    palette = palette_gen.generate(scheme_type=scheme_type, num_colors=palette_size)

    # 预处理调色板的Lab值
    palette_lab = rgbs_to_labs(palette)
    
    # 下采样与颜色量化
    small_img = img.resize(target_size, Image.Resampling.BOX)
    np_img = np.array(small_img.convert("RGB"))
    
    # 使用KD树快速匹配颜色
    # tree = cKDTree(palette_lab)
    quantized = np.zeros_like(np_img)
    for y in range(np_img.shape[0]):
        for x in range(np_img.shape[1]):
            # pixel_lab = rgbs_to_labs([np_img[y, x]])[0]
            # dist, idx = tree.query(np_img[y, x])
            # quantized[y, x] = palette[idx]
            pixel_rgb = np_img[y, x].tolist()
            # 找到最接近的Lab颜色，再转回RGB
            nearest_lab = find_nearest_color(pixel_rgb, palette_lab)
            nearest_rgb = convert_color(LabColor(*nearest_lab), sRGBColor).get_value_tuple()
            nearest_rgb = [int(round(c*255)) for c in nearest_rgb]
            quantized[y, x] = nearest_rgb
    
    # 放大并保存
    Image.fromarray(quantized).resize(
        (target_size[0] * scale_factor, target_size[1] * scale_factor),
        Image.Resampling.NEAREST
    ).save(output_path)

scale = 1.5
# 使用示例：基于图像主色生成三分色方案
generate_pixelart(
    "input.png", "colorpalette.png",
    target_size=(math.ceil(83*scale), math.ceil(125*scale)),
    scale_factor=2,
    scheme_type="analogous", # analogous:类似色 complementary:互补色 triadic:三分色 monochromatic:单色
    palette_size=64
)