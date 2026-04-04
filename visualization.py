import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def stitch_images_with_labels(folder_path, output_path, line_spacing=10, label_height=30, font_size=12):
    """
    将文件夹中的图片拼接成大图，每行一张图片，左侧添加文件名标注

    参数:
        folder_path: 图片文件夹路径
        output_path: 输出图片路径
        line_spacing: 行间距
        label_height: 标注区域高度
        font_size: 标注字体大小
    """
    # 获取文件夹中所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(image_extensions)]

    if not image_files:
        print("文件夹中没有图片文件")
        return

    # 打开所有图片并获取尺寸
    images = []
    max_width = 0
    total_height = 0

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            with Image.open(img_path) as img:
                # 转换为RGB模式以统一处理
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append((img_file, img))
                # 计算最大宽度（图片宽度 + 标注宽度）
                img_width, img_height = img.size
                max_width = max(max_width, img_width)
                # 累加总高度（图片高度 + 标注高度 + 行间距）
                total_height += img_height + label_height + line_spacing
        except Exception as e:
            print(f"无法打开图片 {img_file}: {e}")

    # 减去最后一行多余的行间距
    total_height -= line_spacing

    # 创建输出图片（添加一定的标注宽度）
    label_width = 150  # 标注区域宽度
    output_width = max_width + label_width
    output_image = Image.new('RGB', (output_width, total_height), color='white')
    draw = ImageDraw.Draw(output_image)

    # 尝试加载字体，若失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 拼接图片并添加标注
    current_y = 0
    for img_name, img in images:
        img_width, img_height = img.size

        # 绘制标注文本（文件名）
        label_text = os.path.splitext(img_name)[0]  # 去除文件扩展名
        draw.text((10, current_y + (label_height - font_size) // 2),
                  label_text, font=font, fill='black')

        # 粘贴图片
        img_x = label_width
        img_y = current_y + label_height
        output_image.paste(img, (img_x, img_y))

        # 更新当前Y坐标
        current_y += label_height + img_height + line_spacing

    # 保存结果
    output_image.save(output_path)
    print(f"拼接完成，已保存至 {output_path}")


if __name__ == "__main__":
    # 示例用法
    input_folder = "visualization"  # 输入图片文件夹
    output_file = "visualization/result.jpg"  # 输出图片路径

    # 调用函数进行拼接
    stitch_images_with_labels(
        folder_path=input_folder,
        output_path=output_file,
        line_spacing=5,  # 较小的行间距
        label_height=25,
        font_size=50
    )
