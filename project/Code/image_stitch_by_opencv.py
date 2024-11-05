# 用法
# python image_stitching.py --images images/scottsdale --output output.png --crop 1

# 导入必要的包
from imutils import paths
import numpy as np
import os.path as osp
import os
import imutils
import cv2
import argparse
import logging
import datetime

script_path = osp.dirname(osp.abspath(__file__))
project_path = osp.dirname(script_path)
image_path = osp.join(project_path, 'images')
output_path = osp.join(project_path, 'output')

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_images(image_path):
    logging.info("正在加载图像...")
    image_files = [osp.join(image_path, file) for file in os.listdir(image_path) if file.endswith('.jpg')]
    images = [cv2.imread(file) for file in image_files]
    if any(img is None for img in images):
        raise ValueError("加载图像时出错，请检查图像路径和文件格式。")
    return images

def stitch_images(images):
    logging.info("正在拼接图像...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status != 0:
        raise RuntimeError(f"图像拼接失败，错误代码：{status}")
    return stitched

def crop_stitched_image(stitched):
    logging.info("正在裁剪...")
    stitched = cv2.copyMakeBorder(stitched, 2, 2, 2, 2, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    stitched = stitched[y:y + h, x:x + w]
    return stitched

def main(args):
    try:
        images = load_images(args.images)
        stitched = stitch_images(images)
        if args.crop:
            stitched = crop_stitched_image(stitched)
            # 确保输出文件路径包含有效的扩展名
            valid_extensions = ['.png', '.jpg', '.jpeg']
            if not any(args.output.lower().endswith(ext) for ext in valid_extensions):
                args.output += '.png'
        cv2.imwrite(args.output, stitched)
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)
    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    parser = argparse.ArgumentParser(description="图像拼接脚本")
    parser.add_argument("--images", default=f"{image_path}", help="输入图像文件夹路径")
    parser.add_argument("--output", default=f"{output_path}/{timestamp}.png", help="输出拼接图像文件路径")
    parser.add_argument("--crop", type=int, default=1, help="是否裁剪拼接图像 (1: 是, 0: 否)")
    args = parser.parse_args()
    main(args)