import cv2
import numpy as np
import os

def main():
    # 获取图片路径
    image_path = "C:\\Users\\Desktop\\photo\\"
    image_names = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')]
    num_images = len(image_names)
    print("检索到的图片为：")
    for i, name in enumerate(image_names):
        print(f"Image #{i + 1}: {name}")
    print()

    # 存储图像、尺寸、特征点
    features = []
    images = []
    images_sizes = []
    featurefinder = cv2.SIFT_create()
    for i, name in enumerate(image_names):
        img = cv2.imread(name)
        images.append(img)
        images_sizes.append(img.shape[:2])
        keypoints, descriptors = featurefinder.detectAndCompute(img, None)
        features.append((keypoints, descriptors))
        print(f"image #{i + 1} 特征点为: {len(keypoints)} 个 尺寸为: {images_sizes[-1]}")
    print()

    # 图像特征点匹配
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairwise_matches = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            matches = matcher.knnMatch(features[i][1], features[j][1], k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            pairwise_matches.append((i, j, good_matches))

    # 预估相机参数
    homographies = []
    for i, j, matches in pairwise_matches:
        src_pts = np.float32([features[i][0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([features[j][0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies.append((i, j, H))

    # 光束平差，精确相机参数
    # 这里省略了光束平差的实现，可以使用第三方库如ceres-solver进行实现

    # 波形矫正
    # 这里省略了波形矫正的实现，可以使用OpenCV的waveCorrect函数进行实现

    # 创建mask图像
    masks = [np.ones(img.shape[:2], dtype=np.uint8) * 255 for img in images]

    # 图像、掩码变换
    warper = cv2.CylindricalWarper()
    masks_warp = []
    images_warp = []
    corners = []
    sizes = []
    for i in range(num_images):
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = 1  # 假设焦距为1
        corner, img_warp = warper.warp(images[i], K, np.eye(3), cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask_warp = warper.warp(masks[i], K, np.eye(3), cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        images_warp.append(img_warp)
        masks_warp.append(mask_warp)
        corners.append(corner)
        sizes.append(img_warp.shape[:2])
        print(f"Image #{i + 1} corner: {corner} size: {sizes[-1]}")
    print()

    # 图像融合
    blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
    blender.prepare(corners, sizes)
    for img_warp, mask_warp, corner in zip(images_warp, masks_warp, corners):
        img_warp = img_warp.astype(np.int16)
        blender.feed(img_warp, mask_warp, corner)
    result, result_mask = blender.blend(None, None)
    cv2.imwrite("result.jpg", result)

if __name__ == "__main__":
    main()