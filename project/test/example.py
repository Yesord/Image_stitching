import cv2
import numpy as np
import os
import os.path as osp
import glob

script_path = osp.dirname(osp.abspath(__file__))
project_path = osp.dirname(script_path)
image_path = osp.join(project_path, 'images')

def main():
    # 获取图片路径
    filepath = image_path + "/*.jpg"  # 图片存储路径
    image_names = glob.glob(filepath)  # 所有图片名字
    num_images = len(image_names)  # 图片数量
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
        features.append(cv2.detail.computeImageFeatures2(featurefinder, img))
        print(f"image #{i + 1} 特征点为: {len(keypoints)} 个 尺寸为: {images_sizes[-1]}")
    print()

    # 图像特征点匹配
    matcher = cv2.detail.BestOf2NearestMatcher()
    pairwise_matches = matcher.apply2(features)

    # 预估相机参数
    estimator = cv2.detail_HomographyBasedEstimator()
    cameras = []
    success, cameras = estimator.apply(features, pairwise_matches, cameras)

    if not success:
        print("相机参数估计失败")
        return

    # 确保相机参数的旋转矩阵类型为 CV_32F
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    # 光束平差，精确相机参数
    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), dtype=np.uint8)
    refine_mask[0, 0] = 1
    refine_mask[0, 1] = 1
    refine_mask[0, 2] = 1
    refine_mask[1, 1] = 1
    adjuster.setRefinementMask(refine_mask)
    success, cameras = adjuster.apply(features, pairwise_matches, cameras)

    if not success:
        print("光束平差失败")
        return

    print("精确相机参数")
    for i, cam in enumerate(cameras):
        print(f"camera #{i + 1}:\n内参数矩阵K:\n{cam.K()}\n旋转矩阵R:\n{cam.R}\n焦距focal: {cam.focal}")
    print()

    # 波形矫正
    rmats = [cam.R for cam in cameras]
    rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
    for i in range(num_images):
        cameras[i].R = rmats[i]

    # 创建mask图像
    masks = [np.ones(img.shape[:2], dtype=np.uint8) * 255 for img in images]

    # 图像、掩码变换
    warper = cv2.PyRotationWarper('cylindrical', cameras[0].focal)
    masks_warp = []
    images_warp = []
    corners = []
    sizes = []
    for i in range(num_images):
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = cameras[i].focal  # 使用相机的焦距
        corner, img_warp = warper.warp(images[i], K, cameras[i].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask_warp = warper.warp(masks[i], K, cameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        images_warp.append(img_warp)
        masks_warp.append(mask_warp)
        corners.append(corner)
        sizes.append(img_warp.shape[:2])
        print(f"Image #{i + 1} corner: {corner} size: {sizes[-1]}")
    

    # 图像融合
    blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
    blender.prepare(corners, sizes)
    for img_warp, mask_warp, corner in zip(images_warp, masks_warp, corners):
        img_warp = img_warp.astype(np.int16)
        blender.feed(img_warp, mask_warp, corner)
    result, result_mask = blender.blend(None, None)
    print("图像融合完成")
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()