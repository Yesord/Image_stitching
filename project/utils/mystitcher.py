# import the necessary packages
import numpy as np
import imutils
import cv2

class MyStitcher:
	def __init__(self, image_path, min_matches=10):
		self.image_path = image_path
		self.min_matches = min_matches

	def read_image(self, img_path):
		return cv2.imread(img_path)

	def detect_and_compute(self, img):
		sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
		kp, des = sift.detectAndCompute(img, None)
		return kp, des

	def match_features(self, des1, des2):
		FLANN_INDEX_KDTREE = 0
		indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
		searchParams = dict(checks=32)
		flann = cv2.FlannBasedMatcher(indexParams, searchParams)
		matches = flann.knnMatch(des1, des2, k=2)
		good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]
		return good_matches

	def find_homography(self, kp1, kp2, good_matches):
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
		return M

	def warp_image(self, img, M, dsize):
		return cv2.warpPerspective(img, M, dsize)

	def remove_black_borders(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		x, y, w, h = cv2.boundingRect(contours[0])
		return img[y:y+h, x:x+w]

	def direct_stitching(self, warpImg, img1):
		direct = warpImg.copy()
		direct[0:img1.shape[0], 0:img1.shape[1]] = img1
		return direct

	def optimize_stitching(self, warpImg, img1):
		rows, cols = img1.shape[:2]
		left = 0
		right = cols

		for col in range(0, cols):
			if img1[:, col].any() and warpImg[:, col].any():
				left = col
				break

		res = np.zeros([rows, cols, 3], np.uint8)

		for row in range(0, rows):
			for col in range(0, right):
				if not img1[row, col].any():
					res[row, col] = warpImg[row, col]
				elif not warpImg[row, col].any():
					res[row, col] = img1[row, col]
				else:
					srcImgLen = float(abs(col - left))
					testImgLen = float(abs(col - right))
					alpha = srcImgLen / (srcImgLen + testImgLen)
					res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

		warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
		return warpImg

	def stitch_images(self, image_paths):
		if len(image_paths) < 2:
			print("Need at least two images to stitch")
			return

		base_image_path = image_paths[0]
		base_image = self.read_image(base_image_path)

		for next_image_path in image_paths[1:]:
			next_image = self.read_image(next_image_path)

			kp1, des1 = self.detect_and_compute(base_image)
			kp2, des2 = self.detect_and_compute(next_image)

			good_matches = self.match_features(des1, des2)

			if len(good_matches) > self.min_matches:
				M = self.find_homography(kp1, kp2, good_matches)
				warpImg = self.warp_image(next_image, M, (base_image.shape[1] + next_image.shape[1], max(next_image.shape[0], base_image.shape[0])))

				direct = self.direct_stitching(warpImg, base_image)
				optimize = self.optimize_stitching(warpImg, base_image)
				optimize = self.remove_black_borders(optimize)

				base_image = optimize
			else:
				print(f"Not enough matches between {base_image_path} and {next_image_path}!")
		result_path = image_paths[0].split(".png")[0] + "_stitched.png"
		cv2.imwrite(result_path, base_image)
		print(f"Result saved as {result_path}")
		cv2.imshow("result", base_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


class ImageStitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		# 利用 is_cv3 判断是否在使用 opencv3
		self.isv3 = imutils.is_cv3(or_better=True)

	def stitch(self, images, ratio=0.85, reprojThresh=6.0, showMatches=False):
		# 初始化结果图像为第一张图像
		result = images[0]
		# 初始化全局透视变换矩阵
		H_total = np.eye(3)
		# 初始化变换后的图像列表
		warp_image_list = [result]
		# 循环处理每一张图像
		for i in range(1, len(images)):
			imageA = images[i]
			imageB = result

			# 检测关键点并提取局部不变描述符
			(kpsA, featuresA) = self.detectAndDescribe(imageA)
			(kpsB, featuresB) = self.detectAndDescribe(imageB)

			# 匹配两幅图像的特征
			M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

			# 如果匹配为None，则没有足够的匹配的关键点用来创建全景图
			if M is None:
				return None

			# 否则（有足够的匹配的关键点），用透视图变换将图像拼接在一起
			(matches, H, status) = M
			# 更新全局透视变换矩阵
			H_total = np.dot(H,H_total)

			# 计算结果图像的尺寸
			# 去除imageB的黑边

			warpA = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
			warp_image_list.append(warpA)
			# cv2.imshow("warpA", warpA)
			# cv2.waitKey(0)
			print(warpA.shape)
			print(warp_image_list[0].shape)
			# 将 imageB 转换为灰度图像，并生成掩码，非黑区域值为1，黑色区域为0
			grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
			_, maskB = cv2.threshold(grayB, 1, 255, cv2.THRESH_BINARY)
			# 将掩码应用于 imageB，保留非黑部分
			imageB_masked = cv2.bitwise_and(imageB, imageB, mask=maskB)
			cv2.imshow("imageB_masked", imageB_masked)
			cv2.imshow("warpA", warpA)
			cv2.waitKey(0)
			# 将 imageB 的有效部分（非黑色区域）放入 warpA 中
			# 注意这里要使用掩码，将非黑区域覆盖到 warpA 中对应位置
			for y in range(imageB.shape[0]):
				for x in range(imageB.shape[1]):
					if maskB[y, x] > 0:  # 如果是非黑区域
						warpA[y, x] = imageB_masked[y, x]
			#warpA[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
			result = warpA
			# 去除图像的黑边
			result = self.optimize_stiching(warpA, imageB_masked)
			
			# 检查关键点匹配是否应该可视化
			if showMatches:
				vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

				cv2.imshow("Keypoint Matches", vis)
			# cv2.imshow("Result", result)
			# cv2.waitKey(0)
		# 返回拼接的图像
		return result, warp_image_list
	
	def exposure_compensation(self,img1, img2):
		gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		mean1 = np.mean(gray1)
		mean2 = np.mean(gray2)
		if mean1 == 0:
			mean1 = 1e-3
		if mean2 == 0:
			mean2 = 1e-3
		alpha = mean1 / mean2
		compensated_img2 = cv2.convertScaleAbs(img2, alpha=alpha, beta=0)
		return compensated_img2

	def optimize_stiching(self,warpImg, img1, use_exposure_compensation=True):
		rows, cols = img1.shape[:2]
		left, right = 0, cols

		left = next((col for col in range(cols) if img1[:, col].any() and warpImg[:, col].any()), 0)
		right = next((col for col in range(cols - 1, -1, -1) if img1[:, col].any() and warpImg[:, col].any()), cols)

		res = np.zeros([rows, cols, 3], np.uint8)

		if use_exposure_compensation:
			for row in range(rows):
				for col in range(cols):
					if not img1[row, col].any():
						res[row, col] = warpImg[row, col]
					elif not warpImg[row, col].any():
						res[row, col] = img1[row, col]
					else:
						compensated_pixel = self.exposure_compensation(img1[row:row+1, col:col+1], warpImg[row:row+1, col:col+1])
						res[row, col] = np.clip(img1[row, col] * 0.5 + compensated_pixel * 0.5, 0, 255)
		else:
			alpha_map = np.linspace(0, 1, right - left + 1)
			for row in range(rows):
				for col in range(cols):
					if not img1[row, col].any():
						res[row, col] = warpImg[row, col]
					elif not warpImg[row, col].any():
						res[row, col] = img1[row, col]
					else:
						if col < left:
							alpha = 0
						elif col > right:
							alpha = 1
						else:
							alpha = alpha_map[col - left]
						res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

		warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
		return warpImg



	def detectAndDescribe(self, image):
		# 定义dectAndDescribe函数
		# 将图片转为灰度
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# 检查是否使用opencv3
		if self.isv3:
			# 如果使用的是opencv3使用如下的
			# 检测图片中的关键点
			descriptor = cv2.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# 否则在opencv2.4使用如下的
		else:
			# 检测图片中的关键点
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# 从图像中提取特征
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# 将KeyPoint对象中的关键点转换为NumPy数组
		kps = np.float32([kp.pt for kp in kps])

		# 返回关键点和特征的元组
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# 计算原始匹配并初始化实际的匹配
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# 循环原始匹配点
		for m in rawMatches:
			# 确保距离在一定的比例范围内（即Lowe比率测试）
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# 计算一个转换矩阵至少需要4个匹配
		if len(matches) > 4:
			# 构造两组点
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# 计算两组点之间的变换矩阵
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# 将带有变换矩阵的匹配点和每个匹配点的状态一起返回
			return (matches, H, status)

		# 否则（即匹配的点小于4个）不能计算变换矩阵
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# 初始化输出的可视化图像
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8") # 8位的三维0数组
		# 将两张图像放在同一张图的左右部分
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# 在匹配点中循环操作
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# 仅在关键点成功匹配时进行匹配
			if s == 1:
				# 画出匹配关系
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# 返回输出可视化之后的图像
		return vis
	


class mask_Stitcher:
	def __init__(self) :
		self.ratio=0.85
		self.min_match=10
		self.sift=cv2.SIFT_create()
		self.smoothing_window_size=800

	def registration(self,img1,img2):
		kp1, des1 = self.sift.detectAndCompute(img1, None)
		kp2, des2 = self.sift.detectAndCompute(img2, None)
		matcher = cv2.BFMatcher()
		raw_matches = matcher.knnMatch(des1, des2, k=2)
		good_points = []
		good_matches=[]
		for m1, m2 in raw_matches:
			if m1.distance < self.ratio * m2.distance:
				good_points.append((m1.trainIdx, m1.queryIdx))
				good_matches.append([m1])
		img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
		cv2.imwrite('matching.jpg', img3)
		if len(good_points) > self.min_match:
			image1_kp = np.float32(
				[kp1[i].pt for (_, i) in good_points])
			image2_kp = np.float32(
				[kp2[i].pt for (i, _) in good_points])
			H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
		return H

	def create_mask(self,img1,img2,version):
		height_img1 = img1.shape[0]
		width_img1 = img1.shape[1]
		width_img2 = img2.shape[1]
		height_panorama = height_img1
		width_panorama = width_img1 +width_img2
		offset = int(self.smoothing_window_size / 2)
		barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
		mask = np.zeros((height_panorama, width_panorama))
		if version== 'left_image':
			mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
			mask[:, :barrier - offset] = 1
		else:
			mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
			mask[:, barrier + offset:] = 1
		return cv2.merge([mask, mask, mask])

	def blending(self,img1,img2):
		H = self.registration(img1,img2)
		height_img1 = img1.shape[0]
		width_img1 = img1.shape[1]
		width_img2 = img2.shape[1]
		height_panorama = height_img1
		width_panorama = width_img1 +width_img2

		panorama1 = np.zeros((height_panorama, width_panorama, 3))
		mask1 = self.create_mask(img1,img2,version='left_image')
		panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
		panorama1 *= mask1
		mask2 = self.create_mask(img1,img2,version='right_image')
		panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
		result=panorama1+panorama2

		rows, cols = np.where(result[:, :, 0] != 0)
		min_row, max_row = min(rows), max(rows) + 1
		min_col, max_col = min(cols), max(cols) + 1
		final_result = result[min_row:max_row, min_col:max_col, :]
		return final_result
	
	def stitch(self, images):
		result = images[0]
		for i in range(1, len(images)):
			result = self.blending(result, images[i])
			if result is None:
				print("Error in stitching images")
				return None
		return result