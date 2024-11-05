import cv2
import numpy as np
import os.path as osp

class ImageStitcher:
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



    def stitch_images(self, img1_path, img2_path):
        img1 = self.read_image(img1_path)
        img2 = self.read_image(img2_path)

        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)

        good_matches = self.match_features(des1, des2)

        if len(good_matches) > self.min_matches:
            M = self.find_homography(kp1, kp2, good_matches)
            warpImg = self.warp_image(img2, M, (img1.shape[1] + img2.shape[1], max(img2.shape[0], img1.shape[0])))

            direct = self.direct_stitching(warpImg, img1)
            optimize = self.optimize_stitching(warpImg, img1)
            optimize = self.remove_black_borders(optimize)

            #cv2.imwrite(img1_path.split(".png")[0] + "direct.png", direct)
            cv2.imwrite(img1_path.split(".png")[0] + "optimize.png", optimize)
            print("result saved!")
            cv2.imshow("result", optimize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("not enough matches!")

    def stitch_multiple_images(self, image_paths):
        if len(image_paths) < 2:
            print("Need at least two images to stitch")
            return

        base_image_path = image_paths[0]
        for next_image_path in image_paths[1:]:
            self.stitch_images(base_image_path, next_image_path)
            base_image_path = base_image_path.split(".png")[0] + "optimize.png"

if __name__ == '__main__':
    script_path = osp.dirname(osp.abspath(__file__))
    project_path = osp.dirname(script_path)
    image_path = osp.join(project_path, 'images')

    stitcher = ImageStitcher(image_path)
    images = [
        osp.join(image_path, "2024-11-05_16:09:18.jpg"),
        osp.join(image_path, "2024-11-05_16:09:20.jpg"),
        osp.join(image_path, "2024-11-05_16:09:22.jpg"),
        osp.join(image_path, "2024-11-05_16:09:24.jpg"),
        osp.join(image_path, "2024-11-05_16:09:25.jpg"),
        osp.join(image_path, "2024-11-05_16:09:27.jpg"),
        osp.join(image_path, "2024-11-05_16:09:28.jpg"),

    ]
    stitcher.stitch_multiple_images(images)