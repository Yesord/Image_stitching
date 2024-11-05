import cv2
import numpy as np
import os.path as osp

script_path = osp.dirname(osp.abspath(__file__))
project_path = osp.dirname(script_path)
image_path = osp.join(project_path, 'images')


import cv2
import numpy as np
from scipy.optimize import least_squares

def stitch_images(images):
    # Step 1: Feature Detection and Matching
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Step 2: Feature Matching between consecutive images
    matcher = cv2.BFMatcher()
    matches_list = []

    for i in range(len(images) - 1):
        matches = matcher.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        matches_list.append(good_matches)

    # Step 3: Calculate Homography and Warp Images
    result = images[0]
    for i in range(1, len(images)):
        src_pts = np.float32([keypoints_list[i - 1][m.queryIdx].pt for m in matches_list[i - 1]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_list[i][m.trainIdx].pt for m in matches_list[i - 1]]).reshape(-1, 1, 2)

        # Use RANSAC to estimate a robust homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply a bundle adjustment-like optimization to refine homography
        H = refine_homography(H, src_pts, dst_pts, mask)

        # Warp the current image onto the accumulated result
        height, width = result.shape[:2]
        warped_image = cv2.warpPerspective(images[i], H, (width + images[i].shape[1], height))

        # Combine images using multi-band blending to reduce seams
        result = multi_band_blend(result, warped_image)

    return result

def refine_homography(H, src_pts, dst_pts, mask):
    # Function to optimize homography matrix using least squares
    def residuals(h, src_pts, dst_pts):
        H = h.reshape(3, 3)
        src_pts_homog = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
        projected_pts = np.dot(H, src_pts_homog.T).T
        projected_pts[:, :2] /= projected_pts[:, 2][:, np.newaxis]
        return (projected_pts[:, :2] - dst_pts).ravel()

    src_pts = src_pts[mask.ravel() == 1].reshape(-1, 2)
    dst_pts = dst_pts[mask.ravel() == 1].reshape(-1, 2)
    h0 = H.ravel()
    res = least_squares(residuals, h0, args=(src_pts, dst_pts))
    return res.x.reshape(3, 3)

def multi_band_blend(img1, img2, levels=5):
    # Generate Gaussian pyramid for img1
    G1 = img1.copy()
    gp1 = [G1]
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        gp1.append(G1)

    # Generate Gaussian pyramid for img2
    G2 = img2.copy()
    gp2 = [G2]
    for i in range(levels):
        G2 = cv2.pyrDown(G2)
        gp2.append(G2)

    # Generate Laplacian Pyramid for img1
    lp1 = [gp1[-1]]
    for i in range(levels-1, 0, -1):
        GE = cv2.pyrUp(gp1[i])
        GE = cv2.resize(GE, (gp1[i-1].shape[1], gp1[i-1].shape[0]))
        L = cv2.subtract(gp1[i-1], GE)
        lp1.append(L)

    # Generate Laplacian Pyramid for img2
    lp2 = [gp2[-1]]
    for i in range(levels-1, 0, -1):
        GE = cv2.pyrUp(gp2[i])
        GE = cv2.resize(GE, (gp2[i-1].shape[1], gp2[i-1].shape[0]))
        L = cv2.subtract(gp2[i-1], GE)
        lp2.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for l1, l2 in zip(lp1, lp2):
        rows, cols, dpt = l1.shape
        ls = np.hstack((l1[:, :cols // 2], l2[:, cols // 2:]))
        LS.append(ls)

    # Reconstruct
    ls_ = LS[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])

    return ls_

# Example usage
if __name__ == "__main__":
    # Load input images
    img1 = cv2.imread(image_path + '/2024-11-04_14:59:31.jpg')
    img2 = cv2.imread(image_path + '/2024-11-04_14:59:35.jpg')
    img3 = cv2.imread(image_path + '/2024-11-04_14:59:37.jpg')

    images = [img1, img2, img3]

    # Stitch images
    result = stitch_images(images)

    # Save and display result
    cv2.imwrite("stitched_result.jpg", result)
    cv2.imshow("Stitched Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # image1 = cv2.imread(image_path + '/2024-11-04_14:59:31.jpg')
    # image2 = cv2.imread(image_path + '/2024-11-04_14:59:35.jpg')
    # image3 = cv2.imread(image_path + '/2024-11-04_14:59:37.jpg')
    
