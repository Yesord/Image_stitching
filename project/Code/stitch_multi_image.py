from stitching import Stitcher
import os.path as osp
import os
import cv2
script_path = osp.dirname(osp.abspath(__file__))
project_path = osp.dirname(script_path)
image_path = osp.join(project_path, 'images')
images = [cv2.imread(osp.join(image_path, file)) for file in os.listdir(image_path) if file.endswith('.jpg')]

stitcher = Stitcher()

stitcher = Stitcher(detector="sift", confidence_threshold=0.5)

panorama = stitcher.stitch(images)

cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()