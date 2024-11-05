import cv2
from datetime import datetime
import os.path as osp

script_path = osp.dirname(osp.abspath(__file__))
project_path = osp.dirname(script_path)
image_path = osp.join(project_path, 'images')

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # if not ret:
        #     print('Failed to capture image')
        #     return
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            cv2.imwrite(f"{image_path}/{current_time}.jpg", frame)
            print(f"{current_time}.jpg Image saved")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()