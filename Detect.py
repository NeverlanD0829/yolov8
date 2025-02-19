import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/chen/Desktop/yolo-V8/runs/detect/yolov8s-C2fiEMA-inMPDIOU.yaml/weights/best.pt') # select your model.pt path
    model.predict(source='/home/chen/Desktop/Tomato_dataset/datasets/tomato/images/test',
                  imgsz=640,
                  project='runs/predict',
                  name='exp',
                  save=True,
                  # classes=0, 是否指定检测某个类别.
                )