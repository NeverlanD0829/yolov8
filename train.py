from functools import cache
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # model = YOLO('yolov8m.yaml')  # build a new model from YAML
    # model = YOLO('yolov8s-p2.yaml')  # build a new model from YAML
    # model = YOLO('yolov8s-C2fiEMA.yaml')  # build a new model from YAML
    # model = YOLO('yolov8s-C2fiEMA-inMPDIOU.yaml')  # build a new model from YAML
    # model = YOLO('yolov8s-C2fiEMA-inMPDIOU-AFPN.yaml')  # build a new model from YAML
    model.train(data=r"ultralytics/cfg/datasets/tomato.yaml",
                cache = False,
                epochs=100, 
                imgsz=640,
                batch=32,
                optimizer='Adam',
                workers = 12,
                amp = True
                )