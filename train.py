from functools import cache
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8s-ema.yaml')  # build a new model from YAML
    model.train(data=r"ultralytics/cfg/datasets/tomato.yaml",
                cache = False,
                epochs=100, 
                imgsz=640,
                batch=16,
                optimizer='Adam',
                workers = 8,
                amp = True
                )