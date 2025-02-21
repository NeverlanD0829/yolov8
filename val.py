import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/chen/Desktop/yolo-V8/yolov8n.pt')
    model.val(data=r'/home/chen/Desktop/yolo-V8/ultralytics/cfg/datasets/sun.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )