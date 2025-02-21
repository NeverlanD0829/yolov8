import cv2
import os
from ultralytics import YOLO

def center_pass(input_image_dir,output_dir,model,image_files):
    for image_file in image_files:
        image_path = os.path.join(input_image_dir, image_file)
        results = model.predict(image_path)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        for i, r in enumerate(results):  
            boxes = r.boxes.xyxy.cpu().numpy()  
            class_ids = r.boxes.cls.cpu().numpy()  
            for j, (box, class_id) in enumerate(zip(boxes, class_ids)):
                if r.names[int(class_id)] == "rt": 
                    x1, y1, x2, y2 = map(int, box[:4])  
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    crop_size = 800
                    crop_x1 = max(center_x - crop_size // 2, 0)
                    crop_y1 = max(center_y - crop_size // 2, 0)
                    crop_x2 = min(center_x + crop_size // 2, width)
                    crop_y2 = min(center_y + crop_size // 2, height)
                    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_cropped_{i}_{j}.jpg")
                    cv2.imwrite(output_path, cropped_image)
                    print(f"Saved cropped image to {output_path}")

def main():
    model = YOLO("/home/chen/Desktop/yolo-V8/runs/detect/yolov8s-C2fiEMA-inMPDIOU-AFPN/weights/best.pt")  # 替换为你的模型路径
    input_image_dir = "/home/chen/Desktop/Tomato_dataset/datasets/tomato/images/doubletest/"  # 替换为你的图片文件夹路径
    output_dir = "MSTTGD_2 origin"
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    center_pass(input_image_dir,output_dir,model,image_files)

if __name__ == '__main__':
    main()