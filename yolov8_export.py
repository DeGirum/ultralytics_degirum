import argparse
from ultralytics import YOLO

def export_yolo_model(model_name, export_format, imgsz):
    model = YOLO(model_name)
    
    model.export(
        format=export_format,
        act='nn.ReLU6()',  # act='nn.SiLU()' by default
        simplify=True,
        export_hw_optimized=True,
        separate_6_outputs=True,
        imgsz=imgsz,
        int8=True,
        data="coco.yaml"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to a specific format.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the YOLO model.")
    parser.add_argument("--export_format", type=str, required=True, help="Export format (onnx, tflite).")
    parser.add_argument("--imgsz", type=int, required=True)
    
    args = parser.parse_args()
    
    export_yolo_model(args.model_name, args.export_format, args.imgsz)


from ultralytics import YOLO
model = YOLO("relu6-yolov8.yaml").load('yolov8n_relu6_300.pt')
model.export(format='onnx', 
            #  act='nn.ReLU6()',  # act='nn.SiLU()' by default
             simplify=True, 
             export_hw_optimized=True, 
             separate_6_outputs=True
            )

from ultralytics import YOLO
model = YOLO('relu6-yolov8-int8.tflite', task='detect')
model.val(data='coco.yaml')
