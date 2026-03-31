from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data="perception/lane_detection/configs/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        project="perception/lane_detection/runs",
        name="lane_seg"
    )

if __name__ == "__main__":
    main()