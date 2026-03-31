def main():
    model = YOLO("perception/lane_detection/runs/lane_seg/weights/best.pt")
    results = model.predict(
        source="perception/lane_detection/datasets/images/val",
        save=True,
        project="perception/lane_detection/runs",
        name="lane_infer"
    )
    print(results)

if __name__ == "__main__":
    main()