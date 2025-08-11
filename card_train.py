from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    data_yaml_file = r"C:\Users\cbnits-304\OneDrive - NEXTZEN MINDS\Desktop\CREDIT_CARD_OCR\data.yaml"
    project_dir = r"C:\Users\cbnits-304\OneDrive - NEXTZEN MINDS\Desktop\CREDIT_CARD_OCR"
    res_folder = "Results"
    batch_size = 32

    result = model.train(
        data = data_yaml_file,
        batch = batch_size,
        epochs = 50,
        project = project_dir,
        name = res_folder,
        device = 'cpu',
        imgsz = 640,
        patience = 5,
        verbose =True,
        val =True
    )

if __name__ == "__main__":
    main()
