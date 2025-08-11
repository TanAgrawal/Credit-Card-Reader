import cv2
import easyocr
import re
import numpy as np
from ultralytics import YOLO

model = YOLO(r"C:\Users\cbnits-304\OneDrive - NEXTZEN MINDS\Desktop\CREDIT_CARD_OCR\Results2\weights\best.pt")
reader = easyocr.Reader(['en'], gpu=False)

REQUIRED_FIELDS = {'number', 'name', 'exp', 'expiry', 'exp date'}

CHAR_CORRECTIONS = {
    'number': {
        'u': '0', 'U': '0',
        'o': '0', 'O': '0',
        'i': '1', 'I': '1', 'l': '1', 'L': '1',
        'b': '6',
        's': '5', 'S': '5',
        'z': '2', 'Z': '2'
    },
    'name': {
        '0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'
    }
}

def expand_box(x1, y1, x2, y2, img_shape, pad=5):
    h, w = img_shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2

def enhance_expiry_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def aggregate_text(results):
    return ' '.join([line[1] for line in results])

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 /-]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def correct_text(text, field_type):
    mapping = CHAR_CORRECTIONS.get(field_type.lower(), {})
    corrected = ''.join([mapping.get(c, c) for c in text])
    return corrected

def extract_expiry_from_results(results):
    full_text = ' '.join([line[1] for line in results])
    full_text = clean_text(full_text)
    patterns = [
        r'\b(0[1-9]|1[0-2])[/\-](\d{2,4})\b',
        r'\b(0[1-9]|1[0-2])(\d{2,4})\b',
        r'\b(0[1-9]|1[0-2])\s(\d{2,4})\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, full_text)
        if match:
            return match.group(0)
    return 'N/A'

def process_frame(frame, results):
    annotated = frame.copy()
    field_data = {}
    detected_labels = set()

    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id].lower()
            detected_labels.add(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, frame.shape)
            roi = frame[y1:y2, x1:x2]

            if label in ['exp', 'expiry', 'exp date']:
                roi_proc = enhance_expiry_image(roi)
                ocr_result = reader.readtext(roi_proc)
                text = extract_expiry_from_results(ocr_result)
            else:
                ocr_result = reader.readtext(roi)
                raw_text = aggregate_text(ocr_result)
                text = clean_text(raw_text)

            text = correct_text(text, label)

            field_data[label] = text

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label}: {text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, field_data, detected_labels

cap = cv2.VideoCapture(0)
print("Press 'c' to capture when screen turns green. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    annotated, _, detected_labels = process_frame(frame, results)

    ready_to_capture = any(label in detected_labels for label in ['number']) and \
                       any(label in detected_labels for label in ['name']) and \
                       any(label in detected_labels for label in ['exp', 'expiry', 'exp date'])

    if ready_to_capture:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 255, 0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        cv2.putText(annotated, "Card aligned! Press 'c' to capture", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Credit Card OCR", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and ready_to_capture:
        final_results = model.predict(frame, verbose=False)
        final_annotated, field_data, _ = process_frame(frame, final_results)

        print("\nExtracted Fields:")
        for key, value in field_data.items():
            print(f"{key}: {value}")

        cv2.imshow("Captured Frame", final_annotated)
        cv2.imwrite("captured_card.png", final_annotated)
        cv2.waitKey(0)  

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
