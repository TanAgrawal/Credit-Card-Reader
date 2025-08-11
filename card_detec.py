from ultralytics import YOLO
import cv2 as cv
import yaml
import numpy as np
import easyocr
import re

def enhance_expiry_image(gray_img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_img)

    kernel = np.ones((2, 2), np.uint8)
    dilated = cv.dilate(enhanced, kernel, iterations=1)

    _, thresh = cv.threshold(dilated, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def extract_expiry_from_results(results):
    for line in results:
        if isinstance(line, tuple):
            text = line[1]
        else:
            text = line

        match = re.search(r'\b(0[1-9]|1[0-2])[/\-](\d{2,4})\b', text)
        if match:
            return match.group(0)
    return 'N/A'

def clean_text(label, raw_text, full_results=None):
    if label == 'number':
        digits = re.findall(r'\d+', raw_text)
        return ' '.join(digits) if digits else 'N/A'

    elif label == 'name':
        cleaned = re.sub(r'[^A-Za-z ]', '', raw_text)
        return cleaned.upper() if cleaned else 'N/A'

    elif label == 'Exp date' and full_results:
        return extract_expiry_from_results(full_results)

    return raw_text.strip() or 'N/A'

reader = easyocr.Reader(['en'])

img_path = r"C:\Users\cbnits-304\OneDrive - NEXTZEN MINDS\Desktop\CREDIT_CARD_OCR\train\images\979ad884-e163-4845-a779-2dee13d257b2_jpeg.rf.ebed0d60732b0ca51589be1748b86a0d.jpg"
label_path = r"C:\Users\cbnits-304\OneDrive - NEXTZEN MINDS\Desktop\CREDIT_CARD_OCR\train\labels\979ad884-e163-4845-a779-2dee13d257b2_jpeg.rf.ebed0d60732b0ca51589be1748b86a0d.txt"
yaml_path = r"C:\Users\cbnits-304\OneDrive - NEXTZEN MINDS\Desktop\CREDIT_CARD_OCR\data.yaml"

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)
label_names = data['names']
print("Labels:", label_names)

img = cv.imread(img_path)
H, W, _ = img.shape

with open(label_path, 'r') as file:
    lines = file.readlines()
annotations = [tuple(line.split()) for line in lines]

print("Annotations:", annotations)
print()
for label, x, y, w, h in annotations:
    label_id = int(label)
    label_name = label_names[label_id]

    x = float(x); y = float(y); w = float(w); h = float(h)
    x1 = int((x - w/2) * W)
    x2 = int((x + w/2) * W)
    y1 = int((y - h/2) * H)
    y2 = int((y + h/2) * H)

    cv.rectangle(img, (x1, y1), (x2, y2), (0, 100, 255), 1)
    cv.putText(img, label_name, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        print(f"{label_name}: Empty ROI")
        continue

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blur = cv.bilateralFilter(gray, 11, 17, 17)
    morph = cv.morphologyEx(blur, cv.MORPH_OPEN, np.ones((2,2), np.uint8))
    norm = cv.normalize(morph, None, 0, 255, cv.NORM_MINMAX)

    if label_name == 'Exp date':
        processed = enhance_expiry_image(gray)
    else:
        processed = cv.bilateralFilter(gray, 11, 17, 17)
    resized = cv.resize(processed, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    ocr_results = reader.readtext(resized, detail=1)
    text_raw = ocr_results[0][1] if ocr_results else ''
    text_clean = clean_text(label_name, text_raw, ocr_results)
    print(f"{label_name}: {text_clean}")

    cv.imshow(label_name, resized)

cv.imshow("Annotated Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
