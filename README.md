# 💳 Credit Card Detail Reader

A real-time computer vision pipeline that extracts credit card details (number, name, expiry) using YOLOv8 and OCR. Designed for fintech-grade UX with robust preprocessing and postprocessing.

## Features

- YOLOv8-based detection of card fields
- Field-specific OCR using EasyOCR or pytesseract
- Adaptive thresholding, contour filtering, and masking logic
- Character mapping for common OCR errors (e.g., 'O' → '0')
- Real-time UX: guide rectangle, green overlay, one-time extraction

## Important

Dataset needs to be installed from Roboflow, Information about the Dataset can be inferred from the README files of ROBOFLOW, Refer results folder for the confirmation of the dataset 

## Installation

```bash
git clone https://github.com/TanAgrawal/credit-card-reader.git
cd credit-card-reader
pip install -r requirements.txt

## Usage
bash

python main.py
