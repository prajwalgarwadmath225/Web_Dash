import os
import fitz
import cv2
from paddleocr import PaddleOCR
from sam_utils import process_image_for_crops

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def pdf_page_to_image(pdf_path, page_number=0, dpi=300, output_folder='output'):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}_{dpi}dpi.png")
        pix.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error converting PDF page to image: {e}")
        return None

def save_ocr_results(image, ocr_results, output_path):
    for line in ocr_results[0]:
        box, (text, _) = line
        x_min = int(min([pt[0] for pt in box]))
        y_min = int(min([pt[1] for pt in box]))
        x_max = int(max([pt[0] for pt in box]))
        y_max = int(max([pt[1] for pt in box]))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)

def process_pdf(pdf_path, output_folder='output'):
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        print(f"Total pages in PDF: {total_pages}")
        for page_num in range(total_pages):
            image_path = pdf_page_to_image(pdf_path, page_number=page_num, output_folder=output_folder)
            if not image_path or not os.path.exists(image_path):
                continue
            process_image_for_crops(image_path, output_folder=output_folder)
    except Exception as e:
        print(f"Error processing PDF: {e}")