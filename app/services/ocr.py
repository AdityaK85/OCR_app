from pathlib import Path
from functools import lru_cache
import time
import cv2
import numpy as np
import fitz
from rapidocr import RapidOCR

def resize_for_ocr(img,max_width=1600,max_height=2200):
    height, width = img.shape[:2]
    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale < 1.0:
        width = int(width * scale)
        height = int(height * scale)
        img = cv2.resize(img,(width, height),interpolation=cv2.INTER_AREA)
    return img


def preprocess_image( img, profile="fast"):
    img = resize_for_ocr(img)
    if profile in ("none", "fast"):
        return img

    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    if profile in ("balanced","high_accuracy"):
        gray = cv2.fastNlMeansDenoising(gray,None,7,7,21)
        gray = cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)

    if profile == "high_accuracy":
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)

    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)



@lru_cache(maxsize=4)
def get_ocr(language="en"):
    ocr = RapidOCR()
    print("RapidOCR initialized.")
    return ocr


def parse_rapidocr(result):
    output = []
    if result is None:
        return output

    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)

    if boxes is None:
        boxes = []

    if texts is None:
        texts = []

    if scores is None:
        scores = []

    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()

    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    boxes = list(boxes)
    texts = list(texts)
    scores = list(scores)

    for index, box in enumerate(boxes):

        text = (texts[index] if index < len(texts) else "" )
        if text is None:
            continue

        text = str(text).strip()
        if not text:
            continue

        confidence = None
        if index < len(scores):
            try:
                confidence = float(scores[index])
            except ( TypeError, ValueError ):
                confidence = None

        if hasattr(box, "tolist"):
            box = box.tolist()

        output.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": box,
            }
        )

    return output

def image_ocr_array(img,language="en",preprocess="fast"):
    if img is None:
        raise ValueError("Invalid or corrupted image")

    start = time.perf_counter()
    img = preprocess_image(img,preprocess,)
    height, width = img.shape[:2]
    ocr = get_ocr(language)
    result = ocr(img)
    items = parse_rapidocr(result)
    text = "\n".join( item["text"] for item in items )
    elapsed = ( time.perf_counter() - start)

    return {
        "page": 1,
        "width": width,
        "height": height,
        "processing_time": round( elapsed, 3 ),
        "results": items,
        "text": text,
    }

def image_ocr( path, language="en", preprocess="fast"):
    path = Path(path)
    img = cv2.imread( str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid or corrupted image")

    result = image_ocr_array( img, language=language, preprocess=preprocess )
    return {
        "pages": [
            result
        ],
        "text": result["text"],
        "processing_time": result[
            "processing_time"
        ],
    }


def pdf_ocr( path, language="en", preprocess="fast"):
    
    start = time.perf_counter()
    doc = fitz.open(str(path))
    pages = []
    alltext = []
    try:
        for page_number, page in enumerate( doc,  start=1 ):
            matrix = fitz.Matrix(
                150 / 72,
                150 / 72,
            )

            pix = page.get_pixmap( matrix=matrix, alpha=False)
            arr = np.frombuffer( pix.samples, dtype=np.uint8)
            channels = pix.n
            arr = arr.reshape( pix.height, pix.width, channels )

            if channels == 3:
                img = cv2.cvtColor( arr,cv2.COLOR_RGB2BGR,)
            elif channels == 4:
                img = cv2.cvtColor(arr,cv2.COLOR_RGBA2BGR)
            else:
                img = arr

            result = image_ocr_array( img, language=language, preprocess=preprocess, )
            result["page"] = page_number
            pages.append(result)
            alltext.append( result["text"] )

    finally:
        doc.close()

    elapsed = (time.perf_counter() - start)

    return {
        "pages": pages,
        "text": "\n".join(alltext),
        "page_count": len(pages),
        "processing_time": round( elapsed, 3,),
    }


def process_file( path, language="en", preprocess="fast"):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_ocr(path,language=language,preprocess=preprocess,)
    return image_ocr(path,language=language,preprocess=preprocess,)