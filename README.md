## Install
Python 3.12+ is recommended.

```bash
python -m venv venv
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
<!-- CURL 
Simple image processing OCR API

curl --location 'http://localhost:8000/api/v1/ocr?preprocess=high_accuracy' \
--header 'Content-Type: multipart/form-data; boundary=<calculated when request is sent>' \
--form 'file=@"/C:/Users/HP/Downloads/testinvoice.png"'

 -->