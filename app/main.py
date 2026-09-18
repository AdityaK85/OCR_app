from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid, shutil, json
from .services.ocr import process_file

BASE=Path(__file__).resolve().parent.parent
UPLOAD=BASE/'storage/uploads'; RESULTS=BASE/'storage/results'
UPLOAD.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
ALLOWED={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.pdf'}
app=FastAPI(title='OCR API',version='1.0.0',description='FastAPI OCR service')

@app.post('/api/v1/ocr')
def ocr(file: UploadFile=File(...), language: str=Query('en'), preprocess: str=Query('balanced')):
    ext=Path(file.filename or '').suffix.lower()
    if ext not in ALLOWED: 
        raise HTTPException(400,f'Unsupported file type: {ext}')
    doc_id=str(uuid.uuid4()); path=UPLOAD/f'{doc_id}{ext}'
    with path.open('wb') as out: 
        shutil.copyfileobj(file.file,out)
    try:
        result=process_file(path, language=language, preprocess=preprocess)
        result.update({'document_id':doc_id,'filename':file.filename,'status':'completed'})
        (RESULTS/f'{doc_id}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500,f'OCR processing failed: {e}')

@app.get('/api/v1/ocr/{document_id}')
def get_result(document_id:str):
    p=RESULTS/f'{document_id}.json'
    if not p.exists(): 
        raise HTTPException(404,'Result not found')
    return json.loads(p.read_text(encoding='utf-8'))
