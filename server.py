from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

# التأكد من استخدام الـ GPU إذا كان متاحاً (device=0 للـ CUDA)
device = 0 if torch.cuda.is_available() else -1
print(f"Running on: {'GPU' if device == 0 else 'CPU'}")

# تحميل النموذج محلياً
classifier = pipeline('text-classification', 
                      model='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment', 
                      device=device)

class TextData(BaseModel):
    sentences: list[str]

@app.get("/")
def home():
    return {"message": "Server is running! Send data to /predict"}    

@app.post("/predict")
async def predict(data: TextData):
    try:
        results = classifier(data.sentences)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # تشغيل السيرفر على منفذ 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
