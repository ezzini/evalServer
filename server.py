from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import bert_score

app = FastAPI()

# التأكد من استخدام الـ GPU إذا كان متاحاً (device=0 للـ CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# تحميل نموذج التصنيف
classifier = pipeline('text-classification', 
                      model='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment', 
                      device=0 if device == "cuda" else -1)

# تحميل نموذج Perplexity
# ppl_model_id = 'aubmindlab/aragpt2-base'
# ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_id)
# ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_id).to(device)
# ppl_model.eval()

class TextData(BaseModel):
    sentences: list[str]

class BertScoreRequest(BaseModel):
    sentences: list[str]
    references: list[str]

@app.get("/")
def home():
    return {"message": "Server is running! Send data to /predict, /perplexity, or /bertscore"}    

@app.post("/predict")
async def predict(data: TextData):
    try:
        results = classifier(data.sentences)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/perplexity")
# async def calculate_perplexity(data: TextData):
#     try:
#         sentences = data.sentences
#         ppls = []
#         for sentence in sentences:
#             inputs = ppl_tokenizer(sentence, return_tensors="pt").to(device)
#             with torch.no_grad():
#                 outputs = ppl_model(**inputs, labels=inputs["input_ids"])
#                 loss = outputs.loss
#                 ppl = torch.exp(loss)
#                 ppls.append(ppl.item())
#         mean_ppl = np.mean(ppls)
#         return {"mean_perplexity": float(mean_ppl)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/bertscore")
# async def calculate_bertscore(data: BertScoreRequest):
#     try:
#         cands = data.sentences
#         refs = data.references
#         # BERTScore calculation
#         P, R, F1 = bert_score.score(cands, refs, lang='ar', verbose=False, device=device)
#         return {"f1": F1.mean().item()}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # تشغيل السيرفر على منفذ 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
