from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(req: TranslateRequest):
    if len(req.text) > 500:
        return {"error": "Text too long"}

    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    return {
        "translated_text": result
    }
