from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List
from transformers import MarianMTModel, MarianTokenizer
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarianMT Translation API",
    description="Self-hosted translation service using MarianMT models",
    version="1.0.0"
)

# Health check endpoint for Render
@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "service": "MarianMT Translation API",
        "loaded_models": list(models_cache.keys())
    }

# Model mapping for language pairs
MODEL_MAP = {
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-ar": "Helsinki-NLP/opus-mt-en-ar",
    "ar-en": "Helsinki-NLP/opus-mt-ar-en",
    "fr-ar": "Helsinki-NLP/opus-mt-fr-ar",
    "ar-fr": "Helsinki-NLP/opus-mt-ar-fr",
}

# Cache for loaded models
models_cache = {}
tokenizers_cache = {}

def get_model_pair(source_lang: str, target_lang: str):
    """Get or load model for language pair"""
    pair_key = f"{source_lang}-{target_lang}"
    
    if pair_key not in models_cache:
        if pair_key not in MODEL_MAP:
            raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}. Available pairs: {list(MODEL_MAP.keys())}")
        
        model_name = MODEL_MAP[pair_key]
        logger.info(f"🔄 Loading model: {model_name} for pair {pair_key}")
        try:
            tokenizers_cache[pair_key] = MarianTokenizer.from_pretrained(model_name)
            models_cache[pair_key] = MarianMTModel.from_pretrained(model_name)
            logger.info(f"✅ Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise ValueError(f"Failed to load model {model_name} for pair {pair_key}: {str(e)}")
    
    return tokenizers_cache[pair_key], models_cache[pair_key]

class TranslateRequest(BaseModel):
    text: Union[str, List[str]]
    source_lang: str = "auto"
    target_lang: str

@app.post("/translate")
def translate(req: TranslateRequest):
    """Translate text using MarianMT"""
    try:
        # Normalize language codes
        source_lang = normalize_lang(req.source_lang)
        target_lang = normalize_lang(req.target_lang)
        
        # Validate language codes
        valid_langs = {"en", "fr", "ar"}
        if source_lang not in valid_langs:
            raise ValueError(f"Invalid source_lang: {source_lang}. Must be one of: {valid_langs}")
        if target_lang not in valid_langs:
            raise ValueError(f"Invalid target_lang: {target_lang}. Must be one of: {valid_langs}")
        
        # Handle auto-detect (default to English if auto)
        if source_lang == "auto":
            source_lang = "en"  # Default assumption, can be improved with language detection
        
        # If source and target are same, return original
        if source_lang == target_lang:
            if isinstance(req.text, str):
                return {"translated_text": req.text}
            else:
                return {"translated_text": req.text}
        
        # Log the translation request for debugging
        logger.info(f"🔄 Translation request: {source_lang} -> {target_lang}, text: '{req.text[:50] if isinstance(req.text, str) else str(req.text)[:50]}'")
        
        # Get model for language pair
        pair_key = f"{source_lang}-{target_lang}"
        logger.info(f"📦 Using model pair: {pair_key}")
        
        tokenizer, model = get_model_pair(source_lang, target_lang)
        
        # Handle single string or array of strings
        texts = [req.text] if isinstance(req.text, str) else req.text
        
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append(text)
                continue
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Validate result - check if it looks like the wrong language
            if target_lang == "ar" and not contains_arabic(result):
                logger.warning(f"⚠️  Translation to Arabic returned non-Arabic text: '{result[:50]}'")
                # Try to detect what language we got
                if contains_french(result):
                    logger.error(f"❌ CRITICAL: Arabic translation returned French! Source: '{text[:50]}', Result: '{result[:50]}'")
            elif target_lang == "fr" and not contains_french(result) and not contains_english(result):
                logger.warning(f"⚠️  Translation to French may be incorrect: '{result[:50]}'")
            
            results.append(result)
            logger.info(f"✅ Translated ({source_lang}->{target_lang}): '{text[:30]}' -> '{result[:30]}'")
        
        # Return single string or array based on input
        if isinstance(req.text, str):
            return {"translated_text": results[0] if results else ""}
        else:
            return {"translated_text": results}
    
    except ValueError as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def contains_arabic(text: str) -> bool:
    """Check if text contains Arabic characters"""
    for char in text:
        if '\u0600' <= char <= '\u06FF':
            return True
    return False

def contains_french(text: str) -> bool:
    """Check if text contains French characters"""
    french_chars = "éèêëàâçùûüôöîï"
    return any(char in text.lower() for char in french_chars)

def contains_english(text: str) -> bool:
    """Check if text is likely English (basic heuristic)"""
    # Simple check: if it's mostly ASCII and not Arabic/French, assume English
    if contains_arabic(text) or contains_french(text):
        return False
    # If it's mostly ASCII printable, likely English
    ascii_count = sum(1 for c in text if ord(c) < 128 and c.isprintable())
    return ascii_count > len(text) * 0.8

def normalize_lang(lang: str) -> str:
    """Normalize language code to standard format"""
    lang = lang.lower().strip()
    # Map common variations
    lang_map = {
        "english": "en",
        "french": "fr",
        "français": "fr",
        "arabic": "ar",
        "عربي": "ar",
    }
    return lang_map.get(lang, lang[:2] if len(lang) >= 2 else "en")
