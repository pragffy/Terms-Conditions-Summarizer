import os
import asyncio
import PyPDF2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from deep_translator import GoogleTranslator
import edge_tts

app = FastAPI(title="T&C Summarizer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio directory exists
os.makedirs("audio", exist_ok=True)

# ------------------------------------------------------------
# Backend Logic
# ------------------------------------------------------------
def extract_text_from_pdf(pdf_file_obj):
    reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, max_words=3000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_text(long_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    chunks = chunk_text(long_text)
    sub_summaries = []
    
    map_prompt = """ACT AS a senior legal analyst specializing in consumer-facing contracts.
Your ONLY task is to extract clauses that directly impact the end user.
INSTRUCTIONS:
1. Read the provided T&C document.
2. Identify user-impacting clauses.
3. Produce a plain-English summary.
4. Strictly adhere to exactly 130 words.
5. No introductory phrases.
6. Use bullet points.
OUTPUT FORMAT: Bullet points only.
DOCUMENT:
{text_chunk}"""

    for i, chunk in enumerate(chunks):
        try:
            response = model.generate_content(map_prompt.format(text_chunk=chunk))
            sub_summaries.append(response.text)
        except Exception as e:
            print(f"Error in map phase chunk {i}: {e}")
            
    if not sub_summaries:
        raise ValueError("Failed to generate summary.")
        
    if len(sub_summaries) == 1:
        return sub_summaries[0]
        
    reduce_prompt = """ACT AS a senior legal analyst. Consolidate these sub-summaries.
INSTRUCTIONS:
1. Deduplicate repeated clauses.
2. Produce a plain-English summary of ONLY user-impacting points.
3. Strictly adhere to a target length of exactly 130 words.
4. No introductory phrases.
5. Use bullet points.
OUTPUT FORMAT: Bullet points only.
SUB-SUMMARIES:
{sub_summaries_text}"""

    combined_text = "\n\n".join(sub_summaries)
    final_summary = model.generate_content(reduce_prompt.format(sub_summaries_text=combined_text))
    return final_summary.text

def translate_summary_free(summary, target_lang_code):
    try:
        translator = GoogleTranslator(source='en', target=target_lang_code)
        return translator.translate(summary)
    except Exception as e:
        print(f"Error in translation to {target_lang_code}: {e}")
        return f"[{target_lang_code} translation error]"

RATES = {
    "hi": "+18%", "kn": "+63%", "te": "+59%", 
    "bn": "+79%", "ta": "+71%", "mr": "+80%", "ml": "+70%"
}
VOICES = {
    "hi": "hi-IN-SwaraNeural", "kn": "kn-IN-SapnaNeural", "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural", "ta": "ta-IN-PallaviNeural", "mr": "mr-IN-AarohiNeural", "ml": "ml-IN-SobhanaNeural"
}

async def generate_audio(text, lang):
    out_file = f"audio/summary_{lang}.mp3"
    communicate = edge_tts.Communicate(text, VOICES[lang], rate=RATES[lang])
    await communicate.save(out_file)
    return f"/audio/summary_{lang}.mp3"

def validate_summary(summary, min_words=125, max_words=135):
    word_count = len(summary.split())
    if word_count < min_words:
        msg = f"⚠️ Warning: This T&C is unusually short. The summary is accurate but only {word_count} words."
        return summary, msg
    elif word_count > max_words:
        msg = f"⚠️ Truncated summary from {word_count} to {max_words} words."
        truncated = " ".join(summary.split()[:max_words])
        return truncated, msg
    else:
        msg = f"✅ Perfect! {word_count} words (within 130 target)."
        return summary, msg

# ------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------
@app.post("/api/summarize")
async def api_summarize(
    api_key: str = Form(...),
    text: str = Form(None),
    file: UploadFile = File(None)
):
    if not api_key:
        raise HTTPException(status_code=400, detail="API Key is required")
        
    raw_text = ""
    if text and text.strip():
        raw_text = text.strip()
    elif file:
        raw_text = extract_text_from_pdf(file.file)
    else:
        raise HTTPException(status_code=400, detail="Please upload a PDF or paste text.")
        
    if not raw_text:
        raise HTTPException(status_code=400, detail="No text found in input.")
        
    try:
        # Summarize to English
        summary_en = summarize_text(raw_text, api_key)
        final_summary, status_msg = validate_summary(summary_en)
        
        # Translate
        lang_codes = ["hi", "kn", "te", "bn", "ta", "mr", "ml"]
        translations = {code: translate_summary_free(final_summary, code) for code in lang_codes}
        
        # Generate Audio sequentially to avoid Microsoft edge-tts throttling
        audio_paths = []
        for code in lang_codes:
            path = await generate_audio(translations[code], code)
            audio_paths.append(path)
        
        audio_urls = {code: path for code, path in zip(lang_codes, audio_paths)}
        
        return {
            "status": status_msg + "\n✅ Translations and audio generated!",
            "english_summary": final_summary,
            "audio_urls": audio_urls
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount directories
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

# Mount frontend at the very end so it doesn't override API routes
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
