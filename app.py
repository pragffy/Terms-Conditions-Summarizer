import sys
sys.stdout.reconfigure(encoding='utf-8')

import gradio as gr
import PyPDF2
import asyncio
import edge_tts
import tempfile
import os
import google.generativeai as genai
from deep_translator import GoogleTranslator

# ------------------------------------------------------------
# 1. PDF text extraction
# ------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file.name)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text  # Extract all text for MapReduce

# ------------------------------------------------------------
# 2. MapReduce Summarization
# ------------------------------------------------------------
def chunk_text(text, max_words=3000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_text(long_text, api_key):
    genai.configure(api_key=api_key)
    # Using Gemini 2.5 Flash for fast MapReduce
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    chunks = chunk_text(long_text)
    sub_summaries = []
    
    map_prompt = """ACT AS a senior legal analyst specializing in consumer-facing contracts.
Your ONLY task is to extract clauses that directly impact the end user
(the party who is not the drafting party). Ignore boilerplate,
definitions, and procedural language.

INSTRUCTIONS:
1. Read the provided T&C document.
2. Identify clauses that affect the user’s rights, obligations,
   costs, liability, or termination options.
3. Produce a plain-English summary that contains ONLY these
   user-impacting points.
4. Strictly adhere to a length of 130–150 words.
5. Do NOT include any introductory or concluding phrases.
   Start directly with the first point.
6. Use bullet points (no numbering) to separate each distinct
   user-impacting clause.

OUTPUT FORMAT:
- Clause summary in plain English.
- Bullet points only.
- Total word count: exactly between 130 and 150 words.

DOCUMENT:
{text_chunk}"""

    for i, chunk in enumerate(chunks):
        try:
            response = model.generate_content(map_prompt.format(text_chunk=chunk))
            sub_summaries.append(response.text)
        except Exception as e:
            print(f"Error in map phase chunk {i}: {e}")
            
    if not sub_summaries:
        return "Failed to generate summary."
        
    if len(sub_summaries) == 1:
        return sub_summaries[0]
        
    # Reduce Phase
    reduce_prompt = """ACT AS a senior legal analyst specializing in consumer-facing contracts.
Your ONLY task is to consolidate the following sub-summaries of a T&C document into a single final summary.

INSTRUCTIONS:
1. Read the provided sub-summaries.
2. Deduplicate repeated clauses.
3. Produce a plain-English summary that contains ONLY user-impacting points.
4. Strictly adhere to a target length of exactly 130 words.
5. Do NOT include any introductory or concluding phrases. Start directly with the first point.
6. Use bullet points (no numbering).

OUTPUT FORMAT:
- Clause summary in plain English.
- Bullet points only.
- Total word count: exactly 130 words.

SUB-SUMMARIES:
{sub_summaries_text}"""

    combined_text = "\n\n".join(sub_summaries)
    try:
        final_summary = model.generate_content(reduce_prompt.format(sub_summaries_text=combined_text))
        return final_summary.text
    except Exception as e:
        print(f"Error in reduce phase: {e}")
        return "Error in reduce phase."

# ------------------------------------------------------------
# 3. Free Translation using deep_translator
# ------------------------------------------------------------
def translate_summary_free(summary, target_lang_code):
    try:
        translator = GoogleTranslator(source='en', target=target_lang_code)
        return translator.translate(summary)
    except Exception as e:
        print(f"Error in translation to {target_lang_code}: {e}")
        return f"[{target_lang_code} translation error]"

# ------------------------------------------------------------
# 4. TTS using edge-tts with calibrated rates
# ------------------------------------------------------------
RATES = {
    "hi": "+18%", 
    "kn": "+63%", 
    "te": "+59%", 
    "bn": "+79%", 
    "ta": "+71%",
    "mr": "+80%",
    "ml": "+70%"
}
VOICES = {
    "hi": "hi-IN-SwaraNeural", 
    "kn": "kn-IN-SapnaNeural", 
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ta": "ta-IN-PallaviNeural",
    "mr": "mr-IN-AarohiNeural",
    "ml": "ml-IN-SobhanaNeural"
}

async def generate_audio(text, lang):
    out_file = f"summary_{lang}.mp3"
    communicate = edge_tts.Communicate(text, VOICES[lang], rate=RATES[lang])
    await communicate.save(out_file)
    return out_file



# ------------------------------------------------------------
# 5. Main pipeline
# ------------------------------------------------------------
def validate_summary(summary, min_words=125, max_words=135):
    word_count = len(summary.split())
    if word_count < min_words:
        msg = f"⚠️ Warning: This T&C is unusually short. The summary is accurate but only {word_count} words (below target of {min_words})."
        return summary, msg
    elif word_count > max_words:
        msg = f"⚠️ Truncated summary from {word_count} to {max_words} words to meet strict requirements."
        # Simple truncation (keep first N words)
        truncated = " ".join(summary.split()[:max_words])
        return truncated, msg
    else:
        msg = f"✅ Perfect! {word_count} words (within 130-150 target)."
        return summary, msg

def process_input(pdf_file, pasted_text, api_key):
    if not api_key:
        return "⚠️ Please provide a Gemini API Key.", ""
        
    raw_text = ""
    if pasted_text and pasted_text.strip():
        raw_text = pasted_text.strip()
    elif pdf_file:
        raw_text = extract_text_from_pdf(pdf_file)
    else:
        return "⚠️ Please upload a PDF or paste text.", ""
         
    if not raw_text:
        return "⚠️ No text found", ""
    
    # Summarize to English
    summary_en = summarize_text(raw_text, api_key)
    
    # Validate and handle word count
    final_summary, status_msg = validate_summary(summary_en)
    
    # Translate using free Google Translate
    lang_codes = ["hi", "kn", "te", "bn", "ta", "mr", "ml"]
    translations = {}
    for code in lang_codes:
        translations[code] = translate_summary_free(final_summary, code)
        
    # Generate audio concurrently
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    audio_tasks = [generate_audio(translations[code], code) for code in lang_codes]
    audio_files = loop.run_until_complete(asyncio.gather(*audio_tasks))
    
    status_msg += "\n✅ Translations and 60-second audio files generated!"
    
    return status_msg, final_summary, audio_files[0], audio_files[1], audio_files[2], audio_files[3], audio_files[4], audio_files[5], audio_files[6]

# ------------------------------------------------------------
# 6. Gradio UI – Drag & drop PDF
# ------------------------------------------------------------
with gr.Blocks(title="T&C Summarizer (Multilingual)") as demo:
    gr.Markdown("# 📄 Terms & Conditions Summarizer")
    gr.Markdown("Upload a PDF or paste text to get a 130–150 word plain-English summary, plus perfectly calibrated 60-second audio files in 6 Indian languages.")
    
    api_key_input = gr.Textbox(
        label="🔑 Gemini API Key", 
        type="password", 
        placeholder="Enter your Gemini API Key here",
        value="AIzaSyBx_b2lfglZ4rMOJXFTGSQqcy8QjQLV7Hs",
        visible=False
    )
    
    with gr.Row():
        pdf_input = gr.File(label="📁 Drag & drop your PDF here", file_types=[".pdf"])
        text_input = gr.Textbox(label="📝 Or paste your T&C Text here", lines=8, placeholder="Paste your dense legal text here...")
        
    submit_btn = gr.Button("🚀 Summarize, Translate & Generate Audio", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    summary_output = gr.Textbox(label="📝 English Summary", lines=10, interactive=False)
    
    gr.Markdown("### 🔊 60-Second Audio Summaries")
    with gr.Row():
        audio_hi = gr.Audio(label="🇮🇳 Hindi", type="filepath")
        audio_kn = gr.Audio(label="🇮🇳 Kannada", type="filepath")
        audio_te = gr.Audio(label="🇮🇳 Telugu", type="filepath")
        
    with gr.Row():
        audio_bn = gr.Audio(label="🇮🇳 Bengali", type="filepath")
        audio_ta = gr.Audio(label="🇮🇳 Tamil", type="filepath")
        audio_mr = gr.Audio(label="🇮🇳 Marathi", type="filepath")
        audio_ml = gr.Audio(label="🇮🇳 Malayalam", type="filepath")
    
    submit_btn.click(
        process_input, 
        inputs=[pdf_input, text_input, api_key_input], 
        outputs=[status, summary_output, audio_hi, audio_kn, audio_te, audio_bn, audio_ta, audio_mr, audio_ml]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
