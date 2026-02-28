import mysql.connector
import re
from collections import Counter
from spellchecker import SpellChecker
from faster_whisper import WhisperModel
from groq import Groq
import urllib.parse
import os
from dotenv import load_dotenv


from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("API KEY:", GROQ_API_KEY)
print("Loaded:", os.getenv("GROQ_API_KEY"))

# ============================================================
#   CONFIGURATION — Change settings here
# ============================================================



# ✅ Toggle: True = use Groq API | False = use local distilbart (offline)
USE_GROQ = True

# ============================================================
#   LOAD MODELS
# ============================================================
print("Loading models... Please wait")

# faster-whisper for transcription (3-4x faster than original whisper)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Only load distilbart if Groq is disabled (saves memory and time)
summarizer = None
if not USE_GROQ:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("✅ Loaded distilbart (offline mode)")

# Initialize Groq client
groq_client = None
if USE_GROQ:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq API connected")

print("Models loaded successfully\n")


# ============================================================
#   DATABASE CONNECTION
# ============================================================
def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv("DB_PASSWORD"),
            database="silent_classroom_assistant"
        )
    except Exception as e:
        print("❌ Database connection error:", e)
        return None


# ============================================================
#   INSERT DATA INTO DATABASE
# ============================================================
def insert_data(audio_file, transcript, summary, notes, keywords):
    conn = None
    try:
        conn = connect_db()
        if conn is None:
            return

        cursor = conn.cursor()

        query = """
        INSERT INTO lecture_notes (audio_file, transcript, summary, notes, keywords)
        VALUES (%s, %s, %s, %s, %s)
        """

        cursor.execute(query, (
            audio_file,
            transcript,
            summary,
            "\n".join(notes),
            ", ".join(keywords)
        ))

        conn.commit()
        print("✅ Data saved to MySQL")

    except Exception as e:
        print("❌ Database error:", e)

    finally:
        if conn:
            conn.close()


# ============================================================
#   SPEECH TO TEXT  (faster-whisper)
# ============================================================
def speech_to_text(audio_file):
    print("🎤 Transcribing audio...\n")

    if not os.path.exists(audio_file):
        raise Exception("Audio file not found")

    # beam_size=1 makes transcription faster
    segments, _ = whisper_model.transcribe(audio_file, beam_size=1)
    text = " ".join([seg.text for seg in segments])
    return text


# ============================================================
#   CLEAN TRANSCRIPT
# ============================================================
def clean_transcript(text):
    fillers = ["you know", "uh", "um", "actually", "basically",
               "kind of", "sort of", "okay", "so", "right"]

    text = text.lower()

    for f in fillers:
        text = text.replace(f, "")

    # Remove repeated consecutive words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ============================================================
#   GROQ API — Summarize + Notes + Keywords in one call
# ============================================================
def groq_process(transcript):
    print("🤖 Processing with Groq AI (Llama 3)...")

    prompt = f"""
You are a smart lecture note assistant for students.

Given the lecture transcript below, do exactly these 3 tasks:

1. Write a clean summary in 3-5 sentences.
2. Generate 6-8 bullet point study notes (each note must be meaningful and at least 8 words).
3. Extract exactly 6 important keywords or key phrases.

Return your response in this EXACT format and nothing else:

SUMMARY:
<your summary here>

NOTES:
• <note 1>
• <note 2>
• <note 3>
• <note 4>
• <note 5>
• <note 6>

KEYWORDS:
<keyword1>, <keyword2>, <keyword3>, <keyword4>, <keyword5>, <keyword6>

Transcript:
{transcript}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# ============================================================
#   PARSE GROQ RESPONSE
# ============================================================
def parse_groq_output(raw_output):
    summary = ""
    notes = []
    keywords = []

    try:
        # Extract SUMMARY
        if "SUMMARY:" in raw_output:
            summary = raw_output.split("SUMMARY:")[1].split("NOTES:")[0].strip()

        # Extract NOTES
        if "NOTES:" in raw_output:
            notes_raw = raw_output.split("NOTES:")[1].split("KEYWORDS:")[0].strip()
            notes = [
                "• " + line.replace("•", "").replace("-", "").strip()
                for line in notes_raw.split("\n")
                if line.strip() and (line.strip().startswith("•") or line.strip().startswith("-"))
            ]

        # Extract KEYWORDS
        if "KEYWORDS:" in raw_output:
            kw_raw = raw_output.split("KEYWORDS:")[1].strip()
            keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]

    except Exception as e:
        print("⚠️ Error parsing Groq output:", e)

    # Fallback if parsing failed
    if not notes:
        notes = ["• " + raw_output[:200]]
    if not keywords:
        keywords = []

    return summary, notes, keywords


# ============================================================
#   OFFLINE FALLBACK — Split text for distilbart
# ============================================================
def split_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


# ============================================================
#   OFFLINE FALLBACK — Summarize with distilbart
# ============================================================
def summarize_text_offline(text):
    summaries = []
    chunks = split_text(text)

    for i, chunk in enumerate(chunks, 1):
        print(f"📄 Summarizing chunk {i}/{len(chunks)}...")
        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)


# ============================================================
#   SPELL CORRECTION (fixed — skips technical words)
# ============================================================
def correct_text(text):
    spell = SpellChecker()

    # Whitelist — these words will never be wrongly corrected
    spell.word_frequency.load_words([
        "nlp", "ai", "ml", "whisper", "bart", "distilbart", "transcription",
        "tokenization", "summarization", "neural", "dataset", "algorithm",
        "preprocessing", "backend", "frontend", "api", "fastapi", "mysql",
        "html", "pdf", "lecture", "keyword", "extraction", "classification",
        "embedding", "transformer", "encoder", "decoder", "attention",
        "gradient", "epoch", "batch", "inference", "pytorch", "tensorflow",
        "numpy", "pandas", "corpus", "tokenizer", "bigram", "trigram",
        "stopword", "lemmatization", "stemming", "vocab", "softmax", "relu",
        "groq", "llama"
    ])

    corrected = []

    for word in text.split():
        pure = re.sub(r'[^a-zA-Z]', '', word)

        if not pure:
            corrected.append(word)
            continue

        # Skip capitalized words (proper nouns / acronyms like NLP, BART)
        if pure[0].isupper():
            corrected.append(word)
            continue

        # Skip long words — likely technical terms
        if len(pure) > 10:
            corrected.append(word)
            continue

        if pure.lower() not in spell:
            fix = spell.correction(pure)
            if fix and fix != pure and len(fix) > 2:
                word = word.replace(pure, fix)

        corrected.append(word)

    return " ".join(corrected)


# ============================================================
#   CONVERT TO BULLETS (offline mode only)
# ============================================================
def convert_to_bullets(text):
    bullets = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for s in sentences:
        s = s.strip().capitalize()
        if len(s.split()) < 6:
            continue
        bullets.append("• " + s)

    if len(bullets) == 0:
        bullets.append("• " + text)

    return bullets


# ============================================================
#   KEYWORDS (offline mode only)
# ============================================================
def extract_concepts(bullets, top_n=6):
    text = " ".join(bullets).lower()
    phrases = re.findall(
        r'\b[a-zA-Z]{4,}\s[a-zA-Z]{4,}(?:\s[a-zA-Z]{4,})?\b',
        text
    )
    freq = Counter(phrases)
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
    return [p for p, _ in ranked[:top_n]]


# ============================================================
#   HIGHLIGHT KEYWORDS IN NOTES
# ============================================================
def underline_keywords(bullets, concepts):
    formatted = []
    for line in bullets:
        for c in concepts:
            line = re.sub(
                rf'\b{re.escape(c)}\b',
                f"__{c.upper()}__",
                line,
                flags=re.IGNORECASE
            )
        formatted.append(line)
    return formatted


# ============================================================
#   GENERATE YOUTUBE LINKS
# ============================================================
def generate_youtube_links(concepts):
    links = []
    for c in concepts:
        query = urllib.parse.quote_plus(c + " explained tutorial")
        url = f"https://www.youtube.com/results?search_query={query}"
        links.append({"topic": c.title(), "link": url})
    return links


# ============================================================
#   SAVE HTML
# ============================================================
def save_as_html(notes, youtube_links, filename="lecture_notes.html"):
    html = """
    <html>
    <head>
    <title>Lecture Notes</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        h1 { color: #333; }
        ul { line-height: 1.8; }
        a { color: blue; }
    </style>
    </head>
    <body>
    <h1>📘 Lecture Notes</h1>
    <ul>
    """

    for n in notes:
        n = re.sub(r'__(.*?)__', r'<b>\1</b>', n)
        html += f"<li>{n}</li>"

    html += "</ul><h2>📺 YouTube Resources</h2><ul>"

    for item in youtube_links:
        html += f'<li><a href="{item["link"]}">{item["topic"]}</a></li>'

    html += "</ul></body></html>"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================
#   MAIN FUNCTION
# ============================================================
def silent_classroom_assistant(audio_file):

    print("🚀 Silent Classroom Assistant running...\n")

    if not audio_file or not os.path.exists(audio_file):
        raise Exception("Invalid audio file")

    # STEP 1 — Speech to Text
    raw_text = speech_to_text(audio_file)

    if not raw_text.strip():
        raise Exception("No speech detected")

    # STEP 2 — Clean transcript
    cleaned_text = clean_transcript(raw_text)

    # STEP 3 — Summarize + Notes + Keywords
    if USE_GROQ:
        # ✅ GROQ MODE — fast, smart, free
        raw_output = groq_process(cleaned_text)
        summary, bullets, concepts = parse_groq_output(raw_output)
        print(f"✅ Groq returned {len(bullets)} notes and {len(concepts)} keywords")

    else:
        # 🔁 OFFLINE FALLBACK — distilbart
        print("⚙️ Running in offline mode (distilbart)...")
        summary = summarize_text_offline(cleaned_text)
        summary = correct_text(summary)
        bullets = convert_to_bullets(summary)
        concepts = extract_concepts(bullets)

    # STEP 4 — Highlight keywords in notes
    notes = underline_keywords(bullets, concepts)

    # STEP 5 — YouTube Links
    youtube_links = generate_youtube_links(concepts)

    # STEP 6 — Save to Database
    insert_data(audio_file, raw_text, summary, notes, concepts)

    # STEP 7 — Save HTML
    save_as_html(notes, youtube_links)

    print("✅ Process completed\n")

    return {
        "notes": notes,
        "keywords": concepts,
        "youtube_links": youtube_links,
        "summary": summary
    }