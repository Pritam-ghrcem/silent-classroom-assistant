import mysql.connector
import whisper
import re
from collections import Counter
from transformers import pipeline
from spellchecker import SpellChecker
import urllib.parse
import os

# ---------------- LOAD MODELS ----------------
print("Loading models... Please wait")

model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("Models loaded successfully\n")

# ---------------- DATABASE CONNECTION ----------------
def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="Pritam123",
            database="silent_classroom_assistant"
        )
    except Exception as e:
        print("❌ Database connection error:", e)
        return None

# ---------------- INSERT DATA ----------------
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

# ---------------- SPEECH TO TEXT ----------------
def speech_to_text(audio_file):
    print("🎤 Transcribing audio...\n")

    if not os.path.exists(audio_file):
        raise Exception("Audio file not found")

    result = model.transcribe(audio_file)
    return result["text"]

# ---------------- CLEAN TEXT ----------------
def clean_transcript(text):
    fillers = ["you know", "uh", "um", "actually", "basically",
               "kind of", "sort of", "okay", "so", "right"]

    text = text.lower()

    for f in fillers:
        text = text.replace(f, "")

    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ---------------- SPLIT TEXT ----------------
def split_text(text, max_words=250):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# ---------------- SUMMARIZE ----------------
def summarize_text(text):
    summaries = []
    chunks = split_text(text)

    for i, chunk in enumerate(chunks, 1):
        print(f"📄 Summarizing chunk {i}/{len(chunks)}...")

        result = summarizer(
            chunk,
            max_length=80,
            min_length=30,
            do_sample=False
        )

        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)

# ---------------- SPELL CORRECTION ----------------
def correct_text(text):
    spell = SpellChecker()
    corrected = []

    for word in text.split():
        pure = re.sub(r'[^a-zA-Z]', '', word)

        if pure and pure.lower() not in spell:
            fix = spell.correction(pure)
            if fix:
                word = word.replace(pure, fix)

        corrected.append(word)

    return " ".join(corrected)

# ---------------- BULLETS ----------------
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

# ---------------- KEYWORDS (IMPORTANT FIX) ----------------
def extract_concepts(bullets, top_n=6):

    text = " ".join(bullets).lower()

    # Extract meaningful phrases (2-3 words)
    phrases = re.findall(
        r'\b[a-zA-Z]{4,}\s[a-zA-Z]{4,}(?:\s[a-zA-Z]{4,})?\b',
        text
    )

    freq = Counter(phrases)

    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))

    return [p for p, _ in ranked[:top_n]]

# ---------------- HIGHLIGHT ----------------
def underline_keywords(bullets, concepts):
    formatted = []

    for line in bullets:
        for c in concepts:
            line = re.sub(
                rf'\b{c}\b',
                f"__{c.upper()}__",
                line,
                flags=re.IGNORECASE
            )
        formatted.append(line)

    return formatted

# ---------------- YOUTUBE LINKS ----------------
def generate_youtube_links(concepts):
    links = []

    for c in concepts:
        query = urllib.parse.quote_plus(c + " explained tutorial")
        url = f"https://www.youtube.com/results?search_query={query}"

        links.append({
            "topic": c.title(),
            "link": url
        })

    return links

# ---------------- SAVE HTML ----------------
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

# ---------------- MAIN FUNCTION ----------------
def silent_classroom_assistant(audio_file):

    print("🚀 Silent Classroom Assistant running...\n")

    if not audio_file or not os.path.exists(audio_file):
        raise Exception("Invalid audio file")

    # 1. Speech to Text
    raw_text = speech_to_text(audio_file)

    if not raw_text.strip():
        raise Exception("No speech detected")

    # 2. Clean
    cleaned_text = clean_transcript(raw_text)

    # 3. Summarize
    summary = summarize_text(cleaned_text)

    # 4. Correct
    summary = correct_text(summary)

    # 5. Bullets
    bullets = convert_to_bullets(summary)

    # 6. Keywords
    concepts = extract_concepts(bullets)

    # 7. Highlight
    notes = underline_keywords(bullets, concepts)

    # 8. YouTube Links
    youtube_links = generate_youtube_links(concepts)

    # 9. Save DB
    insert_data(audio_file, raw_text, summary, notes, concepts)

    # 10. Save HTML
    save_as_html(notes, youtube_links)

    print("✅ Process completed\n")

    return {
        "notes": notes,
        "keywords": concepts,
        "youtube_links": youtube_links,
        "summary": summary
    }