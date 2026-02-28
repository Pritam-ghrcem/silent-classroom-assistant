import mysql.connector
import whisper
import re
from collections import Counter
from transformers import pipeline
from spellchecker import SpellChecker
import urllib.parse
# ---------------- DATABASE CONNECTION ----------------
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Pritam123",   # 👈 put your MySQL password
        database="silent_classroom_assistant"
    )

# ---------------- INSERT DATA ----------------
def insert_data(audio_file, transcript, summary, notes, keywords):
    try:
        conn = connect_db()
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
        conn.close()

# ---------------- STEP 1: SPEECH TO TEXT ----------------
def speech_to_text(audio_file):
    print("Transcribing audio...\n")
    model = whisper.load_model("base")
    return model.transcribe(audio_file)["text"]

# ---------------- STEP 2: CLEAN SPOKEN NOISE ----------------
def clean_transcript(text):
    fillers = [
        "you know", "uh", "um", "actually", "basically",
        "kind of", "sort of", "okay", "so", "right"
    ]

    text = text.lower()

    for f in fillers:
        text = text.replace(f, "")

    # remove repeated words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ---------------- STEP 3: SPLIT LONG TEXT ----------------
def split_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# ---------------- STEP 4: AI SUMMARIZATION ----------------
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []

    chunks = split_text(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        result = summarizer(
            chunk,
            max_length=120,
            min_length=50,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)

# ---------------- STEP 5: SPELL CORRECTION ----------------
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

# ---------------- STEP 6: CONVERT TO BULLETS ----------------
def convert_to_bullets(text):
    bullets = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for s in sentences:
        s = s.strip().capitalize()
        if len(s.split()) < 8:
            continue
        bullets.append("- " + s)

    return bullets

# ---------------- STEP 7: EXTRACT KEY CONCEPTS ----------------
def extract_concepts(bullets, top_n=6):
    text = " ".join(bullets)

    phrases = re.findall(
        r'\b[a-zA-Z]{4,}\s[a-zA-Z]{4,}(?:\s[a-zA-Z]{4,})?\b',
        text.lower()
    )

    freq = Counter(phrases)
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))

    return [p for p, _ in ranked[:top_n]]

# ---------------- STEP 8: HIGHLIGHT CONCEPTS ----------------
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

# ---------------- STEP 9: YOUTUBE SEARCH LINKS ----------------
def generate_youtube_links(concepts):
    links = []
    for c in concepts:
        query = urllib.parse.quote_plus(c + " tutorial")
        url = f"https://www.youtube.com/results?search_query={query}"
        links.append((c.title(), url))
    return links

# ---------------- STEP 10: GENERATE HTML (PRINT / PDF) ----------------
def save_as_html(notes, youtube_links, filename="lecture_notes.html"):
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Silent Classroom Assistant</title>
<style>
    body {
        font-family: "Segoe UI", Arial, sans-serif;
        background-color: #f4f6f8;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 900px;
        margin: 40px auto;
        background: #ffffff;
        padding: 30px 40px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    h1 {
        text-align: center;
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 6px;
        margin-top: 30px;
    }
    ul {
        line-height: 1.8;
        margin-top: 20px;
    }
    li {
        margin-bottom: 14px;
        font-size: 16px;
    }
    strong {
        text-decoration: underline;
    }
    a {
        color: #1a73e8;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    .print-btn {
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 10px 18px;
        border-radius: 6px;
        cursor: pointer;
        margin: 15px 0;
    }
    .print-btn:hover {
        background-color: #1a252f;
    }
    @media print {
        .print-btn {
            display: none;
        }
    }
    .note {
        margin-top: 30px;
        padding: 12px;
        background-color: #fff8e1;
        border-left: 4px solid #f1c40f;
        font-size: 14px;
        color: #555;
    }
    footer {
        margin-top: 40px;
        text-align: center;
        font-size: 14px;
        color: #777;
    }
</style>
</head>
<body>

<div class="container">
<h1>Silent Classroom Assistant</h1>

<button class="print-btn" onclick="window.print()">Print / Save as PDF</button>

<h2>Structured Lecture Notes</h2>
<ul>
"""

    for n in notes:
        n = re.sub(r'__(.*?)__', r'<strong>\1</strong>', n)
        n = n.lstrip("- ").strip()
        html += f"<li>{n}</li>\n"

    html += """
</ul>

<h2>Recommended YouTube Resources (Optional)</h2>
<ul>
"""

    for title, link in youtube_links:
        html += f'<li><a href="{link}" target="_blank">{title} – YouTube Search</a></li>\n'

    html += """
</ul>

<div class="note">
Note: Notes and links are generated automatically from lecture audio.
Minor spelling or grammatical errors may occur due to speech recognition.
</div>

<footer>
Generated by Silent Classroom Assistant (AI-based Note Generation System)
</footer>
</div>
</body>
</html>
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

# ---------------- STEP 11: MAIN PIPELINE ----------------
def silent_classroom_assistant():
    print("Silent Classroom Assistant running...\n")

    audio_file = "Recording (4).wav"

    raw_text = speech_to_text(audio_file)
    cleaned_text = clean_transcript(raw_text)

    print("Generating structured notes...\n")

    summary = summarize_text(cleaned_text)
    summary = correct_text(summary)

    bullets = convert_to_bullets(summary)

    concepts = extract_concepts(bullets)
    notes = underline_keywords(bullets, concepts)
    youtube_links = generate_youtube_links(concepts)

    # 🔥 SAVE TO DATABASE
    insert_data(audio_file, raw_text, summary, notes, concepts)

    print("----- Structured Lecture Notes -----\n")
    for n in notes:
        print(n)

    save_as_html(notes, youtube_links)

    print("\nHTML file generated successfully")

# ---------------- RUN ----------------
silent_classroom_assistant()
