# ===================== IMPORTS =====================
import whisper
import re
import time
import urllib.parse
from collections import Counter

import numpy as np
import sounddevice as sd
import soundfile as sf

from transformers import pipeline
from spellchecker import SpellChecker


# ===================== STEP 1: RECORD SYSTEM AUDIO =====================
def record_system_audio(duration=60, filename="system_audio.wav"):
    print("\nGet ready...")
    print("▶ Start your YouTube lecture NOW")
    time.sleep(3)

    samplerate = 48000
    channels = 2

    print("🎙 Recording system audio for", duration, "seconds...")

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype="float32"
    )

    sd.wait()
    sf.write(filename, recording, samplerate)

    print("✅ Audio saved as:", filename)


# ===================== STEP 2: SPEECH TO TEXT =====================
def speech_to_text(audio_file):
    print("\n🧠 Transcribing audio...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]


# ===================== STEP 3: CLEAN TRANSCRIPT =====================
def clean_transcript(text):
    fillers = [
        "you know", "uh", "um", "actually",
        "basically", "kind of", "sort of",
        "okay", "so", "right"
    ]

    text = text.lower()
    for f in fillers:
        text = text.replace(f, "")

    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ===================== STEP 4: SPLIT TEXT =====================
def split_text(text, max_words=300):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


# ===================== STEP 5: SUMMARIZATION =====================
def summarize_text(text):
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    summaries = []
    chunks = split_text(text)

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=120,
            min_length=50,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)


# ===================== STEP 6: SPELL CORRECTION =====================
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


# ===================== STEP 7: BULLET POINTS =====================
def convert_to_bullets(text):
    bullets = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for s in sentences:
        s = s.strip().capitalize()
        if len(s.split()) >= 8:
            bullets.append("- " + s)

    return bullets


# ===================== STEP 8: KEY CONCEPTS =====================
def extract_concepts(bullets, top_n=6):
    text = " ".join(bullets)

    phrases = re.findall(
        r'\b[a-zA-Z]{4,}\s[a-zA-Z]{4,}(?:\s[a-zA-Z]{4,})?\b',
        text.lower()
    )

    freq = Counter(phrases)
    ranked = sorted(
        freq.items(),
        key=lambda x: (-x[1], -len(x[0]))
    )

    return [p for p, _ in ranked[:top_n]]


# ===================== STEP 9: HIGHLIGHT CONCEPTS =====================
def underline_keywords(bullets, concepts):
    output = []
    for line in bullets:
        for c in concepts:
            line = re.sub(
                rf'\b{c}\b',
                f"__{c.upper()}__",
                line,
                flags=re.IGNORECASE
            )
        output.append(line)
    return output


# ===================== STEP 10: YOUTUBE LINKS =====================
def generate_youtube_links(concepts):
    links = []
    for c in concepts:
        query = urllib.parse.quote_plus(c + " tutorial")
        url = f"https://www.youtube.com/results?search_query={query}"
        links.append((c.title(), url))
    return links


# ===================== STEP 11: SAVE HTML =====================
def save_as_html(notes, youtube_links):
    html = """
<!DOCTYPE html>
<html>
<head>
<title>Silent Classroom Assistant</title>
<style>
body { font-family: Arial; background: #f4f6f8; }
.container { max-width: 900px; margin: auto; background: white; padding: 30px; }
h1 { text-align: center; }
li { margin-bottom: 12px; }
strong { text-decoration: underline; }
</style>
</head>
<body>
<div class="container">
<h1>Silent Classroom Assistant</h1>
<ul>
"""

    for n in notes:
        n = re.sub(r'__(.*?)__', r'<strong>\1</strong>', n)
        html += f"<li>{n[2:]}</li>\n"

    html += "</ul><h2>YouTube Resources</h2><ul>"

    for title, link in youtube_links:
        html += f'<li><a href="{link}" target="_blank">{title}</a></li>'

    html += """
</ul>
<p><i>Generated automatically from system audio</i></p>
</div>
</body>
</html>
"""

    with open("lecture_notes.html", "w", encoding="utf-8") as f:
        f.write(html)


# ===================== MAIN PIPELINE =====================
def silent_classroom_assistant():
    print("\n--- SILENT CLASSROOM ASSISTANT STARTED ---")

    record_system_audio(duration=60)

    raw_text = speech_to_text("system_audio.wav")
    cleaned = clean_transcript(raw_text)

    summary = summarize_text(cleaned)
    summary = correct_text(summary)

    bullets = convert_to_bullets(summary)
    concepts = extract_concepts(bullets)

    notes = underline_keywords(bullets, concepts)
    youtube_links = generate_youtube_links(concepts)

    print("\n--- GENERATED NOTES ---\n")
    for n in notes:
        print(n)

    save_as_html(notes, youtube_links)

    print("\n✅ Output saved as lecture_notes.html")


# ===================== RUN =====================
if __name__ == "__main__":
    silent_classroom_assistant()
