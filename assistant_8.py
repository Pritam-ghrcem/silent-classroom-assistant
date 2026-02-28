import whisper
import re
from collections import Counter
from transformers import pipeline
from spellchecker import SpellChecker
import urllib.parse

# ---------------- SPEECH TO TEXT ----------------
def speech_to_text(audio_file):
    print("Transcribing:", audio_file)
    model = whisper.load_model("base")
    return model.transcribe(audio_file)["text"]

# ---------------- CLEAN TEXT ----------------
def clean_transcript(text):
    fillers = [
        "you know", "uh", "um", "actually", "basically",
        "kind of", "sort of", "okay", "so", "right"
    ]

    text = text.lower()
    for f in fillers:
        text = text.replace(f, "")

    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ---------------- SPLIT TEXT ----------------
def split_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# ---------------- SUMMARIZATION ----------------
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []

    for chunk in split_text(text):
        result = summarizer(
            chunk,
            max_length=160,
            min_length=80,
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

# ---------------- BULLET POINTS ----------------
def convert_to_bullets(text):
    bullets = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for s in sentences:
        s = s.strip().capitalize()
        if len(s.split()) >= 8:
            bullets.append("- " + s)

    return bullets

# ---------------- KEY CONCEPTS ----------------
def extract_concepts(bullets, top_n=6):
    text = " ".join(bullets)

    phrases = re.findall(
        r'\b[a-zA-Z]{4,}\s[a-zA-Z]{4,}(?:\s[a-zA-Z]{4,})?\b',
        text.lower()
    )

    freq = Counter(phrases)
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))

    return [p for p, _ in ranked[:top_n]]

# ---------------- HIGHLIGHT CONCEPTS ----------------
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

# ---------------- UNIVERSAL FLOW DIAGRAM ----------------
def generate_universal_flow_diagram(bullets):
    steps = []

    for b in bullets[:5]:  # take top 5 important points
        clean = b.lstrip("- ").strip()
        steps.append(clean)

    if not steps:
        return "Concept → Explanation → Key Points → Conclusion"

    diagram = steps[0]
    for step in steps[1:]:
        diagram += "\n   ↓\n" + step

    return diagram

# ---------------- YOUTUBE LINKS ----------------
def generate_youtube_links(concepts):
    links = []
    for c in concepts:
        q = urllib.parse.quote_plus(c + " tutorial")
        links.append((c.title(), f"https://www.youtube.com/results?search_query={q}"))
    return links

# ---------------- SAVE HTML ----------------
def save_as_html(notes, diagram, youtube_links):
    html = """
<!DOCTYPE html>
<html>
<head>
<title>Silent Classroom Assistant</title>
<style>
body { font-family: Arial; background: #f4f6f8; }
.container { max-width: 900px; margin: auto; background: white; padding: 30px; }
li { margin-bottom: 12px; }
strong { text-decoration: underline; }
pre { background: #f1f1f1; padding: 15px; }
</style>
</head>
<body>
<div class="container">
<h1>Silent Classroom Assistant</h1>

<h2>Structured Lecture Notes</h2>
<ul>
"""

    for n in notes:
        n = re.sub(r'__(.*?)__', r'<strong>\1</strong>', n)
        html += f"<li>{n[2:]}</li>"

    html += f"""
</ul>

<h2>Conceptual Flow Diagram</h2>
<pre>{diagram}</pre>

<h2>Recommended Learning Resources</h2>
<ul>
"""

    for title, link in youtube_links:
        html += f'<li><a href="{link}" target="_blank">{title}</a></li>'

    html += """
</ul>
<p><i>Generated automatically from lecture audio</i></p>
</div>
</body>
</html>
"""

    with open("lecture_notes.html", "w", encoding="utf-8") as f:
        f.write(html)

# ---------------- MAIN ----------------
def silent_classroom_assistant():
    audio_file = input("Enter ANY audio file path: ").strip()

    raw = speech_to_text(audio_file)
    cleaned = clean_transcript(raw)

    summary = summarize_text(cleaned)
    summary = correct_text(summary)

    bullets = convert_to_bullets(summary)
    concepts = extract_concepts(bullets)
    notes = underline_keywords(bullets, concepts)

    diagram = generate_universal_flow_diagram(bullets)
    youtube_links = generate_youtube_links(concepts)

    save_as_html(notes, diagram, youtube_links)
    print("\n✅ Notes generated successfully: lecture_notes.html")

# ---------------- RUN ----------------
silent_classroom_assistant()
