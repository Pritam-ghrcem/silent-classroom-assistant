import whisper
import re
from collections import Counter
from transformers import pipeline
from spellchecker import SpellChecker

# -------- STEP 1: Speech to Text --------
def speech_to_text(audio_file):
    print("Transcribing audio...\n")
    model = whisper.load_model("base")
    return model.transcribe(audio_file)["text"]

# -------- STEP 2: Clean Spoken Noise (GENERIC) --------
def clean_transcript(text):
    fillers = [
        "you know", "uh", "um", "actually", "basically",
        "kind of", "sort of", "okay", "so", "right"
    ]

    text = text.lower()

    for f in fillers:
        text = text.replace(f, "")

    # remove word repetitions
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -------- STEP 3: Chunk Long Text (MODEL-SAFE) --------
def split_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# -------- STEP 4: Summarization (TOPIC-INDEPENDENT) --------
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []

    for i, chunk in enumerate(split_text(text), 1):
        print(f"Summarizing part {i}...")
        result = summarizer(
            chunk,
            max_length=120,
            min_length=50,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)

# -------- STEP 5: Generic Spell Correction --------
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

# -------- STEP 6: Remove Weak / Broken Sentences --------
def convert_to_bullets(text):
    bullets = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for s in sentences:
        s = s.strip().capitalize()
        if len(s.split()) < 8:
            continue
        bullets.append("- " + s)

    return bullets

# -------- STEP 7: CONCEPT-BASED HIGHLIGHTING (GENERIC) --------
def underline_keywords(bullets):
    text = " ".join(bullets)

    # extract 2–3 word phrases (topic independent)
    phrases = re.findall(
        r'\b[a-zA-Z]{4,}\s[a-zA-Z]{4,}(?:\s[a-zA-Z]{4,})?\b',
        text.lower()
    )

    freq = Counter(phrases)
    concepts = [p for p, _ in freq.most_common(6)]

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

# -------- STEP 8: MAIN PIPELINE --------
def silent_classroom_assistant():
    print("Silent Classroom Assistant running...\n")

    raw = speech_to_text("Recording (4).wav")
    cleaned = clean_transcript(raw)

    print("Generating notes...\n")

    summary = summarize_text(cleaned)
    summary = correct_text(summary)

    bullets = convert_to_bullets(summary)
    notes = underline_keywords(bullets)

    print("----- Structured Lecture Notes -----\n")
    for line in notes:
        print(line)

# -------- RUN --------
silent_classroom_assistant()
