import whisper
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# -------- STEP 1: Speech to Text (WHISPER) --------
def speech_to_text(audio_file):
    print("Transcribing audio using Whisper...\n")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# -------- STEP 2: Clean Classroom Speech --------
def clean_transcript(text):
    fillers = [
        "you know", "uh", "um", "actually", "basically",
        "kind of", "sort of", "okay", "so", "right"
    ]

    text = text.lower()

    for f in fillers:
        text = text.replace(f, "")

    # remove repeated words (e.g., different different)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -------- STEP 3: Text Summarization (NO PROMPT LEAKAGE) --------
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarizer(
        "Lecture content:\n" + text,
        max_length=180,
        min_length=90,
        do_sample=False
    )

    return summary[0]["summary_text"]

# -------- STEP 4: Convert Summary to Bullet Points --------
def convert_to_bullets(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    bullets = []

    for s in sentences:
        s = s.strip().capitalize()
        if len(s) >= 30:
            bullets.append("- " + s)

    return bullets

# -------- STEP 5: Smart Keyword Underlining (GENERAL & FILTERED) --------
def underline_keywords(bullets, top_n=6):
    text = " ".join(bullets)

    # words that should NEVER be highlighted
    banned_words = {
        "lecture", "following", "avoid", "use", "using",
        "suitable", "study", "content", "questions"
    }

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15
    )

    vectorizer.fit([text])
    keywords = [
        kw for kw in vectorizer.get_feature_names_out()
        if kw.lower() not in banned_words
    ][:top_n]

    formatted = []
    for line in bullets:
        for kw in keywords:
            line = re.sub(
                rf'\b{kw}\b',
                f"__{kw.upper()}__",
                line,
                flags=re.IGNORECASE
            )
        formatted.append(line)

    return formatted

# -------- STEP 6: Main Program --------
def silent_classroom_assistant():
    print("Silent Classroom Assistant running...\n")

    raw_text = speech_to_text("Recording (3).wav")
    cleaned_text = clean_transcript(raw_text)

    print("----- Cleaned Lecture Text -----\n")
    print(cleaned_text)

    print("\nGenerating structured notes...\n")
    summary = summarize_text(cleaned_text)

    bullets = convert_to_bullets(summary)
    structured_notes = underline_keywords(bullets)

    print("----- Structured Lecture Notes -----\n")
    for point in structured_notes:
        print(point)

# -------- RUN PROGRAM --------
silent_classroom_assistant()
