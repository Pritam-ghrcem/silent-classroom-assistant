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

    # remove repeated words (example: different different)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -------- STEP 3: Split Long Text into Chunks --------
def split_text(text, max_words=400):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks

# -------- STEP 4: Summarization (SAFE FOR LONG LECTURES) --------
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    chunks = split_text(text)
    summaries = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")

        summary = summarizer(
            "Lecture content:\n" + chunk,
            max_length=130,
            min_length=60,
            do_sample=False
        )

        summaries.append(summary[0]["summary_text"])

    return " ".join(summaries)

# -------- STEP 5: Convert Summary to Bullet Points --------
def convert_to_bullets(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    bullets = []

    for s in sentences:
        s = s.strip().capitalize()
        if len(s) >= 30:
            bullets.append("- " + s)

    return bullets

# -------- STEP 6: Smart Keyword Underlining (GENERAL) --------
def underline_keywords(bullets, top_n=6):
    text = " ".join(bullets)

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

# -------- STEP 7: Main Program --------
def silent_classroom_assistant():
    print("Silent Classroom Assistant running...\n")

    raw_text = speech_to_text("Recording (4).wav")
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
