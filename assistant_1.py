import speech_recognition as sr
from transformers import pipeline

# -------- STEP 1: Speech to Text --------
def speech_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        return "Error recognizing audio"

# -------- STEP 2: Text Summarization --------
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# -------- STEP 3: Main Program --------
def silent_classroom_assistant():
    print("Silent Classroom Assistant running...\n")

    lecture_text = speech_to_text("audio.wav")
    print("----- Lecture Text -----\n")
    print(lecture_text)

    print("\nGenerating summary...\n")
    summary = summarize_text(lecture_text)

    print("----- Lecture Summary -----\n")
    print(summary)

# Run program
silent_classroom_assistant()
