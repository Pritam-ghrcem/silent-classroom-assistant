import speech_recognition as sr

# -------- STEP 1: Speech to Text --------
def speech_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        return text
    except:
        return "Error recognizing audio"

# -------- STEP 2: Simple Text Summarization --------
def summarize_text(text):
    sentences = text.split(".")
    if len(sentences) >= 3:
        return sentences[0] + "." + sentences[1] + "."
    else:
        return text

# -------- STEP 3: Main Program --------
def silent_classroom_assistant():
    print("Silent Classroom Assistant running...\n")

    lecture_text = speech_to_text("audio.wav")
    print("----- Lecture Text -----\n")
    print(lecture_text)

    print("\n----- Lecture Summary -----\n")
    summary = summarize_text(lecture_text)
    print(summary)

# Run program
silent_classroom_assistant()
