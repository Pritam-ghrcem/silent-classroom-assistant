import mysql.connector
import whisper
import re
from collections import Counter
from transformers import pipeline
import urllib.parse
import os
import requests

# ---------------- CONFIG ----------------
YOUTUBE_API_KEY = "PUT_YOUR_API_KEY_HERE"   # ⚠️ Replace this

# ---------------- LOAD MODELS ----------------
print("Loading models...")
model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Models loaded!")

# ---------------- DATABASE ----------------
def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="Pritam123",
            database="silent_classroom_assistant"
        )
    except Exception as e:
        print("DB Error:", e)
        return None

def insert_data(audio_file, transcript, summary, notes, keywords):
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
    conn.close()

# ---------------- SPEECH ----------------
def speech_to_text(audio_file):
    if not os.path.exists(audio_file):
        raise Exception("File not found")

    result = model.transcribe(audio_file)
    return result["text"]

# ---------------- CLEAN ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- SUMMARY ----------------
def summarize(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    summary = ""

    for c in chunks:
        res = summarizer(c, max_length=80, min_length=30, do_sample=False)
        summary += res[0]['summary_text'] + " "

    return summary

# ---------------- BULLETS ----------------
def generate_bullets(text):
    sents = re.split(r'(?<=[.!?])\s+', text)
    out = []

    for s in sents:
        if len(s.split()) > 6:
            clean = s.replace("_", "").capitalize()
            out.append("• " + clean)

    return out if out else ["• " + text]

# ---------------- KEYWORDS ----------------
def extract_keywords(bullets):
    text = " ".join(bullets)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)

    freq = Counter(words)
    return [w for w, _ in freq.most_common(5)]

# ---------------- YOUTUBE ----------------
def youtube_videos(concepts):
    videos = []

    if YOUTUBE_API_KEY == "PUT_YOUR_API_KEY_HERE":
        print("⚠️ YouTube API key not set!")
        return []

    for c in concepts:
        url = "https://www.googleapis.com/youtube/v3/search"

        params = {
            "part": "snippet",
            "q": c + " explained",
            "key": YOUTUBE_API_KEY,
            "type": "video",
            "maxResults": 1
        }

        try:
            res = requests.get(url, params=params).json()

            if "items" in res and len(res["items"]) > 0:
                vid = res["items"][0]

                videos.append({
                    "topic": c,
                    "video_id": vid["id"]["videoId"]
                })

        except Exception as e:
            print("YouTube Error:", e)

    return videos

# ---------------- HTML ----------------
def save_html(notes, videos):
    html = "<h1>Lecture Notes</h1><ul>"

    for n in notes:
        html += f"<li>{n}</li>"

    html += "</ul><h2>Videos</h2>"

    for v in videos:
        html += f"""
        <iframe width="400" height="200"
        src="https://www.youtube.com/embed/{v['video_id']}"></iframe>
        """

    with open("lecture_notes.html", "w", encoding="utf-8") as f:
        f.write(html)

# ---------------- MAIN ----------------
def silent_classroom_assistant(file):
    try:
        text = speech_to_text(file)
        text = clean_text(text)

        summary = summarize(text)

        notes = generate_bullets(summary)
        keys = extract_keywords(notes)
        vids = youtube_videos(keys)

        insert_data(file, text, summary, notes, keys)
        save_html(notes, vids)

        return {
            "notes": notes,
            "keywords": keys,
            "youtube_videos": vids
        }

    except Exception as e:
        print("Error:", e)
        return {
            "notes": [],
            "keywords": [],
            "youtube_videos": [],
            "error": str(e)
        }