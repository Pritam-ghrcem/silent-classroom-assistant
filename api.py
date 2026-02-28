from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import pdfkit

from assistant_13 import silent_classroom_assistant

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PDF CONFIG ----------------
config = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "Silent Classroom API Running 🚀"}

# ---------------- PROCESS AUDIO ----------------
@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):

    allowed_types = (".wav", ".mp3", ".m4a")
    if not file.filename.lower().endswith(allowed_types):
        return {"status": "error", "message": "Only audio files allowed"}

    unique_id = str(uuid.uuid4())
    file_path = f"temp_{unique_id}_{file.filename}"

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📁 File saved: {file_path}")

        if not os.path.exists(file_path):
            raise Exception("File saving failed")

        # Run AI pipeline
        result = silent_classroom_assistant(file_path)

        print("✅ Processing completed")

        # 🔥 Ensure result is valid dictionary
        if not isinstance(result, dict):
            result = {
                "notes": [str(result)],
                "keywords": [],
                "youtube_links": []
            }

        # 🔥 Ensure keys exist
        result.setdefault("notes", [])
        result.setdefault("keywords", [])
        result.setdefault("youtube_links", [])

        return {
            "status": "success",
            "data": result,
            "download_html": "http://127.0.0.1:8000/download-html",
            "download_pdf": "http://127.0.0.1:8000/download-pdf"
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑 Deleted temp file: {file_path}")

# ---------------- DOWNLOAD HTML ----------------
@app.get("/download-html")
def download_html():

    file_path = "lecture_notes.html"

    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename="lecture_notes.html",
            media_type="text/html"
        )

    return {"error": "HTML file not found"}

# ---------------- DOWNLOAD PDF ----------------
@app.get("/download-pdf")
def download_pdf():

    html_file = "lecture_notes.html"
    pdf_file = "lecture_notes.pdf"

    if not os.path.exists(html_file):
        return {"error": "HTML file not found"}

    try:
        pdfkit.from_file(html_file, pdf_file, configuration=config)

        return FileResponse(
            path=pdf_file,
            filename="lecture_notes.pdf",
            media_type="application/pdf"
        )

    except Exception as e:
        return {"error": str(e)}