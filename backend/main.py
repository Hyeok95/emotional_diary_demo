from fastapi import FastAPI, UploadFile, File
from models import DiaryRequest, DiaryResponse
from openai_utils import analyze_diary, transcribe_audio
import os

app = FastAPI()

@app.post("/analyze", response_model=DiaryResponse)
def analyze(req: DiaryRequest):
    emotion, feedback, therapy = analyze_diary(req.content, model=req.model)
    return DiaryResponse(emotion=emotion, feedback=feedback, therapy=therapy)

@app.post("/stt")
def stt(file: UploadFile = File(...)):
    # 파일 저장
    file_location = f"/app/data/voice_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    # Whisper로 STT
    transcript = transcribe_audio(file_location)
    print(transcript)
    # 파일 삭제
    os.remove(file_location)
    return {"text": transcript}
