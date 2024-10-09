from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
from src.logger import logging
from src.dairization import WhisperTranscriber
from src.summarization import summarise_transcript
from src.utils import extract_audio_duration, count_words, display_conversation, extract_speaker_texts, save_transcription
from src.s3_syncer import S3Sync
from datetime import datetime
import pandas as pd
from typing import List, Dict

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
TRAINING_BUCKET_NAME = "focus-transcribe"
timestamp = datetime.now()
timestamp = timestamp.strftime("%m_%d_%y_%H_%M_%S")
s3_sync = S3Sync(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,AWS_REGION)


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class TranscriptionResult(BaseModel):
    conversation: List[str]
    summary_data: Dict
    audio_duration: float
    total_words: int
    words_by_speaker: Dict[str, int]

# Global variable to store processing results
processing_results = {}

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

async def process_audio(file_path: str):
    transcriber = WhisperTranscriber(file_path, huggingface_token)

    yield "data: Loading model...\n"
    transcriber.load_model()
    yield "data: Model loaded successfully\n"

    yield "data: Transcribing audio...\n"
    transcriber.transcribe_audio()
    yield "data: Transcription completed\n"

    yield "data: Aligning transcription...\n"
    transcriber.align_transcription()
    yield "data: Alignment completed\n"

    yield "data: Diarizing audio...\n"
    final_result, uniq_speakers = transcriber.diarize_audio()
    yield "data: Diarization completed\n"

    transcriber.save_to_json(final_result)
    conversation = display_conversation(filename='data.json', uniq_speakers=uniq_speakers)
    speaker_texts = extract_speaker_texts(conversation)

    directory_path = save_transcription(conversation=conversation)
    aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/transcription/{timestamp}"
    s3_sync.sync_folder_to_s3(folder = directory_path,aws_bucket_name=TRAINING_BUCKET_NAME)
    logging.info("Succesfully transcriptions are saved to s3 bucket")


    yield "data: Generating summaries...\n"
    individual_summary = {}
    for speaker, speeches in speaker_texts.items():
        individual_summary[speaker] = summarise_transcript(groq_api_key=groq_api_key,  transcript=speeches)
    
    summary_content = summarise_transcript(groq_api_key=groq_api_key,transcript=conversation)
    
    summary_data = {
        "Speaker": list(individual_summary.keys()) + ["Total Summary"],
        "Summary": list(individual_summary.values()) + [summary_content]
    }

    audio_duration = extract_audio_duration(file_path)
    total_words, words_by_speaker = count_words(conversation)
    

    processing_results['conversation'] = conversation
    processing_results['summary_data'] = summary_data
    processing_results['audio_duration'] = audio_duration
    processing_results['total_words'] = total_words
    processing_results['words_by_speaker'] = words_by_speaker

    yield "data: Processing complete\n"

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = "uploaded_audio.wav"
    
    # Save uploaded audio file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Stream status updates as the processing progresses
    return StreamingResponse(process_audio(file_path), media_type="text/event-stream")

@app.get("/summary/")
async def get_summary():
    if 'summary_data' not in processing_results:
        return {"error": "Summary not available. Process an audio file first."}
    return processing_results['summary_data']

@app.get("/stats/")
async def get_stats():
    if 'audio_duration' not in processing_results:
        return JSONResponse(
            status_code=400,
            content={"error": "Stats not available. Process an audio file first."}
        )
    return {
        "audio_duration": processing_results['audio_duration'],
        "total_words": processing_results['total_words'],
        "words_by_speaker": processing_results['words_by_speaker']
    }

@app.get("/transcription/")
async def get_transcription():
    if 'conversation' not in processing_results:
        return {"error": "Transcription not available. Process an audio file first."}
    return {"conversation": processing_results['conversation']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
