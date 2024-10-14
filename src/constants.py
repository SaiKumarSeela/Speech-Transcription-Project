import os


""" Constants related to Transcription model """

MODEL_NAME = "distil-large-v2"
MODEL_DIR = os.path.join(os.getcwd(),"distil_whisper_model")
MODEL_PATH = os.path.join(MODEL_DIR,"models--Systran--faster-distil-whisper-large-v2",
                          "snapshots","fe9b404fc56de3f7c38606ef9ba6fd83526d05e4")
COMPUTE_TYPE= "float32"


""" Constants related to Summarization """

SUMMARIZATION_MODEL = "Llama3-8b-8192"


""" Constants related to S3 """
BUCKET_NAME= "focus-transcribe"

