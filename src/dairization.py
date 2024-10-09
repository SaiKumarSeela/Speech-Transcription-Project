import json
import os
from dotenv import load_dotenv
from src.logger import logging
import time
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class WhisperTranscriber:
    def __init__(self, audio_file,hugging_face_token, device="cpu", compute_type="float32", batch_size=16):
        self.audio_file = audio_file
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.model = None
        self.result_trans = None
        self.result_align = None
        self.diarize_segments = None
        self.hugging_face_token = hugging_face_token
        self.cancel_process = False  # Initialize cancel_process attribute

    def start_process(self):
        """Record the start time of the processing."""
        self.start_time = time.time()
        logging.info("Processing started.")

    def end_process(self):
        """Calculate and return the elapsed time since processing started."""
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            logging.info(f"Processing completed in {elapsed_time:.2f} seconds.")
            return elapsed_time
        return 0
    
    def load_model(self):
        import whisperx
        logging.info("Load the Whisper model.")
        
        model_path = "whisper_model"
        
        # Check if the model directory exists and contains necessary files
        if os.path.exists(model_path) and os.path.isfile(os.path.join(model_path, 'model.bin')):
            try:
                self.model = whisperx.load_model(model_path, self.device, compute_type=self.compute_type)

                logging.info("Model loaded successfully from local directory.")
            except Exception as e:
                logging.info(f"Error loading model from local directory: {e}")
        else:
            logging.info("Model not found locally. Downloading...")
            os.makedirs(model_path, exist_ok=True)
            try:
                # Downloading and saving the model in specified path
                self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type, download_root=model_path)
                logging.info("Model downloaded and saved successfully.")
            except Exception as e:
                logging.info(f"Error downloading the model: {e}")

    def transcribe_audio(self):
        import whisperx
        logging.info("Transcribe audio file.")
        audio = whisperx.load_audio(self.audio_file)
        self.result_trans = self.model.transcribe(audio, batch_size=self.batch_size)

    def align_transcription(self):
        import whisperx
        logging.info("Align the transcription output.")
        model_a, metadata = whisperx.load_align_model(language_code=self.result_trans["language"], device=self.device)
        self.result_align = whisperx.align(self.result_trans["segments"], model_a, metadata, self.audio_file, self.device, return_char_alignments=False)

    def diarize_audio(self):
        import whisperx
        logging.info("Identify multiple speakers in audio.")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token= self.hugging_face_token , device=self.device)
        
        audio_data = whisperx.load_audio(self.audio_file)
        self.diarize_segments = diarize_model(audio_data, min_speakers=2, max_speakers=2)
        
        logging.info(self.diarize_segments.speaker.unique())

        uniq_speakers = self.diarize_segments.speaker.unique()
        
        final_result = whisperx.assign_word_speakers(self.diarize_segments, self.result_align)
        
        return final_result, uniq_speakers

    def save_to_json(self, result, filename='data.json'):
        logging.info("Save transcription results to a JSON file.")
        with open(filename, 'w') as json_file:
            json.dump(result, json_file, indent=4)
        logging.info(f"Dictionary has been successfully stored in {filename}.")


# if __name__ == "__main__":
#     audio_file_path = "chunk2.wav"
    
#     transcriber = WhisperTranscriber(audio_file_path,huggingface_token)
    
#     transcriber.load_model()
#     transcriber.transcribe_audio()
#     transcriber.align_transcription()
    
#     final_result = transcriber.diarize_audio()
#     transcriber.save_to_json(final_result)
    