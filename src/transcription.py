import asyncio
import json
import os
from dotenv import load_dotenv
from deepgram import Deepgram

load_dotenv()

# Deepgram API key
DG_KEY = os.getenv("DG_KEY")

# Path to save the transcript text file
TRANSCRIPT_FILE = "transcript.txt"

async def get_transcript(audio_file_path):
    try:
        # STEP 1: Create a Deepgram client using the API key
        deepgram = Deepgram(DG_KEY)

        # Check if the audio file exists
        if os.path.exists(audio_file_path):
            # Open the audio file
            with open(audio_file_path, 'rb') as audio_file:
                source = {'buffer': audio_file, 'mimetype': 'audio/wav'}

                # STEP 2: Configure transcription options
                options = {
                    'punctuate': True,
                    'model': 'nova-2',
                    'language': 'en-US'
                }

                # STEP 3: Call the transcribe method with the audio source and options
                response = await deepgram.transcription.prerecorded(source, options)

                # Extracting the transcript from the response
                transcript = ""
                if response and "results" in response:
                    for alternative in response["results"]["channels"][0]["alternatives"]:
                        transcript += alternative["transcript"] + " "

                # Print and save the extracted transcript
                print(transcript.strip())

                # STEP 4: Write the transcript to a text file
                with open(TRANSCRIPT_FILE, "w") as transcript_file:
                    transcript_file.write(transcript.strip())

                print("Transcript text file generated successfully.")
                return transcript.strip()


        else:
            print("Audio file not found")
            return

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    try:
        audio_file_path = "chunk1.wav" 
        asyncio.get_event_loop().run_until_complete(get_transcript(audio_file_path))
    except Exception as e:
        print(f"Failed to run async main: {e}")