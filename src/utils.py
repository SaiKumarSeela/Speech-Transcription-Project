from os import path
from pydub import AudioSegment
from pytube import YouTube
from moviepy.editor import *
import os
import assemblyai as aai
import json
import wave
import re
from src.logger import logging


def convertmp3_to_wav(input_file_path, output_file_path):
    # Convert MP3 to WAV
    sound = AudioSegment.from_mp3(input_file_path)
    sound.export(output_file_path, format="wav")

    return output_file_path


def extract_audio_from_youtube(youtube_url):
    try:
        # Create a YouTube object with the provided URL
        video = YouTube(youtube_url)

        # Filter for audio streams and select the first available one
        audio_stream = video.streams.filter(only_audio=True).first()

        # Download the audio stream
        audio_stream.download()

        video_path =audio_stream.default_filename
        audio_path = video_path.replace(".mp4",".mp3")

        clip = AudioFileClip(video_path) # type: ignore
        clip.write_audiofile(audio_path)
        
        os.remove(video_path)
        return audio_path

    except Exception as e:
        print(f"An error occurred: {e}")

def get_transcript_using_assemblyai(assembly_api_key, mp3file_path):
    aai.settings.api_key = assembly_api_key 

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(mp3file_path)

    if transcript.status == aai.TranscriptStatus.error:
        return transcript.error
    else:
        return transcript.text
    

def extract_audio_duration(file_path):
    logging.info("Extracts the duration of an audio file in seconds.")
    
    if not os.path.isfile(file_path):
        return 'File does not exist.'
    
    try:
        with wave.open(file_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration_in_seconds = frames / float(rate)

        # Convert duration from seconds to minutes
        duration_in_minutes = duration_in_seconds / 60
        
        logging.info(f"Total Duration: {duration_in_minutes:.2f} minutes")
        return round(duration_in_minutes, 2)  # Round to 2 decimal places
    
    except Exception as e:
        return str(e)

def count_words(transcript):
    logging.info("Counts total words and words spoken by each speaker.")
    total_words = 0
    speaker_word_count = {}

    for segment in transcript:
        # Assuming each segment is formatted as "<strong>Speaker X:</strong> text"
        if ':' in segment:
            speaker, text = segment.split(':', 1)
            text = text.strip()
            word_count = len(text.split())
            
            total_words += word_count
            
            # Update speaker word count
            speaker_name = speaker.strip('<strong>')  # Remove markdown formatting
            if speaker_name not in speaker_word_count:
                speaker_word_count[speaker_name] = 0
            speaker_word_count[speaker_name] += word_count
        
    logging.info(f"total_words: {total_words}, speaker_work_count: {speaker_word_count}")

    return total_words, speaker_word_count

def display_conversation(filename='data.json', uniq_speakers=None):
    logging.info("Display the conversation from the JSON file.")
    
    with open(filename, 'r') as file:
        data = json.load(file)

    # Create dynamic speaker mapping based on unique speakers
    if uniq_speakers is None:
        uniq_speakers = list(set(segment['speaker'] for segment in data['segments']))

    speaker_map = {speaker: f"Speaker {i + 1}" for i, speaker in enumerate(uniq_speakers)}

    conversation = []
    current_speaker = None
    current_text = ""

    for segment in data['segments']:
        speaker = segment['speaker']
        text = segment['text'].strip()

        if speaker == current_speaker:
            # If the same speaker continues, append to current text
            current_text += f" {text}"
        else:
            # If a new speaker starts, save the previous text and reset
            if current_speaker is not None:
                conversation.append(f"<strong>{speaker_map[current_speaker]}:</strong> {current_text.strip()}")  # Add previous speaker's text
            
            # Update to the new speaker
            current_speaker = speaker
            current_text = text

    # Don't forget to add the last speaker's text after the loop
    if current_speaker is not None:
        conversation.append(f"<strong>{speaker_map[current_speaker]}:</strong> {current_text.strip()}")
            
    return conversation

def extract_speaker_texts(conversation):
    logging.info("Extract individual speaker texts from the conversation output.")
    speaker_texts = {}
    speaker_pattern = re.compile(r'<strong>(Speaker \d+):</strong>(.*)')

    for line in conversation:
        match = speaker_pattern.match(line)
        if match:
            speaker = match.group(1).strip() 
            speech = match.group(2).strip()  
            
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            
            speaker_texts[speaker].append(speech)

    return speaker_texts

def save_transcription(conversation):
    directory = 'transcriptions'
    if not os.path.exists(directory):
        os.makedirs(directory)

    full_transcription_file = os.path.join(directory, 'transcription_with_speakers.txt')
    no_speakers_file = os.path.join(directory, 'transcription_with_no_speakers.txt')

    # Empty the existing files if present
    open(full_transcription_file, 'w').close()
    open(no_speakers_file, 'w').close()

    with open(full_transcription_file, 'a') as file_full:
        with open(no_speakers_file, 'a') as file_no_speakers:
            for entry in conversation:
                # Remove HTML tags for full transcription
                entry_no_tags = re.sub(r'<.*?>', '', entry)
                file_full.write(entry_no_tags + '\n')
                
                # Remove speaker name for no speakers transcription
                entry_without_speaker = entry_no_tags.split(': ', 1)[-1]
                file_no_speakers.write(entry_without_speaker + ' ')
    
    return directory
