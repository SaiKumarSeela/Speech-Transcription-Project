import streamlit as st
import json
import os
from src.logger import logging
from src.dairization import WhisperTranscriber
from dotenv import load_dotenv
from src.summarization import summarise_transcript
from src.utils import extract_audio_duration, count_words, display_conversation, extract_speaker_texts
import pandas as pd

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

def show_transcription(conversation):
    st.subheader("Transcription")
    for entry in conversation:
        st.markdown(f'<div class="transcript-line">{entry}</div>', unsafe_allow_html=True)

def show_summary(summary_data):
    st.subheader("Summary")
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("summary.csv", index=False)

    st.table(summary_df)

def show_stats(audio_duration, total_words, words_by_speaker):
    # Creating a DataFrame to compile the statistics
    stat_data = {
        'Audio Duration (m)': [audio_duration],
        'Total Words': [total_words],
    }
    
    # Add words by each speaker dynamically to the DataFrame
    for speaker, word_count in words_by_speaker.items():
        stat_data[f'Words by {speaker}'] = [word_count]

    stats_df = pd.DataFrame(stat_data)
    stats_df.to_csv("output.csv", index=False)
    st.subheader("Statistics")
    st.table(stats_df)

def show_topics():
    st.subheader("Topics")
    st.text("This is where Topics content would be displayed.")

def show_intents():
    st.subheader("Intents")
    st.text("This is where Intents content would be displayed.")

def main(huggingface_token,groq_api_key):
    st.set_page_config(layout="wide")
    # Custom CSS for centering content and styling clickable headers
    st.markdown("""
    <style>
    .stButton > button {
        background-color: transparent; /* No background */
        border: none;                  /* No border */
        color: white !important;       /* Default text color */
        font-weight: bold;             /* Bold text */
        cursor: pointer;               /* Pointer cursor on hover */
    }
    .stButton > button:hover {
        color: #e6e6e6 !important;     /* Hover color */
        text-decoration: underline;    /* Underline on hover */
    }
    .selected {
        color: red !important;         /* Text color when selected */
        text-decoration: underline;     /* Underline when selected */
    }
    .transcript-line {
        border-bottom: 1px solid #e6e6e6;
        padding-bottom: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Audio Transcription and Speaker Diarization")

    audio_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=['wav', 'mp3'])

    if audio_file is not None:
        # Save uploaded audio file
        with open("uploaded_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())

        transcriber = WhisperTranscriber("uploaded_audio.wav", huggingface_token)

        transcriber.start_process()  # Record start time
        if 'model_loaded' not in st.session_state:
            with st.spinner("Loading model..."):
                transcriber.load_model()
                st.session_state.model_loaded = True  # Track model loading status
                st.markdown("✅ Model Loaded successfully!")  # Immediate feedback

        if 'conversation' not in st.session_state:
            with st.spinner("Transcribing audio..."):
                transcriber.transcribe_audio()
                st.markdown("✅ Transcribing completed!")  # Immediate feedback

            with st.spinner("Aligning transcription..."):
                transcriber.align_transcription()
                st.markdown("✅ Alignment completed!")  # Immediate feedback

            with st.spinner("Diarizing audio..."):
                final_result, uniq_speakers = transcriber.diarize_audio()
                st.markdown("✅ Diarization completed!")  # Immediate feedback

            # Save results to JSON
            transcriber.save_to_json(final_result)
            conversation = display_conversation(filename='data.json', uniq_speakers=uniq_speakers)
            speaker_texts = extract_speaker_texts(conversation)
            print(speaker_texts)
            individual_summary = {}
            for speaker, speeches in speaker_texts.items():
                individual_summary[speaker] = summarise_transcript(groq_api_key=groq_api_key, mp3file_path="uploaded_audio.wav", transcript=speeches)
                
            print(individual_summary)
            summary_content = summarise_transcript(groq_api_key=groq_api_key, mp3file_path="uploaded_audio.wav", transcript=conversation)
            
            # Create a DataFrame for displaying summaries
            summary_data = {
                "Speaker": list(individual_summary.keys()) + ["Total Summary"],
                "Summary": list(individual_summary.values()) + [summary_content]
            }
            print(summary_data)
            logging.info(f"summary data: {summary_data}")
            # Gathered data from previous steps
            audio_duration = extract_audio_duration('uploaded_audio.wav')
            total_words, words_by_speaker = count_words(conversation)
            elapsed_time = transcriber.end_process()  # Record end time and calculate elapsed time
            st.success(f"Audio processing complete in {elapsed_time:.2f} seconds!")

            # Store results in session state for future use
            st.session_state.conversation = conversation
            st.session_state.summary_data = summary_data
            st.session_state.audio_duration = audio_duration
            st.session_state.total_words = total_words
            st.session_state.words_by_speaker = words_by_speaker
        
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown('<p class="column-header">🗣️ Transcription</p>', unsafe_allow_html=True)
            show_transcription(st.session_state.conversation)
        
        with col2:
            cols = st.columns(4)
            sections = ["📌Topics", "🗣️Intents", "📒Summary", "📊Stats"]
            
            if 'selected_section' not in st.session_state:
                st.session_state.selected_section = "📌Topics" 
            
            for i, section in enumerate(sections):
                is_selected = st.session_state.selected_section == section
                
                if cols[i].button(section, key=section):
                    st.session_state.selected_section = section
                
            # Display content based on selection
            if st.session_state.selected_section == "📌Topics":
                show_topics()
            elif st.session_state.selected_section == "🗣️Intents":
                show_intents()
            elif st.session_state.selected_section == "📒Summary":
                show_summary(st.session_state.summary_data)
            elif st.session_state.selected_section == "📊Stats":
                show_stats(st.session_state.audio_duration, 
                           st.session_state.total_words, 
                           st.session_state.words_by_speaker)

if __name__ == "__main__":
    main(huggingface_token,groq_api_key)