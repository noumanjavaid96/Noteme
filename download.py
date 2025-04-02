import streamlit as st
from groq import Groq
import json
import os
import time
import numpy as np
import tempfile
from io import BytesIO, StringIO
from md2pdf.core import md2pdf
from dotenv import load_dotenv
from datetime import datetime
import threading
from download import download_video_audio, delete_download

# Override the max file size (40MB in bytes)
MAX_FILE_SIZE = 41943040  # 40MB in bytes
FILE_TOO_LARGE_MESSAGE = "File too large. Maximum size is 40MB."

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
audio_file_path = None

# Initialize session states
if 'api_key' not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY
    
if 'recording' not in st.session_state:
    st.session_state.recording = False
    
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
    
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
    
if 'groq' not in st.session_state:
    if st.session_state.api_key:
        st.session_state.groq = Groq(api_key=st.session_state.api_key)

# Set page configuration
st.set_page_config(
    page_title="ScribeWizard 🧙‍♂️",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed model selections
LLM_MODEL = "deepseek-r1-distill-llama-70b"
TRANSCRIPTION_MODEL = "distil-whisper-large-v3-en"

class GenerationStatistics:
    def __init__(self, input_time=0, output_time=0, input_tokens=0, output_tokens=0, total_time=0, model_name=LLM_MODEL):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time # Sum of queue, prompt (input), and completion (output) times
        self.model_name = model_name

    def get_input_speed(self):
        """ Tokens per second calculation for input """
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0

    def get_output_speed(self):
        """ Tokens per second calculation for output """
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0

    def add(self, other):
        """ Add statistics from another GenerationStatistics object to this one. """
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")
        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        return (f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s Model: {self.model_name}\n\n"
                f"| Metric | Input | Output | Total |\n"
                f"|-----------------|----------------|-----------------|----------------|\n"
                f"| Speed (T/s) | {self.get_input_speed():.2f} | {self.get_output_speed():.2f} | {(self.input_tokens + self.output_tokens) / self.total_time if self.total_time != 0 else 0:.2f} |\n"
                f"| Tokens | {self.input_tokens} | {self.output_tokens} | {self.input_tokens + self.output_tokens} |\n"
                f"| Inference Time (s) | {self.input_time:.2f} | {self.output_time:.2f} | {self.total_time:.2f} |")

class NoteSection:
    def __init__(self, structure, transcript):
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}
        
        with st.expander("Raw Transcript", expanded=False):
            st.markdown(transcript)

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            st.error(f"Error updating content: {e}")

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def return_existing_contents(self, level=1) -> str:
        existing_content = ""
        for title, content in self.structure.items():
            if self.contents[title].strip():
                existing_content += f"{'#' * level} {title}\n{self.contents[title]}\n\n"
            if isinstance(content, dict):
                existing_content += self.get_markdown_content(content, level + 1)
        return existing_content

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
        for title, content in structure.items():
            if self.contents[title].strip():
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        """ Returns the markdown styled pure string with the contents. """
        if structure is None:
            structure = self.structure
        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content

# Audio recorder functionality
class AudioRecorder:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.thread = None
    
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.thread = threading.Thread(target=self._record_audio)
        self.thread.start()
        
    def _record_audio(self):
        import sounddevice as sd
        with sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.sample_rate):
            while self.recording:
                time.sleep(0.1)
    
    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_data.append(indata.copy())
    
    def stop_recording(self):
        self.recording = False
        if self.thread:
            self.thread.join()
        
        if not self.audio_data:
            return None
            
        # Concatenate all audio chunks
        import numpy as np
        import soundfile as sf
        audio = np.concatenate(self.audio_data, axis=0)
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
        sf.write(temp_file.name, audio, self.sample_rate)
        
        return temp_file.name

def transcribe_audio_with_groq(audio_file_path):
    """Transcribe audio file using Groq's transcription API"""
    if not st.session_state.api_key:
        st.error("Please provide a valid Groq API key in the sidebar.")
        return ""
    
    client = Groq(api_key=st.session_state.api_key)
    
    try:
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file_path, file.read()),
                model=TRANSCRIPTION_MODEL,
                response_format="verbose_json"
            )
            return transcription.text
    except Exception as e:
        st.error(f"Error transcribing audio with Groq: {e}")
        return ""

def process_transcript(transcript):
    """Process transcript with Groq's DeepSeek model for highly structured notes"""
    if not st.session_state.api_key:
        st.error("Please provide a valid Groq API key in the sidebar.")
        return None
    
    client = Groq(api_key=st.session_state.api_key)
    
    # Enhanced structure for better organization
    structure = {
        "Executive Summary": "",
        "Key Insights": "",
        "Action Items": "",
        "Questions & Considerations": "",
        "Detailed Analysis": {
            "Context & Background": "",
            "Main Discussion Points": "",
            "Supporting Evidence": "",
            "Conclusions & Recommendations": ""
        }
    }
    
    prompt = f"""
    You are an expert note organizer with exceptional skills in creating structured, clear, and comprehensive notes. 
    Please analyze the following transcript and transform it into highly organized notes:

    ```
    {transcript}
    ```

    Create a well-structured document with the following sections:

    # Executive Summary
    - Provide a concise 3-5 sentence overview of the main topic and key takeaways
    - Use clear, direct language

    # Key Insights
    - Extract 5-7 critical insights as bullet points
    - Each insight should be bolded and followed by 1-2 supporting sentences
    - Organize these insights in order of importance

    # Action Items
    - Create a table with these columns: Action | Owner/Responsible Party | Timeline | Priority
    - List all tasks, assignments, or follow-up items mentioned
    - If information is not explicitly stated, indicate with "Not specified"

    # Questions & Considerations
    - List all questions raised during the discussion
    - Include concerns or areas needing further exploration
    - For each question, provide brief context explaining why it matters

    # Detailed Analysis
    
    ## Context & Background
    - Summarize relevant background information
    - Explain the context in which the discussion took place
    - Include references to prior work or decisions if mentioned
    
    ## Main Discussion Points
    - Create subsections for each major topic discussed
    - Use appropriate formatting (bullet points, numbered lists) to organize information
    - Include direct quotes when particularly significant, marked with ">"
    
    ## Supporting Evidence
    - Create a table summarizing any data, evidence, or examples mentioned
    - Include source information when available
    
    ## Conclusions & Recommendations
    - Summarize the conclusions reached
    - List any recommendations or next steps discussed
    - Note any decisions that were made

    Make extensive use of markdown formatting:
    - Use tables for structured information
    - Use bold for emphasis on important points
    - Use bullet points and numbered lists for clarity
    - Use headings and subheadings to organize content
    - Include blockquotes for direct citations

    Your notes should be comprehensive but concise, focusing on extracting the maximum value from the transcript.
    """
    
    try:
        stats = GenerationStatistics(model_name=LLM_MODEL)
        start_time = time.time()
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            temperature=0.3,  # Lower temperature for more structured output
            max_tokens=4096,
            top_p=0.95,
            stream=True
        )
        
        input_time = time.time() - start_time
        stats.input_time = input_time
        
        note_section = NoteSection(structure, transcript)
        current_section = None
        current_subsection = None
        notes_content = ""
        
        section_markers = {
            "# Executive Summary": "Executive Summary",
            "## Executive Summary": "Executive Summary",
            "# Key Insights": "Key Insights",
            "## Key Insights": "Key Insights",
            "# Action Items": "Action Items",
            "## Action Items": "Action Items",
            "# Questions & Considerations": "Questions & Considerations",
            "## Questions & Considerations": "Questions & Considerations",
            "# Detailed Analysis": "Detailed Analysis",
            "## Detailed Analysis": "Detailed Analysis",
            "## Context & Background": "Context & Background",
            "### Context & Background": "Context & Background",
            "## Main Discussion Points": "Main Discussion Points",
            "### Main Discussion Points": "Main Discussion Points",
            "## Supporting Evidence": "Supporting Evidence",
            "### Supporting Evidence": "Supporting Evidence",
            "## Conclusions & Recommendations": "Conclusions & Recommendations",
            "### Conclusions & Recommendations": "Conclusions & Recommendations"
        }
        
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                notes_content += content
                
                # Check for section markers in the accumulated content
                for marker, section in section_markers.items():
                    if marker in notes_content:
                        if section in ["Context & Background", "Main Discussion Points", 
                                      "Supporting Evidence", "Conclusions & Recommendations"]:
                            current_section = "Detailed Analysis"
                            current_subsection = section
                        else:
                            current_section = section
                            current_subsection = None
                
                # Update the appropriate section
                if current_section and current_section != "Detailed Analysis":
                    note_section.update_content(current_section, content)
                elif current_section == "Detailed Analysis" and current_subsection:
                    note_section.update_content(current_subsection, content)
        
        output_time = time.time() - start_time - input_time
        stats.output_time = output_time
        stats.total_time = time.time() - start_time
        
        # Display statistics in expandable section
        with st.expander("Generation Statistics", expanded=False):
            st.markdown(str(stats))
        
        return note_section
        
    except Exception as e:
        st.error(f"Error processing transcript: {e}")
        return None

def export_notes(notes, format="markdown"):
    """Export notes in the specified format"""
    if format == "markdown":
        markdown_content = notes.get_markdown_content()
        # Create a download button for the markdown file
        st.download_button(
            label="Download Markdown",
            data=markdown_content,
            file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    elif format == "pdf":
        markdown_content = notes.get_markdown_content()
        pdf_file = BytesIO()
        md2pdf(pdf_file, markdown_content)
        pdf_file.seek(0)
        
        # Create a download button for the PDF file
        st.download_button(
            label="Download PDF",
            data=pdf_file,
            file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

def main():
    st.title("🧙‍♂️ ScribeWizard")
    st.markdown("Transform speech into highly structured notes with AI magic")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Groq API Key", value=st.session_state.api_key or "", type="password")
        
        if api_key:
            st.session_state.api_key = api_key
            if 'groq' not in st.session_state or st.session_state.groq is None:
                st.session_state.groq = Groq(api_key=api_key)
        
        st.markdown("---")
        st.info("Using DeepSeek-R1-Distill-Llama-70B model for note generation and Distil Whisper for transcription")
        
    # Input methods tabs
    input_method = st.radio("Choose input method:", ["Live Recording", "Upload Audio", "YouTube URL", "Text Input"])
    
    audio_recorder = AudioRecorder()
    
    if input_method == "Live Recording":
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.recording:
                if st.button("Start Recording 🎤", key="start_rec"):
                    st.session_state.recording = True
                    audio_recorder.start_recording()
                    st.rerun()
            else:
                if st.button("Stop Recording ⏹️", key="stop_rec"):
                    audio_file = audio_recorder.stop_recording()
                    st.session_state.recording = False
                    
                    if audio_file:
                        st.session_state.audio_data = audio_file
                        st.success("Recording saved!")
                        
                        # Auto-transcribe using Groq
                        with st.spinner("Transcribing audio with Groq..."):
                            transcript = transcribe_audio_with_groq(audio_file)
                            if transcript:
                                st.session_state.transcript = transcript
                                st.success("Transcription complete!")
                    st.rerun()
        
        with col2:
            if st.session_state.recording:
                st.markdown("#### 🔴 Recording in progress...")
                
                # Animated recording indicator
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress((i + 1) % 101)
                    
                    # Break if recording stopped
                    if not st.session_state.recording:
                        break
                st.rerun()
            
        if st.session_state.audio_data:
            st.audio(st.session_state.audio_data)
            
            if st.session_state.transcript:
                if st.button("Generate Structured Notes", key="generate_live"):
                    with st.spinner("Creating highly structured notes..."):
                        notes = process_transcript(st.session_state.transcript)
                        
                        if notes:
                            st.success("Notes generated successfully!")
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Export as Markdown", key="md_live"):
                                    export_notes(notes, "markdown")
                            with col2:
                                if st.button("Export as PDF", key="pdf_live"):
                                    export_notes(notes, "pdf")
    
    elif input_method == "Upload Audio":
        uploaded_file = st.file_uploader("Upload an audio file (max 40MB)", type=["mp3", "wav", "m4a", "ogg"])
        
        if uploaded_file:
            file_size = uploaded_file.size
            if file_size > MAX_FILE_SIZE:
                st.error(f"File size ({file_size/1048576:.2f}MB) exceeds the maximum allowed size of 40MB.")
            else:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_file_path = tmp_file.name
                
                st.audio(uploaded_file)
                
                if st.button("Transcribe and Generate Notes", key="transcribe_upload"):
                    with st.spinner("Transcribing audio with Groq..."):
                        transcript = transcribe_audio_with_groq(audio_file_path)
                        
                        if transcript:
                            st.session_state.transcript = transcript
                            
                            with st.spinner("Creating highly structured notes..."):
                                notes = process_transcript(transcript)
                                
                                if notes:
                                    st.success("Notes generated successfully!")
                                    
                                    # Export options
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("Export as Markdown", key="md_upload"):
                                            export_notes(notes, "markdown")
                                    with col2:
                                        if st.button("Export as PDF", key="pdf_upload"):
                                            export_notes(notes, "pdf")
            
    elif input_method == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube URL:")
        
        if youtube_url:
            if st.button("Process YouTube Content", key="process_yt"):
                with st.spinner("Downloading YouTube content..."):
                    try:
                        audio_path = download_video_audio(youtube_url)
                        
                        if audio_path:
                            st.success("Video downloaded successfully!")
                            st.audio(audio_path)
                            
                            with st.spinner("Transcribing audio with Groq..."):
                                transcript = transcribe_audio_with_groq(audio_path)
                                
                                if transcript:
                                    st.session_state.transcript = transcript
                                    
                                    with st.spinner("Creating highly structured notes..."):
                                        notes = process_transcript(transcript)
                                        
                                        if notes:
                                            st.success("Notes generated successfully!")
                                            
                                            # Export options
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                if st.button("Export as Markdown", key="md_yt"):
                                                    export_notes(notes, "markdown")
                                            with col2:
                                                if st.button("Export as PDF", key="pdf_yt"):
                                                    export_notes(notes, "pdf")
                                    
                            # Clean up downloaded files
                            delete_download(audio_path)
                    
                    except Exception as e:
                        if "exceeds maximum allowed size" in str(e):
                            st.error(f"{FILE_TOO_LARGE_MESSAGE} Try a shorter video.")
                        else:
                            st.error(f"Error processing YouTube video: {e}")
    
    else:  # Text Input
        transcript = st.text_area("Enter transcript text:", height=300)
        
        if transcript:
            st.session_state.transcript = transcript
            
            if st.button("Generate Structured Notes", key="process_text"):
                with st.spinner("Creating highly structured notes..."):
                    notes = process_transcript(transcript)
                    
                    if notes:
                        st.success("Notes generated successfully!")
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Export as Markdown", key="md_text"):
                                export_notes(notes, "markdown")
                        with col2:
                            if st.button("Export as PDF", key="pdf_text"):
                                export_notes(notes, "pdf")

if __name__ == "__main__":
    main()
