import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
import gc
import os
import csv

# Load ground truth transcripts from CSV
def load_ground_truth_transcripts():
    """Load reference transcripts from segments_transcripts.csv"""
    transcripts = {}
    csv_path = "segments_transcripts.csv"

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                segment_name = row['Segment Name']
                transcript = row['Transcript']
                transcripts[segment_name] = transcript
    except Exception as e:
        print(f"Warning: Could not load ground truth transcripts: {e}")

    return transcripts

# Load transcripts at startup
GROUND_TRUTH = load_ground_truth_transcripts()

def get_ground_truth(audio_path):
    """Get ground truth transcript for a given audio file path"""
    if audio_path is None:
        return ""

    # Extract filename from path
    filename = os.path.basename(audio_path)

    # Return the ground truth transcript if available
    return GROUND_TRUTH.get(filename, "Ground truth not available for this file")

# Model configurations
WHISPER_ORIGINAL_MODEL_ID = "openai/whisper-large-v3"
PARAKEET_MODEL_ID = "qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC"

# Global variables for models
whisper_original_pipe = None
parakeet_model = None


def transcribe_whisper_original(audio_path):
    """Transcribe audio using original Whisper model"""
    global whisper_original_pipe

    # Auto-load model on first use
    if whisper_original_pipe is None:
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float32  # Always use fp32 for full capabilities

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_ORIGINAL_MODEL_ID,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(WHISPER_ORIGINAL_MODEL_ID)

            whisper_original_pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=False,
                torch_dtype=torch_dtype,
                device=device,
            )
        except Exception as e:
            return f"❌ Error loading Whisper Original model: {str(e)}"

    try:
        # Load and resample audio to 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)

        # Run inference
        result = whisper_original_pipe(audio)

        return result["text"]
    except Exception as e:
        return f"❌ Error during Whisper Original transcription: {str(e)}"


def transcribe_parakeet(audio_path):
    """Transcribe audio using Parakeet model"""
    global parakeet_model

    # Auto-load model on first use
    if parakeet_model is None:
        try:
            import nemo.collections.asr as nemo_asr

            # 1. Try loading normally
            try:
                parakeet_model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL_ID)
            except Exception:
                # 2. If that fails, try restoring from the specific .nemo file (common for HF Hub models)
                # You might need to download it first using huggingface_hub
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=PARAKEET_MODEL_ID, filename="model.nemo")
                parakeet_model = nemo_asr.models.ASRModel.restore_from(model_path)

            if torch.cuda.is_available():
                parakeet_model = parakeet_model.cuda()

        except Exception as e:
            # Return the ACTUAL error message for debugging
            return f"❌ Critical Error Loading Model: {str(e)}"

    try:
        # NeMo transcription expects a list of paths
        result = parakeet_model.transcribe([audio_path])

        # Extract text safely
        if isinstance(result, list) and len(result) > 0:
            # Check if result is a string or object with 'text' attribute
            first_result = result[0]
            if isinstance(first_result, str):
                return first_result
            elif hasattr(first_result, "text"):
                return first_result.text
            else:
                return str(first_result)
        return str(result)

    except Exception as e:
        return f"❌ Transcription Failed: {str(e)}"


# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.model-info {
    background-color: #f0f8ff;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
"""

# Create Gradio interface with tabs
with gr.Blocks(title="ATC ASR - Model Comparison", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎙️ Air Traffic Control ASR - Model Comparison

        Compare two state-of-the-art ASR models on air traffic control (ATC) communications:

        1. **Whisper Large v3 Original** - Baseline performance
        2. **Parakeet-TDT-0.6B-v3 Fine-tuned for ATC** - WER: 5.99%

        Switch between tabs to test each model individually or compare results side-by-side.
        """
    )

    with gr.Tabs():
        # Tab 1: Whisper Original
        with gr.Tab("⚪ Whisper Original"):
            gr.Markdown(
                """
                ### Whisper Large v3 Original
                - **Model**: openai/whisper-large-v3
                - **WER**: Baseline (not fine-tuned for ATC)
                - **Base**: OpenAI Whisper Large v3
                - **Optimized for**: General speech recognition
                - **Note**: Model loads automatically on first transcription
                """
            )

            whisper_orig_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            whisper_orig_transcribe_btn = gr.Button(
                "Transcribe", variant="primary", size="lg"
            )

            whisper_orig_ground_truth = gr.Textbox(
                label="📝 Reference Transcript (Ground Truth)",
                lines=3,
                interactive=False,
                placeholder="Select a sample audio to see the reference transcript...",
            )

            whisper_orig_output = gr.Textbox(
                label="🤖 Model Transcription",
                lines=8,
                interactive=False,
                placeholder="Transcription will appear here...",
            )

            whisper_orig_transcribe_btn.click(
                fn=transcribe_whisper_original,
                inputs=whisper_orig_audio,
                outputs=whisper_orig_output,
            )

            # Auto-populate ground truth when audio changes
            whisper_orig_audio.change(
                fn=get_ground_truth,
                inputs=whisper_orig_audio,
                outputs=whisper_orig_ground_truth,
            )

            # Example audio samples
            gr.Examples(
                examples=[
                    ["samples/segment_002.wav"],
                    ["samples/segment_003.wav"],
                    ["samples/segment_006.wav"],
                    ["samples/segment_013.wav"],
                    ["samples/segment_017.wav"],
                    ["samples/segment_026.wav"],
                    ["samples/segment_033.wav"],
                    ["samples/segment_036.wav"],
                    ["samples/segment_045.wav"],
                    ["samples/segment_048.wav"],
                ],
                inputs=whisper_orig_audio,
                label="Try Sample Audio Files",
            )

        # Tab 2: Parakeet
        with gr.Tab("🟢 Parakeet Fine-tuned (ATC)"):
            gr.Markdown(
                """
                ### Parakeet-TDT-0.6B-v3 Fine-tuned for ATC
                - **Model**: qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC
                - **WER**: 5.99% (Best performance)
                - **Base**: NVIDIA Parakeet-TDT-0.6B-v3
                - **Optimized for**: ATC communications with superior accuracy
                - **Framework**: NVIDIA NeMo
                - **Note**: Model loads automatically on first transcription
                """
            )

            parakeet_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            parakeet_transcribe_btn = gr.Button(
                "Transcribe", variant="primary", size="lg"
            )

            parakeet_ground_truth = gr.Textbox(
                label="📝 Reference Transcript (Ground Truth)",
                lines=3,
                interactive=False,
                placeholder="Select a sample audio to see the reference transcript...",
            )

            parakeet_output = gr.Textbox(
                label="🤖 Model Transcription",
                lines=8,
                interactive=False,
                placeholder="Transcription will appear here...",
            )

            parakeet_transcribe_btn.click(
                fn=transcribe_parakeet, inputs=parakeet_audio, outputs=parakeet_output
            )

            # Auto-populate ground truth when audio changes
            parakeet_audio.change(
                fn=get_ground_truth,
                inputs=parakeet_audio,
                outputs=parakeet_ground_truth,
            )

            # Example audio samples
            gr.Examples(
                examples=[
                    ["samples/segment_002.wav"],
                    ["samples/segment_003.wav"],
                    ["samples/segment_006.wav"],
                    ["samples/segment_013.wav"],
                    ["samples/segment_017.wav"],
                    ["samples/segment_026.wav"],
                    ["samples/segment_033.wav"],
                    ["samples/segment_036.wav"],
                    ["samples/segment_045.wav"],
                    ["samples/segment_048.wav"],
                ],
                inputs=parakeet_audio,
                label="Try Sample Audio Files",
            )

        # Tab 3: Compare All
        with gr.Tab("📊 Compare Models"):
            gr.Markdown(
                """
                ### Compare Both Models Side-by-Side
                Upload an audio file and get transcriptions from both models simultaneously.

                **Note**: Models load automatically on first use. Initial transcription may take longer while models are loading.
                """
            )

            compare_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            compare_btn = gr.Button(
                "Transcribe with Both Models", variant="primary", size="lg"
            )

            compare_ground_truth = gr.Textbox(
                label="📝 Reference Transcript (Ground Truth)",
                lines=3,
                interactive=False,
                placeholder="Select a sample audio to see the reference transcript...",
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚪ Whisper Original")
                    compare_whisper_orig_output = gr.Textbox(
                        label="Whisper Original",
                        lines=6,
                        interactive=False,
                        placeholder="Transcription will appear here...",
                    )

                with gr.Column():
                    gr.Markdown("### 🟢 Parakeet Fine-tuned")
                    compare_parakeet_output = gr.Textbox(
                        label="Parakeet Fine-tuned (ATC)",
                        lines=6,
                        interactive=False,
                        placeholder="Transcription will appear here...",
                    )

            def transcribe_all(audio_path):
                """Transcribe with both models"""
                if audio_path is None:
                    return (
                        "❌ Please upload an audio file",
                        "❌ Please upload an audio file",
                    )

                whisper_orig_result = transcribe_whisper_original(audio_path)
                parakeet_result = transcribe_parakeet(audio_path)

                return whisper_orig_result, parakeet_result

            compare_btn.click(
                fn=transcribe_all,
                inputs=compare_audio,
                outputs=[
                    compare_whisper_orig_output,
                    compare_parakeet_output,
                ],
            )

            # Auto-populate ground truth when audio changes
            compare_audio.change(
                fn=get_ground_truth,
                inputs=compare_audio,
                outputs=compare_ground_truth,
            )

            # Example audio samples
            gr.Examples(
                examples=[
                    ["samples/segment_002.wav"],
                    ["samples/segment_003.wav"],
                    ["samples/segment_006.wav"],
                    ["samples/segment_013.wav"],
                    ["samples/segment_017.wav"],
                    ["samples/segment_026.wav"],
                    ["samples/segment_033.wav"],
                    ["samples/segment_036.wav"],
                    ["samples/segment_045.wav"],
                    ["samples/segment_048.wav"],
                ],
                inputs=compare_audio,
                label="Try Sample Audio Files",
            )

    gr.Markdown(
        """
        ---
        ### 📊 About the Models

        Models are tested on the [ATC-ASR Dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset).

        **Model Links:**
        - [Whisper Large v3 Original](https://huggingface.co/openai/whisper-large-v3)
        - [Parakeet-TDT-0.6B-v3 Fine-tuned for ATC](https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC)

        ### 💡 Usage Tips
        - Upload audio files in WAV, MP3, or other common formats
        - For best results, use 16kHz sample rate audio
        - Parakeet model is fine-tuned for aviation-specific vocabulary (WER: 5.99%)
        - Compare the original Whisper with the fine-tuned Parakeet to see the improvement
        - Parakeet model requires NVIDIA NeMo toolkit
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
