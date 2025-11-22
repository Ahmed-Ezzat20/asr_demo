import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
import gc
import os

# Model configurations
WHISPER_FINETUNED_MODEL_ID = "jacktol/whisper-large-v3-finetuned-for-ATC"
WHISPER_ORIGINAL_MODEL_ID = "openai/whisper-large-v3"
PARAKEET_MODEL_ID = "qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC"

# Global variables for models
whisper_finetuned_pipe = None
whisper_original_pipe = None
parakeet_model = None


def load_whisper_finetuned():
    """Load fine-tuned Whisper model for ATC"""
    global whisper_finetuned_pipe

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_FINETUNED_MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(WHISPER_FINETUNED_MODEL_ID)

        whisper_finetuned_pipe = pipeline(
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

        return "✅ Whisper Fine-tuned model loaded successfully!"
    except Exception as e:
        return f"❌ Error loading Whisper Fine-tuned model: {str(e)}"


def load_whisper_original():
    """Load original Whisper Large v3 model"""
    global whisper_original_pipe

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_ORIGINAL_MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
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

        return "✅ Whisper Original model loaded successfully!"
    except Exception as e:
        return f"❌ Error loading Whisper Original model: {str(e)}"


def load_parakeet_model():
    """Load Parakeet model for ASR using NeMo"""
    global parakeet_model

    try:
        # Import NeMo ASR
        import nemo.collections.asr as nemo_asr

        # Load the model from Hugging Face
        parakeet_model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL_ID)

        return "✅ Parakeet model loaded successfully!"
    except ImportError:
        return "⚠️ NeMo is not installed. Installing NeMo... Please wait and try loading again after installation completes."
    except Exception as e:
        return f"⚠️ Parakeet model loading failed: {str(e)}\n\nNote: This model requires NVIDIA NeMo. Install with: pip install nemo_toolkit[asr]"


def transcribe_whisper_finetuned(audio_path):
    """Transcribe audio using fine-tuned Whisper model"""
    if whisper_finetuned_pipe is None:
        return "❌ Whisper Fine-tuned model not loaded. Please load the model first."

    try:
        # Load and resample audio to 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)

        # Run inference
        result = whisper_finetuned_pipe(audio)

        return result["text"]
    except Exception as e:
        return f"❌ Error during Whisper Fine-tuned transcription: {str(e)}"


def transcribe_whisper_original(audio_path):
    """Transcribe audio using original Whisper model"""
    if whisper_original_pipe is None:
        return "❌ Whisper Original model not loaded. Please load the model first."

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
    if parakeet_model is None:
        return "❌ Parakeet model not loaded. Please load the model first."

    try:
        # NeMo models expect file paths directly
        result = parakeet_model.transcribe([audio_path])

        # Extract text from result
        if isinstance(result, list) and len(result) > 0:
            if hasattr(result[0], "text"):
                return result[0].text
            else:
                return str(result[0])
        else:
            return str(result)
    except Exception as e:
        return f"❌ Error during Parakeet transcription: {str(e)}"


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
with gr.Blocks(title="ATC ASR - Three Model Comparison", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎙️ Air Traffic Control ASR - Three Model Comparison
        
        Compare three state-of-the-art ASR models on air traffic control (ATC) communications:
        
        1. **Whisper Large v3 Fine-tuned for ATC** - WER: 6.5%
        2. **Whisper Large v3 Original** - Baseline performance
        3. **Parakeet-TDT-0.6B-v3 Fine-tuned for ATC** - WER: 5.99%
        
        Switch between tabs to test each model individually or compare results side-by-side.
        """
    )

    with gr.Tabs():
        # Tab 1: Whisper Fine-tuned
        with gr.Tab("🔵 Whisper Fine-tuned (ATC)"):
            gr.Markdown(
                """
                ### Whisper Large v3 Fine-tuned for ATC
                - **Model**: jacktol/whisper-large-v3-finetuned-for-ATC
                - **WER**: 6.5%
                - **Base**: OpenAI Whisper Large v3
                - **Optimized for**: ATC communications with aviation-specific vocabulary
                """
            )

            with gr.Row():
                whisper_ft_load_btn = gr.Button("Load Model", variant="primary")
                whisper_ft_status = gr.Textbox(label="Status", interactive=False)

            whisper_ft_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            whisper_ft_transcribe_btn = gr.Button(
                "Transcribe", variant="primary", size="lg"
            )

            whisper_ft_output = gr.Textbox(
                label="Transcription",
                lines=8,
                interactive=False,
                placeholder="Transcription will appear here...",
            )

            whisper_ft_load_btn.click(
                fn=load_whisper_finetuned, outputs=whisper_ft_status
            )

            whisper_ft_transcribe_btn.click(
                fn=transcribe_whisper_finetuned,
                inputs=whisper_ft_audio,
                outputs=whisper_ft_output,
            )

        # Tab 2: Whisper Original
        with gr.Tab("⚪ Whisper Original"):
            gr.Markdown(
                """
                ### Whisper Large v3 Original
                - **Model**: openai/whisper-large-v3
                - **WER**: Baseline (not fine-tuned for ATC)
                - **Base**: OpenAI Whisper Large v3
                - **Optimized for**: General speech recognition
                """
            )

            with gr.Row():
                whisper_orig_load_btn = gr.Button("Load Model", variant="primary")
                whisper_orig_status = gr.Textbox(label="Status", interactive=False)

            whisper_orig_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            whisper_orig_transcribe_btn = gr.Button(
                "Transcribe", variant="primary", size="lg"
            )

            whisper_orig_output = gr.Textbox(
                label="Transcription",
                lines=8,
                interactive=False,
                placeholder="Transcription will appear here...",
            )

            whisper_orig_load_btn.click(
                fn=load_whisper_original, outputs=whisper_orig_status
            )

            whisper_orig_transcribe_btn.click(
                fn=transcribe_whisper_original,
                inputs=whisper_orig_audio,
                outputs=whisper_orig_output,
            )

        # Tab 3: Parakeet
        with gr.Tab("🟢 Parakeet Fine-tuned (ATC)"):
            gr.Markdown(
                """
                ### Parakeet-TDT-0.6B-v3 Fine-tuned for ATC
                - **Model**: qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC
                - **WER**: 5.99% (Best performance)
                - **Base**: NVIDIA Parakeet-TDT-0.6B-v3
                - **Optimized for**: ATC communications with superior accuracy
                - **Framework**: NVIDIA NeMo
                """
            )

            with gr.Row():
                parakeet_load_btn = gr.Button("Load Model", variant="primary")
                parakeet_status = gr.Textbox(label="Status", interactive=False)

            parakeet_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            parakeet_transcribe_btn = gr.Button(
                "Transcribe", variant="primary", size="lg"
            )

            parakeet_output = gr.Textbox(
                label="Transcription",
                lines=8,
                interactive=False,
                placeholder="Transcription will appear here...",
            )

            parakeet_load_btn.click(fn=load_parakeet_model, outputs=parakeet_status)

            parakeet_transcribe_btn.click(
                fn=transcribe_parakeet, inputs=parakeet_audio, outputs=parakeet_output
            )

        # Tab 4: Compare All
        with gr.Tab("📊 Compare All Models"):
            gr.Markdown(
                """
                ### Compare All Three Models Side-by-Side
                Upload an audio file and get transcriptions from all three models simultaneously.
                """
            )

            gr.Markdown(
                "#### ⚠️ Note: All models must be loaded first in their respective tabs!"
            )

            compare_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            compare_btn = gr.Button(
                "Transcribe with All Models", variant="primary", size="lg"
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔵 Whisper Fine-tuned")
                    compare_whisper_ft_output = gr.Textbox(
                        label="Whisper Fine-tuned (ATC)",
                        lines=6,
                        interactive=False,
                        placeholder="Transcription will appear here...",
                    )

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
                """Transcribe with all three models"""
                if audio_path is None:
                    return (
                        "❌ Please upload an audio file",
                        "❌ Please upload an audio file",
                        "❌ Please upload an audio file",
                    )

                whisper_ft_result = transcribe_whisper_finetuned(audio_path)
                whisper_orig_result = transcribe_whisper_original(audio_path)
                parakeet_result = transcribe_parakeet(audio_path)

                return whisper_ft_result, whisper_orig_result, parakeet_result

            compare_btn.click(
                fn=transcribe_all,
                inputs=compare_audio,
                outputs=[
                    compare_whisper_ft_output,
                    compare_whisper_orig_output,
                    compare_parakeet_output,
                ],
            )

    gr.Markdown(
        """
        ---
        ### 📊 About the Models
        
        All models are tested on the [ATC-ASR Dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset).
        
        **Model Links:**
        - [Whisper Large v3 Fine-tuned for ATC](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC)
        - [Whisper Large v3 Original](https://huggingface.co/openai/whisper-large-v3)
        - [Parakeet-TDT-0.6B-v3 Fine-tuned for ATC](https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC)
        
        ### 💡 Usage Tips
        - Upload audio files in WAV, MP3, or other common formats
        - For best results, use 16kHz sample rate audio
        - Fine-tuned models are optimized for aviation-specific vocabulary
        - Compare the original Whisper with fine-tuned versions to see the improvement
        - Parakeet model requires NVIDIA NeMo toolkit
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
