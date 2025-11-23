import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
import gc
import os

# Ground truth transcripts - stored statically
GROUND_TRUTH = {
    "segment_002.wav": "TOWER HELLO EMIRATES FIVE TWO ONE",
    "segment_003.wav": "EMIRATES FIVE TWO ONE DUBAI TOWER GOOD AFTERNOON ONE TWO LEFT CONTINUE APPROACH PLAN TO VACATE AT MIKE NINER",
    "segment_004.wav": "CONTINUE PLANNING MIKE NINER EMIRATES FIVE TWO ONE",
    "segment_005.wav": "CLEARED TO LAND RUNWAY ONE TWO LEFT THE WIND ONE ZERO ZERO DEGREES ONE ONE KNOTS",
    "segment_006.wav": "CLEARED TO LAND ONE TWO LEFT EMIRATES FIVE TWO ONE",
    "segment_007.wav": "TOWER HELLO EMIRATES FIVE SIX FIVE WITH YOU SIX MILES",
    "segment_008.wav": "FIVE SIX FIVE TOWER GOOD AFTERNOON ONE TWO LEFT CONTINUE APPROACH PLAN TO VACATE MIKE NINER",
    "segment_009.wav": "MIKE NINER FIVE SIX FIVE",
    "segment_010.wav": "FIVE TWO ONE CONTINUE STRAIGHT CLIMB TO FOUR ZERO ZERO ZERO FEET",
    "segment_011.wav": "STRAIGHT AHEAD TO FOUR ZERO ZERO ZERO EMIRATES FIVE TWO ONE",
    "segment_012.wav": "DID YOU SEE THAT",
    "segment_013.wav": "EMIRATES FIVE SIX FIVE GO AROUND I REPEAT GO AROUND CONTINUE STRAIGHT TO FOUR ZERO ZERO ZERO",
    "segment_014.wav": "GO AROUND EMIRATES FIVE SIX FIVE",
    "segment_015.wav": "OKAY YOU NEED TO CHECK THAT AT THE END OF ONE TWO LEFT",
    "segment_016.wav": "ALRIGHT GOT IT",
    "segment_017.wav": "EMIRATES FIVE TWO ONE CRASH VEHICLES PROCEEDING NOW TO YOUR SCENE",
    "segment_018.wav": "SKY DUBAI EIGHT FIVE SEVEN",
    "segment_019.wav": "OKAY HOLDING AT SKY DUBAI EIGHT FIVE SEVEN",
    "segment_020.wav": "EMIRATES FIVE SIX FIVE CONTINUE STRAIGHT AHEAD AND FOUR ZERO ZERO ZERO FEET",
    "segment_021.wav": "GOING STRAIGHT AHEAD PASSING ONE TWO CLIMBING FOUR ZERO ZERO ZERO FEET EMIRATES FIVE SIX FIVE",
    "segment_022.wav": "FIVE SIX FIVE CONTACT RADAR ON ONE TWO SIX TWO",
    "segment_023.wav": "ONE TWO SIX TWO EMIRATES FIVE SIX FIVE",
    "segment_024.wav": "EMIRATES FIVE TWO ONE DO YOU REQUIRE ANY FURTHER ASSISTANCE FROM TOWER SIR",
    "segment_025.wav": "EMIRATES FIVE TWO ONE THAT IS FINE GET OUT OF THERE",
    "segment_026.wav": "ALL STATIONS ON DELIVERY FREQUENCY I NEED RADIO SILENCE FOR NOW RADIO SILENCE FOR THE NEXT ONE ZERO MINUTES AND I LL CALL YOU BACK",
    "segment_027.wav": "FIRE TWO PROCEED TO THE RUNWAY ONE TWO LEFT",
    "segment_028.wav": "PROCEED RUNWAY ONE TWO LEFT PROCEED RUNWAY ONE TWO LEFT PROCEED TO THE AIRCRAFT",
    "segment_029.wav": "FIRE ONE PROCEED RUNWAY ONE TWO LEFT PROCEED TO THE AIRCRAFT FOR FIRE ONE PROCEED TO THE AIRCRAFT",
    "segment_030.wav": "DUBAI SKY DUBAI EIGHT FOUR EIGHT WE LL BE MAINTAINING RUNWAY HEADING AFTER OVERHEAD JUST FOR YOUR INFORMATION",
    "segment_031.wav": "ALL FIRE VEHICLES PROCEED TO THE AIRCRAFT PROCEED TO THE RUNWAY PROCEED TO THE AIRCRAFT",
    "segment_032.wav": "FIRE ONE SEVEN PROCEED TO THE AIRCRAFT ON ONE TWO LEFT",
    "segment_033.wav": "DUBAI SKY DUBAI EIGHT FOUR EIGHT SORRY FOR INSISTING BUT IF YOU TELL US WHAT THE PLAN IS WITH US SO WE MIGHT DECIDE TO DIVERT RIGHT AWAY TO AL MAKTOUM",
    "segment_034.wav": "ALL FIRE VEHICLES PROCEED TO THE AIRCRAFT I REPEAT PROCEED ON ALL TAXIWAYS TO THE AIRCRAFT",
    "segment_035.wav": "FIRE FIVE VIA KILO SIX CROSS RUNWAY ONE TWO RIGHT PROCEED TO THE AIRCRAFT ON ONE TWO LEFT",
    "segment_036.wav": "ALL THE VEHICLES PROCEED RUNWAY ONE TWO LEFT PROCEED TO THE AIRCRAFT RUNWAY IS CLOSED",
    "segment_037.wav": "DUBAI SKY DUBAI EIGHT FOUR EIGHT ON RUNWAY HEADING OUR HOLDING CAPABILITY NOW IS REDUCED TO SEVEN MINUTES PLEASE",
    "segment_038.wav": "FOUR SIX SEVEN DELAY NOT DETERMINED",
    "segment_039.wav": "DUBAI SKY DUBAI EIGHT FOUR EIGHT IF THERE S NO PLAN FOR US WE D LIKE TO DIVERT TO AL MAKTOUM AS WELL DUE TO FUEL",
    "segment_040.wav": "SKY DUBAI FOUR THREE SEVEN",
    "segment_041.wav": "GO FOUR THREE SEVEN",
    "segment_042.wav": "FOR INFORMATION WE D LIKE TO START ONE ENGINE",
    "segment_043.wav": "I WOULDN T RECOMMEND THAT SINCE DELAY IS NOT DETERMINED DO NOT THINK IT WILL GET RESOLVED I LL CALL YOU BACK",
    "segment_044.wav": "OKAY",
    "segment_045.wav": "DUBAI SYRIAN AIR FIVE ONE FIVE WE CAN HOLD FOR FOURTY MINUTES OTHERWISE PROCEEDING TO ALTERNATE",
    "segment_046.wav": "I LET YOU KNOW DELAY NOT DETERMINED INCIDENT ON THE AIRFIELD I LL CALL YOU BACK",
    "segment_047.wav": "OKAY WE CAN LAND AT RAS AL KHAIMAH SYRIAN AIR FIVE ONE FIVE",
    "segment_048.wav": "ZERO SIX ZERO ROGER HOLD YOUR POSITION UNDETERMINED DELAY WE HAD AN INCIDENT ON THE AIRFIELD AND AT THIS STAGE I HAVE NO ADDITIONAL INFORMATION BUT THE AIRPORT THE AIRFIELD IS TEMPORARILY CLOSED",
    "segment_049.wav": "WE CAN HOLD FOURTY MINUTES SYRIAN AIR FIVE ONE FIVE",
    "segment_050.wav": "FDBSEVEN THREE ONE",
    "segment_051.wav": "SKY DUBAI SEVEN THREE ONE",
    "segment_052.wav": "FDBSEVEN THREE ONE",
    "segment_053.wav": "NO NOBODY IS MOVING ANYWHERE AT THIS STAGE UNDETERMINED DELAY THE AIRFIELD IS UNDER CONTROL OF FIRE SERVICES",
    "segment_054.wav": "FDBSEVEN THREE ONE",
    "segment_055.wav": "OKAY CAN WE LAND AT SHARJAH AIRPORT PLEASE",
    "segment_056.wav": "ROGER LEFT HEADING ZERO THREE ZERO AND EXPECT SHARJAH SYRIAN AIR FIVE ONE FIVE",
    "segment_057.wav": "JORDANIAN SIX ONE ONE",
    "segment_058.wav": "JORDANIAN SIX ONE ONE",
    "segment_059.wav": "YES SIR ANY DELAY FOR PUSHBACK OR START UP",
    "segment_060.wav": "JORDANIAN SIX ONE ONE AFFIRMATIVE THERE IS UNDETERMINED DELAY AT THIS POINT FOR THE AIRFIELD I VE NO FURTHER INFORMATION GIVEN I LL CALL YOU BACK WHEN IT IS BACK POSSIBLE IT COULD BE A FEW MINUTES TO GET AN ANSWER",
    "segment_061.wav": "THANK YOU VERY MUCH WE ARE ON CFIVE FIVE RIGHT AND WE ARE READY FOR THE PUSHBACK",
    "segment_062.wav": "UNDERSTOOD",
}


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
PARAKEET_MREZZAT_MODEL_ID = "MrEzzat/parakeet_atc"

# Global variables for models
whisper_original_pipe = None
parakeet_model = None
parakeet_mrezzat_model = None


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
                parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=PARAKEET_MODEL_ID
                )
            except Exception:
                # 2. If that fails, try restoring from the specific .nemo file (common for HF Hub models)
                # You might need to download it first using huggingface_hub
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(
                    repo_id=PARAKEET_MODEL_ID, filename="model.nemo"
                )
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


def transcribe_parakeet_mrezzat(audio_path):
    """Transcribe audio using MrEzzat's Parakeet model"""
    global parakeet_mrezzat_model

    # Auto-load model on first use
    if parakeet_mrezzat_model is None:
        try:
            import nemo.collections.asr as nemo_asr
            from huggingface_hub import hf_hub_download

            # Download the .nemo file from HuggingFace
            nemo_file = hf_hub_download(
                repo_id=PARAKEET_MREZZAT_MODEL_ID, filename="parakeet_atc_uae.nemo"
            )

            # Load the model using restore_from
            parakeet_mrezzat_model = nemo_asr.models.ASRModel.restore_from(
                restore_path=nemo_file
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                parakeet_mrezzat_model = parakeet_mrezzat_model.to("cuda")

        except Exception as e:
            # Return the ACTUAL error message for debugging
            return f"❌ Critical Error Loading Model: {str(e)}"

    try:
        # NeMo transcription expects a list of paths
        result = parakeet_mrezzat_model.transcribe([audio_path])

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
with gr.Blocks(title="ATC ASR - Model Comparison") as demo:
    # Inject CSS using gr.HTML (compatible with all Gradio versions)
    gr.HTML(f"<style>{custom_css}</style>")

    gr.Markdown(
        """
        # 🎙️ Air Traffic Control ASR - Model Comparison

        Compare three state-of-the-art ASR models on air traffic control (ATC) communications:

        1. **Whisper Large v3 Original** - Baseline performance
        2. **Parakeet-TDT-0.6B-v3 Fine-tuned for ATC** - WER: 5.99%
        3. **Parakeet MrEzzat ATC** - Custom fine-tuned model for UAE ATC

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

        # Tab 3: Parakeet MrEzzat
        with gr.Tab("🟣 Parakeet MrEzzat (ATC)"):
            gr.Markdown(
                """
                ### Parakeet ATC - MrEzzat Custom Model
                - **Model**: MrEzzat/parakeet_atc
                - **Base**: NVIDIA Parakeet-TDT
                - **Optimized for**: ATC communications
                - **Framework**: NVIDIA NeMo
                - **Note**: Model loads automatically on first transcription
                """
            )

            parakeet_mrezzat_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            parakeet_mrezzat_transcribe_btn = gr.Button(
                "Transcribe", variant="primary", size="lg"
            )

            parakeet_mrezzat_ground_truth = gr.Textbox(
                label="📝 Reference Transcript (Ground Truth)",
                lines=3,
                interactive=False,
                placeholder="Select a sample audio to see the reference transcript...",
            )

            parakeet_mrezzat_output = gr.Textbox(
                label="🤖 Model Transcription",
                lines=8,
                interactive=False,
                placeholder="Transcription will appear here...",
            )

            parakeet_mrezzat_transcribe_btn.click(
                fn=transcribe_parakeet_mrezzat,
                inputs=parakeet_mrezzat_audio,
                outputs=parakeet_mrezzat_output,
            )

            # Auto-populate ground truth when audio changes
            parakeet_mrezzat_audio.change(
                fn=get_ground_truth,
                inputs=parakeet_mrezzat_audio,
                outputs=parakeet_mrezzat_ground_truth,
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
                inputs=parakeet_mrezzat_audio,
                label="Try Sample Audio Files",
            )

        # Tab 4: Compare All
        with gr.Tab("📊 Compare Models"):
            gr.Markdown(
                """
                ### Compare All Three Models Side-by-Side
                Upload an audio file and get transcriptions from all three models simultaneously.

                **Note**: Models load automatically on first use. Initial transcription may take longer while models are loading.
                """
            )

            compare_audio = gr.Audio(
                label="Upload Audio File", type="filepath", sources=["upload"]
            )

            compare_btn = gr.Button(
                "Transcribe with All Models", variant="primary", size="lg"
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

                with gr.Column():
                    gr.Markdown("### 🟣 Parakeet MrEzzat")
                    compare_parakeet_mrezzat_output = gr.Textbox(
                        label="Parakeet MrEzzat (ATC)",
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

                whisper_orig_result = transcribe_whisper_original(audio_path)
                parakeet_result = transcribe_parakeet(audio_path)
                parakeet_mrezzat_result = transcribe_parakeet_mrezzat(audio_path)

                return whisper_orig_result, parakeet_result, parakeet_mrezzat_result

            compare_btn.click(
                fn=transcribe_all,
                inputs=compare_audio,
                outputs=[
                    compare_whisper_orig_output,
                    compare_parakeet_output,
                    compare_parakeet_mrezzat_output,
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
