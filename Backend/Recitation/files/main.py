import gradio as gr
import os
from groq import Groq
import Levenshtein

# Initialize GROQ client
API_KEY = "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"
client = Groq(api_key=API_KEY)

def transcribe_audio_groq(audio_path):
    """
    Transcribes speech from an audio file using the GROQ Whisper model.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: The transcription of the speech in the audio file.
    """
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        return transcription  # Return the plain text response directly
    except Exception as e:
        return f"Error in transcription: {e}"

def levenshtein_similarity(transcription1, transcription2):
    """
    Calculate the Levenshtein similarity between two transcriptions.
    Args:
        transcription1 (str): The first transcription.
        transcription2 (str): The second transcription.
    Returns:
        float: A normalized similarity score between 0 and 1, where 1 indicates identical transcriptions.
    """
    distance = Levenshtein.distance(transcription1, transcription2)
    max_len = max(len(transcription1), len(transcription2))
    return 1 - distance / max_len if max_len > 0 else 1.0

def evaluate_audio_similarity(original_audio, user_audio):
    """
    Compares the similarity between the transcription of an original audio file and a user's audio file.
    Args:
        original_audio (str): Path to the original audio file.
        user_audio (str): Path to the user's audio file.
    Returns:
        tuple: Transcriptions and Levenshtein similarity score.
    """
    transcription_original = transcribe_audio_groq(original_audio)
    transcription_user = transcribe_audio_groq(user_audio)
    similarity_score_levenshtein = levenshtein_similarity(transcription_original, transcription_user)
    return transcription_original, transcription_user, similarity_score_levenshtein

def find_differences_groq(transcription_original, transcription_user):
    """
    Use the Groq API to identify differences between the original and user transcriptions.
    Args:
        transcription_original (str): The original transcription.
        transcription_user (str): The user's transcription.
    Returns:
        str: Explanation of the differences.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that find the mistake between two texts. Do not provide the long explaination Just provide me the mistakes in the output nothing else.",
        },
        {
            "role": "user",
            "content": (
                f"Original transcription: '{transcription_original}'\n"
                f"User transcription: '{transcription_user}'\n"
                "Explain the differences between these texts."
            ),
        },
    ]
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in generating explanation: {e}"

def perform_testing(original_audio, user_audio):
    if original_audio is not None and user_audio is not None:
        transcription_original, transcription_user, similarity_score = evaluate_audio_similarity(original_audio, user_audio)
        explanation = find_differences_groq(transcription_original, transcription_user)
        return (
            f"**Original Transcription:** {transcription_original}",
            f"**User Transcription:** {transcription_user}",
            f"**Levenshtein Similarity Score:** {similarity_score:.2f}",
            f"**Explanation of Differences:** {explanation}"
        )

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# Recitation and Mistake Finder")

    original_audio_upload = gr.Audio(label="Upload Original Audio", type="filepath")
    user_audio_upload = gr.Audio(label="Upload User Audio", type="filepath")
    upload_button = gr.Button("Perform Testing")
    output_original_transcription = gr.Markdown()
    output_user_transcription = gr.Markdown()
    output_similarity_score = gr.Markdown()
    output_explanation = gr.Markdown()

    upload_button.click(
        perform_testing,
        inputs=[original_audio_upload, user_audio_upload],
        outputs=[output_original_transcription, output_user_transcription, output_similarity_score, output_explanation]
    )

app.launch(share=True)
