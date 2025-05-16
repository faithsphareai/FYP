import os
import tempfile
from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
import librosa
from audioread.exceptions import NoBackendError
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from librosa.sequence import dtw
from google import genai
from google.genai import types

app = FastAPI()

# Global variables to hold our loaded models/clients.
client = None
comparer = None

# ---------------------------
# DTW-based Comparison Class
# ---------------------------
class QuranRecitationComparer:
    def __init__(self, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-arabic", auth_token=None):
        """Initialize the Quran recitation comparer with a specific Wav2Vec2 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model and processor once during initialization.
        if auth_token:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, token=auth_token)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name, token=auth_token)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        # Cache for embeddings to avoid recomputation.
        self.embedding_cache = {}

    def load_audio(self, file_path, target_sr=16000, trim_silence=True, normalize=True):
        """Load and preprocess an audio file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        try:
            y, sr = librosa.load(file_path, sr=target_sr)
        except NoBackendError as e:
            raise RuntimeError(
                "Failed to load audio using librosa. Please ensure you have a valid audio backend installed (e.g., ffmpeg)."
            ) from e
        if normalize:
            y = librosa.util.normalize(y)
        if trim_silence:
            y, _ = librosa.effects.trim(y, top_db=30)
        return y

    def get_deep_embedding(self, audio, sr=16000):
        """Extract frame-wise deep embeddings using the pretrained model."""
        input_values = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embedding_seq = hidden_states.squeeze(0).cpu().numpy()
        return embedding_seq

    def compute_dtw_distance(self, features1, features2):
        """Compute the DTW distance between two sequences of features."""
        D, wp = dtw(X=features1, Y=features2, metric='euclidean')
        distance = D[-1, -1]
        normalized_distance = distance / len(wp)
        return normalized_distance

    def interpret_similarity(self, norm_distance):
        """Interpret the normalized distance value."""
        if norm_distance == 0:
            result = "The recitations are identical based on the deep embeddings."
            score = 100
        elif norm_distance < 1:
            result = "The recitations are extremely similar."
            score = 95
        elif norm_distance < 5:
            result = "The recitations are very similar with minor differences."
            score = 80
        elif norm_distance < 10:
            result = "The recitations show moderate similarity."
            score = 60
        elif norm_distance < 20:
            result = "The recitations show some noticeable differences."
            score = 40
        else:
            result = "The recitations are quite different."
            score = max(0, 100 - norm_distance)
        return result, score

    def get_embedding_for_file(self, file_path):
        """Get embedding for a file, using cache if available."""
        if file_path in self.embedding_cache:
            return self.embedding_cache[file_path]
        audio = self.load_audio(file_path)
        embedding = self.get_deep_embedding(audio)
        self.embedding_cache[file_path] = embedding
        return embedding

    def predict(self, file_path1, file_path2):
        """
        Predict the similarity between two audio files.
        Returns:
            float: Similarity score
            str: Interpretation of similarity
        """
        embedding1 = self.get_embedding_for_file(file_path1)
        embedding2 = self.get_embedding_for_file(file_path2)
        norm_distance = self.compute_dtw_distance(embedding1.T, embedding2.T)
        interpretation, similarity_score = self.interpret_similarity(norm_distance)
        return similarity_score, interpretation

    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self.embedding_cache = {}

# ---------------------------
# Application Startup
# ---------------------------
@app.on_event("startup")
async def startup_event():
    global client, comparer
    # Load the GenAI API key from environment variable.
    genai_api_key = os.getenv("GENAI_API_KEY")
    if not genai_api_key:
        raise EnvironmentError("GENAI_API_KEY environment variable not set")
    client = genai.Client(api_key=genai_api_key)

    # Retrieve HuggingFace auth token from environment variable (if needed).
    hf_auth_token = os.getenv("HF_AUTH_TOKEN")
    # Initialize the comparer instance once at startup.
    comparer = QuranRecitationComparer(auth_token=hf_auth_token)

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Audio Similarity API!",
        "usage": {
            "endpoints": {
                "gemini": {
                    "path": "/compare-audio",
                    "description": "POST two audio files (user recitation and professional qarri) for similarity analysis using Gemini."
                },
                "dtw": {
                    "path": "/compare-dtw",
                    "description": "POST two audio files (user recitation and professional qarri) for similarity analysis using deep embeddings and DTW."
                }
            }
        }
    }

@app.post("/compare-audio")
async def compare_audio(
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...)
):
    """
    Compare two audio files using the Gemini approach.
    The first audio is the user's recitation and the second is the professional qarri recitation.
    """
    # Read the uploaded audio files.
    audio1_bytes = await audio1.read()
    audio2_bytes = await audio2.read()

    # Create a refined prompt that clearly identifies the audio sources.
    prompt = (
        """Please analyze and compare the two provided audio clips.
The first audio is the user's recitation, and the second audio is the professional qarri recitation.
Evaluate their similarity on a scale from 0 to 1, where:
  - 1 indicates the user's recitation contains no mistakes compared to the professional version,
  - 0 indicates there are significant mistakes.
Provide your response with:
  1. A numerical similarity score on the first line.
  2. A single sentence that indicates whether the user's recitation is similar, moderately similar, or dissimilar to the professional qarri."""
    )

    # Generate the content using the Gemini model with the two audio inputs.
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            prompt,
            types.Part.from_bytes(
                data=audio1_bytes,
                mime_type=audio1.content_type,
            ),
            types.Part.from_bytes(
                data=audio2_bytes,
                mime_type=audio2.content_type,
            )
        ]
    )
    return {"result": response.text}

@app.post("/compare-dtw")
async def compare_dtw(
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...)
):
    """
    Compare two audio files using deep embeddings and DTW.
    The first audio is the user's recitation and the second is the professional qarri recitation.
    """
    # Save the uploaded files to temporary files so they can be processed by the comparer.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp1:
        tmp1.write(await audio1.read())
        tmp1_path = tmp1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
        tmp2.write(await audio2.read())
        tmp2_path = tmp2.name

    try:
        # Get similarity score and interpretation using DTW-based approach.
        similarity_score, interpretation = comparer.predict(tmp1_path, tmp2_path)
    finally:
        # Clean up temporary files.
        os.remove(tmp1_path)
        os.remove(tmp2_path)

    return {
        "similarity_score": similarity_score,
        "interpretation": interpretation
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
