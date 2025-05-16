# Quran Recitation Comparison API

This API allows you to compare two Quran recitations and determine their similarity using deep learning embeddings from the Wav2Vec2 model.

## Features

- Compare two audio files of Quran recitations
- Get a similarity score (0-100) and interpretation
- Uses Wav2Vec2 embeddings and Dynamic Time Warping (DTW) for comparison

## API Endpoints

### POST /compare

Upload two audio files to compare:

```
curl -X POST "http://localhost:7860/compare" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file1=@path/to/first/audio.mp3" \
  -F "file2=@path/to/second/audio.mp3"
```

Response:
```json
{
  "similarity_score": 60.0,
  "interpretation": "The recitations show moderate similarity."
}
```

### GET /health

Check if the API is healthy and the model is loaded:

```
curl -X GET "http://localhost:7860/health"
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Setup

1. The API requires a Hugging Face token as an environment variable:
   ```
   HF_TOKEN=your_hugging_face_token
   ```

2. The model uses `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` by default.

3. Audio files should be in a format supported by librosa (mp3, wav, etc.).