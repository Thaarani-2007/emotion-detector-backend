from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import librosa
import os
import tempfile
import traceback
from pydub import AudioSegment
import pydub.utils

# === Set ffmpeg & ffprobe paths ===
FFMPEG_PATH = r"C:\Users\Thaar\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\Users\Thaar\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe"

print("📌 Checking ffmpeg and ffprobe paths...")
if not os.path.isfile(FFMPEG_PATH):
    raise FileNotFoundError(f"❌ ffmpeg not found at: {FFMPEG_PATH}")
if not os.path.isfile(FFPROBE_PATH):
    raise FileNotFoundError(f"❌ ffprobe not found at: {FFPROBE_PATH}")

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH
pydub.utils.get_encoder_name = lambda: FFMPEG_PATH
pydub.utils.get_prober_name = lambda: FFPROBE_PATH
print("✅ ffmpeg and ffprobe paths set successfully")

# === Init Flask ===
app = Flask(__name__)
CORS(app)

# === Load model & encoder ===
try:
    model = joblib.load("model/xgb_ravdess_emotion_model.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    print("✅ Model and label encoder loaded successfully")
except Exception as e:
    print("❌ Failed to load model/encoder:", str(e))
    traceback.print_exc()
    model = None
    label_encoder = None

# === Convert non-WAV to WAV ===
def convert_to_wav(input_path):
    output_path = input_path + "_converted.wav"
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError("Uploaded file not found on disk")

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio.export(output_path, format="wav")
        print(f"✅ Converted to WAV: {output_path}")
        return output_path
    except Exception as e:
        print("❌ Conversion error:", e)
        traceback.print_exc()
        raise RuntimeError(f"Audio conversion failed: {e}")

# === Feature extraction ===
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=60)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_mel = librosa.power_to_db(mel)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)

        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(log_mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(zcr.T, axis=0)
        ])

        print(f"✅ Features extracted. Shape: {features.shape}")
        return features.reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")

# === Prediction endpoint ===
@app.route("/predict", methods=["POST"])
def predict():
    uploaded_path = None
    wav_path = None

    try:
        if model is None or label_encoder is None:
            return jsonify({"error": "Model or encoder not loaded"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "File name is empty"}), 400

        file_ext = os.path.splitext(file.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            uploaded_path = temp_file.name
            file.save(uploaded_path)

        if file_ext != ".wav":
            wav_path = convert_to_wav(uploaded_path)
        else:
            wav_path = uploaded_path

        features = extract_features(wav_path)
        proba = model.predict_proba(features)[0]
        predicted_index = np.argmax(proba)
        emotion = label_encoder.inverse_transform([predicted_index])[0]
        confidence = round(float(proba[predicted_index]) * 100, 2)

        print(f"🎯 Prediction: {emotion} ({confidence}%)")

        return jsonify({
            "emotion": emotion,
            "confidence": confidence
        })

    except Exception as e:
        err_trace = traceback.format_exc()
        print("❌ Exception during prediction:", str(e))
        print(err_trace)
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "trace": err_trace
        }), 500

    finally:
        for path in [uploaded_path, wav_path]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    print(f"🧹 Deleted temp file: {path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error: {cleanup_error}")

# === Run Flask ===
if __name__ == "__main__":
    print("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
