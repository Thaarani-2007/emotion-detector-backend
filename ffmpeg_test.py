import os
from pydub import AudioSegment
import pydub.utils

FFMPEG_PATH = r"C:\Users\Thaar\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\Users\Thaar\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe"

# Set both ffmpeg and ffprobe paths at the correct level
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH
pydub.utils.get_encoder_name = lambda: FFMPEG_PATH
pydub.utils.get_prober_name = lambda: FFPROBE_PATH

# Confirm paths
print("🔍 Checking ffmpeg path...")
if not os.path.isfile(FFMPEG_PATH):
    print(f"❌ ffmpeg not found at: {FFMPEG_PATH}")
else:
    print(f"✅ ffmpeg found at: {FFMPEG_PATH}")

print("🔍 Checking ffprobe path...")
if not os.path.isfile(FFPROBE_PATH):
    print(f"❌ ffprobe not found at: {FFPROBE_PATH}")
else:
    print(f"✅ ffprobe found at: {FFPROBE_PATH}")

# Input file
input_file = "recorded_audio.mp3"
output_file = "output.wav"

print(f"🔍 Checking input file: {input_file}")
if not os.path.isfile(input_file):
    print(f"❌ Input file not found at: {os.path.abspath(input_file)}")
    exit(1)

try:
    print("🔄 Loading audio...")
    audio = AudioSegment.from_file(input_file)
    print("💾 Exporting to WAV...")
    audio.export(output_file, format="wav")
    print(f"✅ Conversion successful. Output saved as: {output_file}")
except Exception as e:
    print("❌ Error during conversion:", e)
