import os
import whisper

# Load the Whisper medium model
model = whisper.load_model("medium")

# Define input and output directories
input_folder = "./uploads"  # Folder containing the audio files
output_folder = "./transcriptions"  # Folder to save the transcriptions

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of audio files in the input folder
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac"))]

# Transcribe each audio file and save the transcription
for audio_file in audio_files:
    input_path = os.path.join(input_folder, audio_file)
    output_file = os.path.splitext(audio_file)[0] + ".txt"
    output_path = os.path.join(output_folder, output_file)

    print(f"Transcribing {audio_file}...")
    
    # Transcribe the audio
    result = model.transcribe(input_path)

    # Save the transcription to a text file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Saved transcription to {output_path}")

print("All files have been transcribed.")
