import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram
from pydub import AudioSegment


class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = None  # Placeholder for dynamic initialization
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)

        # Dynamically initialize fc1 based on input
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

    @staticmethod
    def process_audio(audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            #print(f"Loaded audio file: {audio_path}, Shape: {waveform.shape}, Sample Rate: {sample_rate}")

            # Convert stereo to mono if required
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Create Mel spectrogram
            spectrogram = MelSpectrogram()(waveform)
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
            return spectrogram
        except Exception as e:
            print(f"Error processing audio file '{audio_path}': {e}")
            raise


# Function to combine all audio files in a folder
def combine_audios(folder_path, output_path):
    try:
        combined = AudioSegment.empty()

        # List all audio files in the folder
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
        if not audio_files:
            print("No audio files found in the folder.")
            return
        
        for audio_file in audio_files:
            audio_path = os.path.join(folder_path, audio_file)
            #print(f"Processing audio file: {audio_path}")
            audio = AudioSegment.from_file(audio_path)
            combined += audio  # Append audio
        
        # Export the combined audio
        combined.export(output_path, format="wav")
        #print(f"Combined audio saved at: {output_path}")
    except Exception as e:
        print(f"Error combining audio files: {e}")
        raise


# Function to load the model and predict confidence level
def predict_confidence(audio_path, model):
    try:
        spectrogram = AudioModel.process_audio(audio_path).to(device)
        #print(f"Spectrogram shape: {spectrogram.shape}")

        with torch.no_grad():
            output = model(spectrogram)
            prediction = output.item()
        
        # Convert prediction to percentage
        prediction_percentage = prediction * 100
        print(f"Prediction: {prediction_percentage:.2f}%")

        if prediction_percentage >= 50:
            print("Confidence Level: Confident")
        else:
            print("Confidence Level: Unconfident")
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


# Main script
if __name__ == "__main__":
    # Define paths
    folder_path = "uploads"  # Replace with your folder path
    output_path = "uploads/combined_audio.wav"  # Output file for combined audio

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")

    # Initialize model
    model = AudioModel().to(device)

    # Load model weights
    try:
        model_path = 'model/audio_model3.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        state_dict = torch.load(model_path, map_location=device)
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        #print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Combine audio files and predict
    try:
        combine_audios(folder_path, output_path)
        predict_confidence(output_path, model)
    except Exception as e:
        print(f"Error in main process: {e}")
