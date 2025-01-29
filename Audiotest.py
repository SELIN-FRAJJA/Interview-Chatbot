import torch
import numpy as np
import librosa
from pydub import AudioSegment
import os

# Define the feature extraction function (same as used during training)
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Define the model architecture (must match the training architecture)
class AudioClassificationModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(64, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.output = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.softmax(self.output(x), dim=1)
        return x

# Initialize the model (match the input size and number of classes)
input_size = 40  # Number of features
num_classes = 2  # Replace with the actual number of classes
model = AudioClassificationModel(input_size, num_classes)

# Load the model weights (state_dict)
model_path = 'model/audio_classification_final.pth'  # Replace with your model file path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define class names manually (replace with your actual class names)
class_names = [
    "confident", "unconfident"
]

# Folder containing the audio files
audio_folder = 'uploads'  # Replace with the path to your folder containing audio files
combined_audio_path = 'uploads/combined_audio.wav'  # Path to save the combined audio file

# Combine all audio files in the folder
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
combined_audio = None

for audio_file in audio_files:
    file_path = os.path.join(audio_folder, audio_file)
    audio = AudioSegment.from_file(file_path)
    if combined_audio is None:
        combined_audio = audio
    else:
        combined_audio += audio

# Save the combined audio file
combined_audio.export(combined_audio_path, format="wav")

# Load and process the combined audio file
features = features_extractor(combined_audio_path)
features = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension

# Perform prediction
with torch.no_grad():
    outputs = model(features)
    predicted_class_index = torch.argmax(outputs, dim=1).item()

# Map the predicted index back to the class name
predicted_label = class_names[predicted_class_index]
print(f"Predicted Class: {predicted_label}")
