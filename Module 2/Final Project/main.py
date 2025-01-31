import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from bs4 import BeautifulSoup
import librosa
import numpy as np
import cv2
import os
import requests
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from urllib.parse import urlparse
import validators

from model import ColorPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MoodoDataset(Dataset):
    def __init__(self, xml_file, audio_dir, feature_length=128):
        self.audio_dir = audio_dir
        self.feature_length = feature_length
        self.data = self._parse_dataset(xml_file)
        self.scaler = StandardScaler()

    def _parse_dataset(self, xml_file):
        with open(xml_file, 'r') as file:
            soup = BeautifulSoup(file, 'xml')
        songs = soup.find_all('song')
        data = []
        for song in songs:
            song_id = song.find('songid').text
            induced = [float(x) for x in song.find('induced').text.split(',')]
            perceived = [float(x) for x in song.find('perceived').text.split(',')]
            song_color = [float(x) for x in song.find('songcolor').text.split(',')]
            data.append((song_id, induced, perceived, song_color))
        return data

    def _extract_audio_features(self, audio_file):
        y, sr = librosa.load(audio_file, sr=22050)
        features = []
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroid))
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        
        if len(features) < self.feature_length: # zero padding
            features.extend([0] * (self.feature_length - len(features)))
        else:
            features = features[:self.feature_length]
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        song_id, induced, perceived, song_color = self.data[idx]
        audio_file = os.path.join(self.audio_dir, f"{song_id}.mp3")
        features = self._extract_audio_features(audio_file)
        
        features = self.scaler.fit_transform(features.reshape(1, -1)).flatten() # normalizing
        features = torch.tensor(features, dtype=torch.float32) 
        song_color = torch.tensor(song_color, dtype=torch.float32)
        return features, song_color


def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')


def generate_video(predicted_colors, output_file='output.mp4', frame_size=(640, 480), fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    for color in predicted_colors:
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        frame[:] = np.array(color) * 255  # convert [0, 1] to [0, 255]
        out.write(frame)
    out.release()


def download_audio_from_link(url, output_file='temp_audio.mp3'):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded audio to {output_file}")
        return output_file
    except Exception as e:
        print(f"Failed to download audio: {e}")
        return None


def predict_colors_for_audio(model, audio_file, feature_length=128):
    
    y, sr = librosa.load(audio_file, sr=22050)
    features = []
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(tempo)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spectral_centroid))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    if len(features) < feature_length:
        features.extend([0] * (feature_length - len(features)))
    else:
        features = features[:feature_length]
    
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    features = torch.tensor(features, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        predicted_colors = model(features).cpu().numpy()
    return predicted_colors


if __name__ == "__main__":
    dataset = MoodoDataset(xml_file='../../../moodo/dataset.xml', audio_dir='../../../moodo/audio')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ColorPredictor(input_size=dataset.feature_length)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dataloader, criterion, optimizer, num_epochs=20)

    
    input_source = input("Enter a link to an audio file or a local file path: ").strip()
    if validators.url(input_source):  # links
        audio_file = download_audio_from_link(input_source)
        if not audio_file:
            print("Failed to process the link. Exiting.")
            exit()
    else:  # local files
        if os.path.exists(input_source):
            audio_file = input_source
        else:
            print("File not found. Exiting.")
            exit()

    predicted_colors = predict_colors_for_audio(model, audio_file)
    generate_video(predicted_colors, output_file='generated_video.mp4')
    print("Video generated successfully!")