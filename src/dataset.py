import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
import os

class HybridSpecDataset(Dataset):
    def __init__(self, df, audio_dir):
        self.df = df
        self.audio_dir = audio_dir
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        try:
            filename = self.df.iloc[idx]['filename']
            path = os.path.join(self.audio_dir, filename)
            y, sr = librosa.load(path, duration=30, sr=16000)
            target = 16000 * 30
            if len(y) < target: y = np.pad(y, (0, target-len(y)))
            else: y = y[:target]
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            img = librosa.power_to_db(mel, ref=np.max)
            img = (img - img.min()) / (img.max() - img.min()) 
            return torch.FloatTensor(img).unsqueeze(0)
        except: return torch.zeros(1, 64, 938)
