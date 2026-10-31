import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

class DAICDepressionDataset(Dataset):
    def __init__(self, metadata_path="data_daic/metadata.csv", split="train", sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.split = split
        
        df = pd.read_csv(metadata_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.classes = ["Normal", "Depressed"]
        
        # Spectrogram transformations
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        self.amp_to_db = T.AmplitudeToDB()

        # SpecAugment transforms (Applied during training)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
        self.time_mask = T.TimeMasking(time_mask_param=30)
        
        # Pitch-shifting transform to break pitch/gender frequency memorization
        self.pitch_shift = T.PitchShift(sample_rate=self.sample_rate, n_steps=2)

        # Standard ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["filepath"]
        label = int(row["label"])
        pid = int(row["participant_id"])
        gender = str(row["gender"])

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or clip to exact target length (3.0 seconds)
        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        # Explicitly disable gradients during dataset augmentation pipeline
        with torch.no_grad():
            if self.split == "train":
                if torch.rand(1).item() < 0.35:
                    try:
                        waveform = self.pitch_shift(waveform)
                    except Exception:
                        pass
                
            spec = self.mel_spectrogram(waveform)
            spec = self.amp_to_db(spec)
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)

            if self.split == "train":
                spec = self.freq_mask(spec)
                spec = self.time_mask(spec)
            
            # Repeat single-channel spectrogram to 3 channels for ResNet18
            spec_3ch = spec.repeat(3, 1, 1)
            
            # Standardize features using ImageNet statistics
            spec_3ch = (spec_3ch - self.mean) / self.std

        return spec_3ch.detach(), label, pid, gender


def get_daic_dataloaders(metadata_path="data_daic/metadata.csv", batch_size=32):
    train_dataset = DAICDepressionDataset(metadata_path, split="train")
    val_dataset = DAICDepressionDataset(metadata_path, split="val")
    test_dataset = DAICDepressionDataset(metadata_path, split="test")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,
        persistent_workers=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset
