import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class DepressionAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=5):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration
        
        self.folder_mapping = {
            'normal1': 0,      # Male Normal
            'normal2': 0,      # Female Normal
            'depression1': 1,  # Male Depressed
            'depression2': 1   # Female Depressed
        }
        self.classes = ['Normal', 'Depressed']
        
        self.filepaths = []
        self.labels = []
        self.genders = []
        
        for folder_name, label_id in self.folder_mapping.items():
            folder_path = os.path.join(root_dir, folder_name)
            gender = 'male' if '1' in folder_name else 'female'
            if os.path.exists(folder_path):
                for audio_file in glob.glob(os.path.join(folder_path, "*.wav")):
                    self.filepaths.append(audio_file)
                    self.labels.append(label_id)
                    self.genders.append(gender)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        self.amp_to_db = T.AmplitudeToDB()

        # In-Memory Cache with Live Progress Tracking
        self.cached_specs = []
        total_files = len(self.filepaths)
        print(f"Pre-processing {total_files} audio files into RAM...")
        
        for idx, path in enumerate(self.filepaths):
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                waveform = T.Resample(sr, self.sample_rate)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if waveform.shape[1] < self.target_length:
                pad_amount = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            else:
                waveform = waveform[:, :self.target_length]
                
            spec = self.mel_spectrogram(waveform)
            spec = self.amp_to_db(spec)
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
            spec_3ch = spec.repeat(3, 1, 1)
            self.cached_specs.append(spec_3ch)
            
            # Print live update on the same terminal line
            print(f"\rCaching audio: {idx + 1}/{total_files}...", end="", flush=True)
            
        print("\nDone! All audio files cached in RAM.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.cached_specs[idx], self.labels[idx], self.genders[idx]

def get_dataloaders(data_dir, batch_size=16):
    dataset = DepressionAudioDataset(data_dir)
    labels = dataset.labels
    
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.30, stratify=labels, random_state=42
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset
