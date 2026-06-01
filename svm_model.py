import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
import torch.nn as nn
from pathlib import Path
import pandas as pd
import librosa
import joblib

# Import your CNN model
from model import AudioCNN


# ✅ Custom Dataset (uses librosa instead of torchaudio)
class ESC50DatasetSVM(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

        # Train/Test split
        if split == "train":
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        # 🔥 Use librosa (no FFmpeg issue)
        audio, sr = librosa.load(audio_path, sr=22050)
        waveform = torch.tensor(audio).unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']


# ✅ SAME transform as CNN
transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=11025
    ),
    T.AmplitudeToDB()
)

# ✅ Load Train Dataset
train_dataset = ESC50DatasetSVM(
    data_dir=Path("ESC-50"),
    metadata_file=Path("ESC-50/meta/esc50.csv"),
    split="train",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# ✅ Load Test Dataset (fold 5)
test_dataset = ESC50DatasetSVM(
    data_dir=Path("ESC-50"),
    metadata_file=Path("ESC-50/meta/esc50.csv"),
    split="test",
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load trained CNN
checkpoint = torch.load("best_model.pth", map_location=device)

model = AudioCNN(num_classes=len(checkpoint['classes']))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 🔥 Remove classifier → use as feature extractor
model.fc = torch.nn.Identity()


# ===============================
# 🔥 FEATURE EXTRACTION FUNCTION
# ===============================
def extract_features(dataloader):
    features = []
    labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            output = model(data)   # shape: (batch_size, 512)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())

    X = np.concatenate(features)
    y = np.concatenate(labels)
    return X, y


print("Extracting TRAIN features...")
X_train, y_train = extract_features(train_loader)

print("Extracting TEST features...")
X_test, y_test = extract_features(test_loader)

print("Feature shape:", X_train.shape)


# ===============================
# 🚀 TRAIN SVM
# ===============================
print("Training SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train, y_train)


# ===============================
# 📊 EVALUATION
# ===============================
y_pred_train = svm.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)

y_pred_test = svm.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"SVM Training Accuracy: {train_acc:.4f}")
print(f"SVM Test Accuracy: {test_acc:.4f}")


# ===============================
# 💾 SAVE MODEL
# ===============================
joblib.dump(svm, "svm_model.pkl")
print("SVM model saved as svm_model.pkl")