"""
Train electric vs acoustic guitar classifier using the IRMAS dataset.

Usage:
    source ../.venv/bin/activate
    python train_guitar_classifier.py --data IRMAS-TrainingData

Downloads:
    https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip
    → unzip and place the folder next to this script as IRMAS-TrainingData/
      (needs: IRMAS-TrainingData/gac/ and IRMAS-TrainingData/gel/)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from guitar_classifier import GuitarCNN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 80.0
FMAX = 16000.0
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(CLIP_DURATION * SR)
TARGET_FRAMES = 128  # time frames in mel spec

EPOCHS = 40
BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
PATIENCE = 8

SAVE_PATH = Path(__file__).parent / "guitar_classifier.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _compute_features(wav_path: Path) -> np.ndarray:
    """
    Load WAV → mono → 2-channel feature map (2, 128, 128).
    ch0: log-mel spectrogram (fmax=16000)
    ch1: spectral contrast   (7 bands zoomed to 128 rows)
    """
    import librosa
    import scipy.ndimage

    y, sr = librosa.load(str(wav_path), sr=None, mono=False)
    if y.ndim == 2:
        y = y.mean(axis=0)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) > CLIP_SAMPLES:
        y = y[:CLIP_SAMPLES]
    elif len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))

    # ch0: log-mel
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]
    ch0 = log_mel.astype(np.float32)

    # ch1: spectral contrast (7, T) → zoom freq axis → (128, TARGET_FRAMES)
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6, fmin=200.0,
    )
    if contrast.shape[1] < TARGET_FRAMES:
        contrast = np.pad(contrast, ((0, 0), (0, TARGET_FRAMES - contrast.shape[1])))
    else:
        contrast = contrast[:, :TARGET_FRAMES]
    ch1 = scipy.ndimage.zoom(
        contrast, (N_MELS / contrast.shape[0], 1.0), order=1
    ).astype(np.float32)

    return np.stack([ch0, ch1], axis=0)   # (2, 128, 128)


class IRMASGuitarDataset(Dataset):
    """
    Labels: 0 = acoustic (gac/), 1 = electric (gel/)
    Caches mel spectrograms as .npy files next to each wav for fast re-loading.
    """

    def __init__(self, data_dir: Path, augment: bool = False):
        self.augment = augment
        self.samples: list[tuple[np.ndarray, int]] = []

        for label, folder in [(0, "gac"), (1, "gel")]:
            wav_files = sorted((data_dir / folder).glob("*.wav"))
            if not wav_files:
                raise FileNotFoundError(
                    f"No WAV files in {data_dir / folder}. "
                    "Check that IRMAS-TrainingData is placed correctly."
                )
            print(f"  {folder}/: {len(wav_files)} files (label={label})")
            for wav in wav_files:
                cache = wav.with_suffix(".v2.npy")
                if cache.exists():
                    mel = np.load(cache)
                else:
                    mel = _compute_features(wav)
                    np.save(cache, mel)
                self.samples.append((mel, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        mel, label = self.samples[idx]
        x = mel.copy()  # (2, 128, 128) — don't modify the cached array

        # Per-channel z-score normalization
        for c in range(x.shape[0]):
            m, s = x[c].mean(), x[c].std() + 1e-8
            x[c] = (x[c] - m) / s

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _load_bleed_pool(data_dir: Path) -> list[np.ndarray]:
    """Load & normalize features from non-guitar IRMAS folders as bleed sources."""
    import random as _random
    NON_GUITAR = ["cel", "cla", "flu", "org", "pia", "sax", "tru", "vio"]
    pool: list[np.ndarray] = []
    for folder in NON_GUITAR:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
        wavs = sorted(folder_path.glob("*.wav"))
        # Sample up to 150 clips per instrument to keep pool manageable
        for wav in _random.sample(wavs, min(150, len(wavs))):
            cache = wav.with_suffix(".bleed.npy")
            if cache.exists():
                feat = np.load(cache)
            else:
                feat = _compute_features(wav)
                np.save(cache, feat)
            # Normalize per channel
            normed = feat.copy()
            for c in range(normed.shape[0]):
                m, s = normed[c].mean(), normed[c].std() + 1e-8
                normed[c] = (normed[c] - m) / s
            pool.append(normed)
    print(f"  Bleed pool: {len(pool)} non-guitar clips from {NON_GUITAR}")
    return pool


def train(data_dir: Path):
    print(f"\nLoading IRMAS data from {data_dir} ...")
    full_dataset = IRMASGuitarDataset(data_dir, augment=False)
    print(f"  Total: {len(full_dataset)} samples")

    # Load non-guitar clips for bleed simulation
    bleed_pool = _load_bleed_pool(data_dir)

    n_val = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Enable augmentation for training subset
    class _AugDataset(Dataset):
        def __init__(self, subset, bleed_pool):
            self.subset = subset
            self.bleed_pool = bleed_pool
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            import random as _r
            feat_tensor, label_tensor = self.subset[idx]
            x = feat_tensor.numpy().copy()              # (2, 128, 128)
            # Time-axis roll — both channels together (axis=2 is time)
            if np.random.rand() < 0.5:
                x = np.roll(x, np.random.randint(-10, 11), axis=2)
            # Frequency masking — only mel channel (ch0)
            if np.random.rand() < 0.3:
                f0 = np.random.randint(0, N_MELS - 10)
                x[0, f0:f0 + np.random.randint(1, 11), :] = 0.0
            # Additive Gaussian noise — both channels
            if np.random.rand() < 0.3:
                x += (np.random.randn(*x.shape) * 0.02).astype(np.float32)
            # ── Instrument bleed simulation (key improvement) ──────────────
            # Mimics other-instrument leakage in Demucs-separated stems.
            # Mixes a random non-guitar feature at -20 to -6 dB.
            if self.bleed_pool and np.random.rand() < 0.6:
                bleed = _r.choice(self.bleed_pool).copy()
                # Random time shift so bleed isn't phase-aligned
                bleed = np.roll(bleed, np.random.randint(0, 128), axis=2)
                scale = 10 ** (np.random.uniform(-20, -6) / 20)
                x[0] = x[0] + bleed[0] * scale
                x[1] = x[1] + bleed[1] * scale * 0.5   # contrast channel less affected
            # Per-channel z-score normalization
            for c in range(x.shape[0]):
                m, s = x[c].mean(), x[c].std() + 1e-8
                x[c] = (x[c] - m) / s
            return torch.tensor(x, dtype=torch.float32), torch.tensor(float(label_tensor))

    train_loader = DataLoader(_AugDataset(train_ds), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    model = GuitarCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve = 0

    print(f"\nTraining {n_train} samples, validating {n_val} samples")
    print(f"Epochs={EPOCHS}, Batch={BATCH}, LR={LR}, Patience={PATIENCE}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(1)
                val_loss += criterion(logits, y).item() * len(x)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y).sum().item()
        val_loss /= n_val
        val_acc = correct / n_val

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}  ({elapsed:.1f}s)")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "sr": SR, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
                    "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
                    "clip_duration": CLIP_DURATION,
                    "n_channels": 2,
                },
                "val_accuracy": best_val_acc,
                "classes": ["acoustic", "electric"],
            }, SAVE_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {SAVE_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train guitar type classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "IRMAS-TrainingData",
        help="Path to IRMAS-TrainingData folder (default: ./IRMAS-TrainingData)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: data directory not found: {args.data}")
        print("\nDownload IRMAS-TrainingData:")
        print("  https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip")
        print("  Unzip and place as backend/IRMAS-TrainingData/")
        exit(1)

    train(args.data)
