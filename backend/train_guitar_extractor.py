"""
Train guitar extractor: extracts acoustic + electric guitar directly from
a full music mix — bypasses Demucs guitar stem entirely.

Data needed:
  Acoustic: GuitarSet (30-second mono mic recordings)
    Download: https://zenodo.org/records/3371780/files/audio_mono-mic.zip
    Unzip as: backend/GuitarSet/audio_mono-mic/

  Electric:  IRMAS-TrainingData/gel/  (3-second clips — concatenated to ~30s)
  Background: IRMAS-TrainingData/ (cel, cla, flu, pia, sax, tru, vio, voi)

Usage:
    cd backend
    python train_guitar_extractor.py \\
        --acoustic GuitarSet/audio_mono-mic \\
        --irmas IRMAS-TrainingData

If --acoustic is omitted, falls back to IRMAS gac/ (shorter but still works).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SR          = 22050
N_FFT       = 2048
HOP         = 512
N_MELS      = 128
CLIP_SEC    = 6.0
CLIP_N      = int(CLIP_SEC * SR)
T_FRAMES    = 256     # mel time frames per chunk

EPOCHS      = 50
BATCH       = 16
LR          = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT   = 0.15
PATIENCE    = 10

BG_INSTRUMENTS = ["cel", "cla", "flu", "org", "pia", "sax", "tru", "vio", "voi"]

SAVE_PATH   = Path(__file__).parent / "guitar_extractor.pt"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_mono(path: Path, sr: int = SR) -> np.ndarray:
    import librosa
    y, file_sr = librosa.load(str(path), sr=None, mono=True)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def _norm(y: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]."""
    peak = np.abs(y).max()
    return y / (peak + 1e-8)


def _random_chunk(y: np.ndarray, n: int) -> np.ndarray:
    """Return a random n-sample chunk from y (with padding if short)."""
    if len(y) <= n:
        return np.pad(y, (0, n - len(y)))
    start = random.randint(0, len(y) - n)
    return y[start: start + n]


def _concat_to_length(clips: list[np.ndarray], n: int) -> np.ndarray:
    """Concatenate clips until we reach n samples."""
    buf = np.concatenate(clips)
    while len(buf) < n:
        buf = np.concatenate([buf] + clips)
    return _random_chunk(buf, n)


def _log_mel(y: np.ndarray) -> np.ndarray:
    """(CLIP_N,) → (N_MELS, T_FRAMES) log-mel, normalised."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    lm  = librosa.power_to_db(mel, ref=np.max)
    if lm.shape[1] < T_FRAMES:
        lm = np.pad(lm, ((0, 0), (0, T_FRAMES - lm.shape[1])))
    else:
        lm = lm[:, :T_FRAMES]
    m, s = lm.mean(), lm.std() + 1e-8
    return ((lm - m) / s).astype(np.float32)


def _stft_mag(y: np.ndarray):
    import librosa
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)   # (F, T)
    return np.abs(D).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GuitarExtractorDataset(Dataset):
    """
    Each sample:
      x  : log-mel of the synthetic mix  (1, N_MELS, T_FRAMES)
      y_ac: mel-domain magnitude of acoustic guitar (N_MELS, T_FRAMES)
      y_el: mel-domain magnitude of electric guitar (N_MELS, T_FRAMES)
      mix_mel_mag: mel magnitude of mix (for mask supervision)
    """

    def __init__(
        self,
        acoustic_wavs: list[Path],
        electric_wavs: list[Path],
        bg_wavs: list[Path],
        n_samples: int = 4000,
    ):
        self.ac_wavs  = acoustic_wavs
        self.el_wavs  = electric_wavs
        self.bg_wavs  = bg_wavs
        self.n        = n_samples
        self._ac_cache: dict[str, np.ndarray] = {}
        self._el_cache: dict[str, np.ndarray] = {}
        self._bg_cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return self.n

    def _get(self, cache, wavs, key):
        k = str(key)
        if k not in cache:
            cache[k] = _norm(_load_mono(key))
        return cache[k]

    def __getitem__(self, _idx):
        import librosa

        # ── acoustic clip ─────────────────────────────────────────────────
        ac_path = random.choice(self.ac_wavs)
        ac_full = self._get(self._ac_cache, self.ac_wavs, ac_path)
        ac      = _random_chunk(ac_full, CLIP_N)

        # ── electric clip (concat short IRMAS clips if needed) ───────────
        el_path = random.choice(self.el_wavs)
        el_full = self._get(self._el_cache, self.el_wavs, el_path)
        if len(el_full) < CLIP_N:
            # build a long-enough buffer from random gel clips
            pool = [self._get(self._el_cache, self.el_wavs, p)
                    for p in random.sample(self.el_wavs, min(20, len(self.el_wavs)))]
            el_full = _concat_to_length(pool, CLIP_N)
        el = _random_chunk(el_full, CLIP_N)

        # ── background (0–3 random instruments) ──────────────────────────
        bg = np.zeros(CLIP_N, dtype=np.float32)
        n_bg = random.randint(0, 3)
        for _ in range(n_bg):
            bg_path = random.choice(self.bg_wavs)
            bg_full = self._get(self._bg_cache, self.bg_wavs, bg_path)
            bg_chunk = _random_chunk(bg_full, CLIP_N)
            bg_gain  = random.uniform(0.2, 0.8)
            bg      += bg_chunk * bg_gain

        # ── gain randomisation ────────────────────────────────────────────
        ac_gain = random.uniform(0.5, 1.0)
        el_gain = random.uniform(0.5, 1.0)

        # Randomly drop one guitar type 20% of the time (train for absence)
        if random.random() < 0.20:
            ac_gain = 0.0
        if random.random() < 0.20:
            el_gain = 0.0

        mix = ac * ac_gain + el * el_gain + bg
        # Peak-normalise mix
        mix_peak = np.abs(mix).max()
        if mix_peak > 0:
            mix /= mix_peak

        # ── features ─────────────────────────────────────────────────────
        mix_lm  = _log_mel(mix)                     # (N_MELS, T_FRAMES)

        # Mel magnitude targets for mask training
        mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
        def _mel_mag(y, gain):
            if gain == 0.0:
                return np.zeros((N_MELS, T_FRAMES), dtype=np.float32)
            mag = _stft_mag(y * gain)           # (F, T)
            mm  = (mel_fb @ mag) ** 0.5          # power → amplitude in mel
            if mm.shape[1] < T_FRAMES:
                mm = np.pad(mm, ((0,0),(0, T_FRAMES - mm.shape[1])))
            else:
                mm = mm[:, :T_FRAMES]
            return mm.astype(np.float32)

        mix_mag = _mel_mag(mix, 1.0)             # (N_MELS, T_FRAMES)
        ac_mag  = _mel_mag(ac,  ac_gain)
        el_mag  = _mel_mag(el,  el_gain)

        # Ideal Ratio Masks
        denom  = mix_mag + 1e-8
        ac_irm = np.clip(ac_mag / denom, 0.0, 1.0)   # (N_MELS, T_FRAMES)
        el_irm = np.clip(el_mag / denom, 0.0, 1.0)

        x = torch.tensor(mix_lm,  dtype=torch.float32).unsqueeze(0)    # (1, M, T)
        y_ac = torch.tensor(ac_irm, dtype=torch.float32)                # (M, T)
        y_el = torch.tensor(el_irm, dtype=torch.float32)                # (M, T)

        return x, y_ac, y_el


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(self.pool(x).flatten(1)).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class _Enc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        s = self.conv(x); return self.pool(s), s


class _Dec(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        import torch.nn.functional as F
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        return self.conv(torch.cat([x, skip], dim=1))


class GuitarExtractorUNet(nn.Module):
    """
    Input : (B, 1, N_MELS, T_FRAMES) log-mel of full mix
    Output: (B, 2, N_MELS, T_FRAMES) soft masks [acoustic, electric]
    """
    def __init__(self):
        super().__init__()
        self.enc1 = _Enc(1,   32)
        self.enc2 = _Enc(32,  64)
        self.enc3 = _Enc(64,  128)
        self.enc4 = _Enc(128, 256)
        self.bot  = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            _SEBlock(512),
        )
        self.dec4 = _Dec(512, 256, 256)
        self.dec3 = _Dec(256, 128, 128)
        self.dec2 = _Dec(128, 64,  64)
        self.dec1 = _Dec(64,  32,  32)
        self.head = nn.Sequential(nn.Conv2d(32, 2, 1), nn.Sigmoid())

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.bot(x4)
        return self.head(self.dec1(self.dec2(self.dec3(self.dec4(b, s4), s3), s2), s1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _collect_wavs(folders: list[Path]) -> list[Path]:
    wavs = []
    for f in folders:
        if f.exists():
            wavs.extend(sorted(f.glob("**/*.wav")))
    return wavs


def train(acoustic_dir: Path | None, irmas_dir: Path):
    # ── collect files ─────────────────────────────────────────────────────
    print("\nCollecting audio files...")

    if acoustic_dir and acoustic_dir.exists():
        ac_wavs = sorted(acoustic_dir.glob("**/*.wav"))
        print(f"  Acoustic (GuitarSet): {len(ac_wavs)} files")
    else:
        ac_wavs = sorted((irmas_dir / "gac").glob("*.wav"))
        print(f"  Acoustic (IRMAS gac fallback): {len(ac_wavs)} files")

    el_wavs = sorted((irmas_dir / "gel").glob("*.wav"))
    print(f"  Electric (IRMAS gel): {len(el_wavs)} files")

    bg_wavs = _collect_wavs([irmas_dir / f for f in BG_INSTRUMENTS])
    print(f"  Background instruments: {len(bg_wavs)} files")

    if not ac_wavs or not el_wavs:
        print("ERROR: Need acoustic and electric WAV files.")
        return

    # ── dataset split ─────────────────────────────────────────────────────
    N_TRAIN = 5000
    N_VAL   = 600

    train_ds = GuitarExtractorDataset(ac_wavs, el_wavs, bg_wavs, n_samples=N_TRAIN)
    val_ds   = GuitarExtractorDataset(ac_wavs, el_wavs, bg_wavs, n_samples=N_VAL)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    # ── device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # ── model ─────────────────────────────────────────────────────────────
    model     = GuitarExtractorUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Mask supervision: MSE between predicted mask and IRM
    criterion = nn.MSELoss()

    best_val  = float("inf")
    no_improve = 0

    print(f"\nTraining {N_TRAIN} synthetic samples/epoch, validating {N_VAL}")
    print(f"Epochs={EPOCHS}, Batch={BATCH}, LR={LR}, Patience={PATIENCE}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss = 0.0
        for x, y_ac, y_el in train_loader:
            x    = x.to(device)
            y_ac = y_ac.to(device)
            y_el = y_el.to(device)

            optimizer.zero_grad()
            masks   = model(x)                           # (B, 2, M, T)
            pred_ac = masks[:, 0, :, :]                  # (B, M, T)
            pred_el = masks[:, 1, :, :]

            loss = criterion(pred_ac, y_ac) + criterion(pred_el, y_el)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(x)

        tr_loss /= N_TRAIN
        scheduler.step()

        # Validate
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for x, y_ac, y_el in val_loader:
                x    = x.to(device)
                y_ac = y_ac.to(device)
                y_el = y_el.to(device)
                masks   = model(x)
                pred_ac = masks[:, 0, :, :]
                pred_el = masks[:, 1, :, :]
                loss = criterion(pred_ac, y_ac) + criterion(pred_el, y_el)
                vl_loss += loss.item() * len(x)
        vl_loss /= N_VAL

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.4f}  val={vl_loss:.4f}  ({elapsed:.1f}s)")

        if vl_loss < best_val:
            best_val = vl_loss
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS,
                    "t_frames": T_FRAMES, "clip_sec": CLIP_SEC,
                },
                "val_loss": best_val,
            }, SAVE_PATH)
            print(f"  ✓ saved (val_loss={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch}")
                break

    print(f"\nBest val_loss: {best_val:.4f}")
    print(f"Model saved → {SAVE_PATH}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic", type=Path, default=None,
                        help="GuitarSet audio_mono-mic dir (optional, falls back to IRMAS gac)")
    parser.add_argument("--irmas", type=Path,
                        default=Path(__file__).parent / "IRMAS-TrainingData",
                        help="IRMAS-TrainingData dir")
    args = parser.parse_args()

    if not args.irmas.exists():
        print(f"ERROR: IRMAS dir not found: {args.irmas}")
        exit(1)

    train(args.acoustic, args.irmas)
