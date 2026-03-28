"""
Guitar type classifier: electric vs acoustic.

Requires a trained model at guitar_classifier.pt (run train_guitar_classifier.py first).
If the .pt file is missing, classify_guitar() returns {"guitar_type": "unknown", ...}
without raising any exceptions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "guitar_classifier.pt"

# Must match training parameters (also stored in checkpoint config)
_DEFAULT_CONFIG = {
    "sr": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 80.0,
    "fmax": 16000.0,
    "clip_duration": 3.0,
}

# Soft-voting thresholds
_ELECTRIC_THRESH = 0.60
_ACOUSTIC_THRESH = 0.40
_MIXED_STD_THRESH = 0.30

# Lazy-loaded singletons
_model: GuitarCNN | None = None
_config: dict | None = None
_device: torch.device | None = None


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation: learn channel importance and re-weight."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x).flatten(1)
        w = self.fc(w).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class _ResBlock(nn.Module):
    """Residual block + SE attention."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = _SEBlock(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.se(self.conv(x)) + self.shortcut(x))


class GuitarCNN(nn.Module):
    """
    ResNet-style classifier with SE channel attention.
    Input:  (B, 2, 128, 128)  — ch0: log-mel, ch1: spectral contrast
    Output: (B, 1)            — raw logit (sigmoid → P(electric))

    Improvements over a plain CNN:
      - Residual connection: no vanishing gradients when deep
      - SE attention: picks channels useful for acoustic vs electric
      - Wide receptive field (7×7 stem): global spectral shape
    """

    def __init__(self):
        super().__init__()
        # Wide stem for global spectral context
        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Residual + SE stages
        self.stage1 = _ResBlock(64,  128, stride=2)
        self.stage2 = _ResBlock(128, 256, stride=2)
        self.stage3 = _ResBlock(256, 512, stride=2)
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model() -> bool:
    """Lazy-load the model. Returns True if model is ready, False if .pt missing."""
    global _model, _config, _device
    if _model is not None:
        return True
    if not MODEL_PATH.exists():
        return False
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        _config = checkpoint.get("config", _DEFAULT_CONFIG)
        _model = GuitarCNN()
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()

        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")

        _model = _model.to(_device)
        val_acc = checkpoint.get("val_accuracy", "?")
        logger.info("Guitar classifier loaded (val_acc=%.3f, device=%s)", val_acc, _device)
        return True
    except Exception as exc:
        logger.warning("Failed to load guitar classifier: %s", exc)
        return False


def _wav_to_features(audio: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Convert mono float32 audio → 2-channel feature map (2, 128, T).
    ch0: log-mel spectrogram (fmax=16000, captures electric high-freq presence)
    ch1: spectral contrast   (captures tonal/acoustic vs flat/distorted-electric)
    """
    import librosa
    import scipy.ndimage

    # ch0: log-mel
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg["sr"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    m, s = log_mel.mean(), log_mel.std() + 1e-8
    ch0 = ((log_mel - m) / s).astype(np.float32)

    # ch1: spectral contrast (7 bands, T frames) → zoom freq axis to 128
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=cfg["sr"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_bands=6,
        fmin=200.0,
    )
    ch1_raw = scipy.ndimage.zoom(
        contrast, (128 / contrast.shape[0], 1), order=1
    ).astype(np.float32)
    m1, s1 = ch1_raw.mean(), ch1_raw.std() + 1e-8
    ch1 = (ch1_raw - m1) / s1

    return np.stack([ch0, ch1], axis=0)   # (2, 128, T)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_guitar(wav_path: str) -> dict:
    """
    Classify the guitar stem as electric, acoustic, mixed, or unknown.

    Returns:
        {
            "guitar_type": "electric" | "acoustic" | "mixed" | "unknown",
            "confidence":  float (0-1),
            "n_chunks":    int,
        }
    """
    _FALLBACK = {"guitar_type": "unknown", "confidence": 0.0, "n_chunks": 0}

    if not _load_model():
        return _FALLBACK

    cfg = _config or _DEFAULT_CONFIG

    try:
        import librosa
        import torchaudio.functional as AF

        # Load audio
        y, sr = librosa.load(wav_path, sr=None, mono=False)
        # Convert to mono
        if y.ndim == 2:
            y = y.mean(axis=0)
        # Resample if needed
        if sr != cfg["sr"]:
            y_t = torch.from_numpy(y).unsqueeze(0)
            y_t = AF.resample(y_t, sr, cfg["sr"])
            y = y_t.squeeze(0).numpy()

        clip_len = int(cfg["clip_duration"] * cfg["sr"])
        stride = int(2.0 * cfg["sr"])  # 2-second stride (1-sec overlap)
        total = len(y)

        if total < clip_len:
            return _FALLBACK

        # Extract chunks
        specs = []
        for start in range(0, total - clip_len + 1, stride):
            chunk = y[start: start + clip_len]
            # Skip silent chunks
            if float(np.sqrt(np.mean(chunk ** 2))) < 0.01:
                continue
            feat = _wav_to_features(chunk, cfg)   # (2, 128, T)
            # Pad/crop time axis to exactly 128 frames
            if feat.shape[2] < 128:
                feat = np.pad(feat, ((0, 0), (0, 0), (0, 128 - feat.shape[2])))
            else:
                feat = feat[:, :, :128]
            specs.append(feat)

        if len(specs) < 3:
            return _FALLBACK

        # Batch inference — shape (N, 2, 128, 128), no unsqueeze needed
        batch = torch.tensor(np.array(specs), dtype=torch.float32)
        batch = batch.to(_device)
        with torch.no_grad():
            logits = _model(batch).squeeze(1)  # (N,)
        probs = torch.sigmoid(logits).cpu().numpy()

        mean_prob = float(probs.mean())
        std_prob = float(probs.std())

        if mean_prob >= _ELECTRIC_THRESH:
            return {"guitar_type": "electric", "confidence": round(mean_prob, 2), "n_chunks": len(specs)}
        elif mean_prob <= _ACOUSTIC_THRESH:
            return {"guitar_type": "acoustic", "confidence": round(1.0 - mean_prob, 2), "n_chunks": len(specs)}
        elif std_prob > _MIXED_STD_THRESH:
            return {"guitar_type": "mixed", "confidence": round(1.0 - std_prob, 2), "n_chunks": len(specs)}
        else:
            gt = "electric" if mean_prob > 0.5 else "acoustic"
            conf = round(abs(mean_prob - 0.5) * 2, 2)
            return {"guitar_type": gt, "confidence": conf, "n_chunks": len(specs)}

    except Exception as exc:
        logger.warning("Guitar classification failed for %s: %s", wav_path, exc)
        return _FALLBACK
