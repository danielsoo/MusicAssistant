"""
Split acoustic vs electric guitar from audio with a trained U-Net.

Train with: ``python scripts/train_separator.py`` (uses ``mixture.wav`` = full-band mix).
Inference expects the **same kind of input as training** — preferably the **original song mix**,
not the Demucs ``guitar`` stem (``main.py`` passes the uploaded file when ``guitar_separator.pt`` exists).

Usage:
    from guitar_separator_inference import separate_guitar
    paths = separate_guitar("/path/to/full_mix_or_guitar.wav", "/output/dir")
    # paths = {"acoustic": "/.../acoustic_guitar.wav",
    #          "electric": "/.../electric_guitar.wav"}
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "guitar_separator.pt"

_DEFAULT_CFG = dict(
    sr=44100, n_fft=2048, hop=512, n_mels=128,
    t_frames=512, clip_sec=6.0,
)

_model = None
_cfg   = None
_device = None


# --- Model definition (same as train_separator.py) ---------------------------

def _build_model():
    """Same architecture as GuitarSeparatorUNet in train_separator.py (layer names match)."""
    import torch.nn as nn
    import torch.nn.functional as F

    class _SEBlock(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid(),
            )
        def forward(self, x):
            w = self.pool(x).flatten(1)
            w = self.fc(w).view(x.shape[0], x.shape[1], 1, 1)
            return x * w

    class _EncoderBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
            self.pool = nn.MaxPool2d(2, 2)
        def forward(self, x):
            skip = self.conv(x); return self.pool(skip), skip

    class _DecoderBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x, skip):
            x = self.up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            return self.conv(torch.cat([x, skip], dim=1))

    class GuitarSeparatorUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = _EncoderBlock(1,   32)
            self.enc2 = _EncoderBlock(32,  64)
            self.enc3 = _EncoderBlock(64,  128)
            self.enc4 = _EncoderBlock(128, 256)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                _SEBlock(512),
            )
            self.dec4 = _DecoderBlock(512, 256, 256)
            self.dec3 = _DecoderBlock(256, 128, 128)
            self.dec2 = _DecoderBlock(128, 64,  64)
            self.dec1 = _DecoderBlock(64,  32,  32)
            self.head = nn.Sequential(nn.Conv2d(32, 2, 1), nn.Sigmoid())

        def forward(self, x):
            x1, s1 = self.enc1(x); x2, s2 = self.enc2(x1)
            x3, s3 = self.enc3(x2); x4, s4 = self.enc4(x3)
            b = self.bottleneck(x4)
            return self.head(self.dec1(self.dec2(self.dec3(self.dec4(b, s4), s3), s2), s1))

    return GuitarSeparatorUNet()


def _load_model() -> bool:
    global _model, _cfg, _device
    if _model is not None:
        return True
    if not MODEL_PATH.exists():
        logger.warning("guitar_separator.pt missing — skipping separation")
        return False
    try:
        ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _cfg = ck.get("config", _DEFAULT_CFG)
        _model = _build_model()
        _model.load_state_dict(ck["model_state_dict"])
        _model.eval()
        _device = (torch.device("mps")  if torch.backends.mps.is_available() else
                   torch.device("cuda") if torch.cuda.is_available() else
                   torch.device("cpu"))
        _model = _model.to(_device)
        logger.info("guitar_separator loaded (device=%s, val_loss=%.4f)",
                    _device, ck.get("val_loss", 0))
        return True
    except Exception as e:
        logger.warning("guitar_separator load failed: %s", e)
        return False


def _chunk_inference(y: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Audio array → separated (acoustic, electric) arrays.
    STFT masking preserves phase to minimize quality loss.
    """
    import librosa

    sr      = cfg["sr"]
    n_fft   = cfg["n_fft"]
    hop     = cfg["hop"]
    n_mels  = cfg["n_mels"]
    T       = cfg["t_frames"]
    clip_s  = cfg["clip_sec"]
    clip_n  = int(clip_s * sr)
    stride  = int(clip_s * sr * 0.75)   # 75% overlap → fewer boundary artifacts

    # Mel filterbank (back-project masks to STFT frequency axis)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (n_mels, F)
    fb_sum = mel_fb.sum(axis=0, keepdims=True) + 1e-8                 # (1, F)

    total   = len(y)
    ac_buf  = np.zeros(total, dtype=np.float32)
    el_buf  = np.zeros(total, dtype=np.float32)
    weight  = np.zeros(total, dtype=np.float32)

    # Hann window for overlap-add
    fade_len = min(clip_n // 4, sr // 2)
    fade_in  = np.hanning(fade_len * 2)[:fade_len].astype(np.float32)
    fade_out = np.hanning(fade_len * 2)[fade_len:].astype(np.float32)

    starts = list(range(0, total - clip_n + 1, stride))
    if not starts:
        starts = [0]

    for start in starts:
        chunk = y[start: start + clip_n]
        if len(chunk) < clip_n:
            chunk = np.pad(chunk, (0, clip_n - len(chunk)))

        # --- STFT ---
        D     = librosa.stft(chunk, n_fft=n_fft, hop_length=hop)   # (F, t)
        mag   = np.abs(D)
        phase = np.angle(D)

        # --- mel + normalize (same preprocessing as training) ---
        power    = mag ** 2
        mel_pow  = mel_fb @ power                                    # (n_mels, t)
        log_mel  = librosa.power_to_db(mel_pow, ref=mel_pow.max() + 1e-8)
        m, s     = log_mel.mean(), log_mel.std() + 1e-8
        norm_mel = ((log_mel - m) / s).astype(np.float32)

        # Align time axis
        t_actual = norm_mel.shape[1]
        if t_actual >= T:
            inp = norm_mel[:, :T]
        else:
            inp = np.pad(norm_mel, ((0, 0), (0, T - t_actual)))

        # --- Model forward ---
        with torch.no_grad():
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            x = x.to(_device)
            masks = _model(x).squeeze(0).cpu().numpy()  # (2, n_mels, T)

        # masks shape: (2, n_mels, T) — T=512 fixed
        # t_actual may exceed T; use min
        t_use = min(t_actual, T)
        ac_mask_mel = masks[0, :, :t_use]   # (n_mels, t_use)
        el_mask_mel = masks[1, :, :t_use]

        # Smooth masks along time axis to prevent sharp transitions → ticks
        from scipy.ndimage import uniform_filter1d
        ac_mask_mel = uniform_filter1d(ac_mask_mel.astype(np.float32), size=7, axis=1)
        el_mask_mel = uniform_filter1d(el_mask_mel.astype(np.float32), size=7, axis=1)

        # --- mel mask → back-project to STFT frequency axis ---
        ac_mask_stft = (mel_fb.T @ ac_mask_mel) / fb_sum.T  # (F, t_use)
        el_mask_stft = (mel_fb.T @ el_mask_mel) / fb_sum.T

        # Clamp masks and suppress weak values (spectral noise gate)
        ac_mask_stft = np.clip(ac_mask_stft, 0.0, 1.0)
        el_mask_stft = np.clip(el_mask_stft, 0.0, 1.0)
        ac_mask_stft = np.where(ac_mask_stft > 0.15, ac_mask_stft, 0.0)
        el_mask_stft = np.where(el_mask_stft > 0.15, el_mask_stft, 0.0)
        total_mask   = ac_mask_stft + el_mask_stft + 1e-8
        ac_mask_stft = ac_mask_stft / np.maximum(total_mask, 1.0)
        el_mask_stft = el_mask_stft / np.maximum(total_mask, 1.0)

        # STFT masking then ISTFT — use both masks directly (not residual)
        stft_slice = mag[:, :t_use] * np.exp(1j * phase[:, :t_use])
        ac_audio = librosa.istft(stft_slice * ac_mask_stft, hop_length=hop, length=clip_n)
        el_audio = librosa.istft(stft_slice * el_mask_stft, hop_length=hop, length=clip_n)

        # --- overlap-add with fade window ---
        end = min(start + clip_n, total)
        seg = end - start

        w = np.ones(seg, dtype=np.float32)
        w[:min(fade_len, seg)]   *= fade_in[:min(fade_len, seg)]
        w[max(0, seg-fade_len):] *= fade_out[max(0, fade_len - (seg - max(0, seg-fade_len))):]

        ac_buf[start:end] += ac_audio[:seg] * w
        el_buf[start:end] += el_audio[:seg] * w
        weight[start:end] += w

    # Weighted average
    w_safe = np.where(weight > 1e-6, weight, 1.0)
    ac_out = ac_buf / w_safe
    el_out = el_buf / w_safe

    # RMS-match each output to the input level so they sound balanced
    input_rms = float(np.sqrt(np.mean(y ** 2))) + 1e-8
    for sig in [ac_out, el_out]:
        rms = float(np.sqrt(np.mean(sig ** 2))) + 1e-8
        sig *= (input_rms / rms) * 0.85
        np.clip(sig, -0.99, 0.99, out=sig)

    return ac_out, el_out


def separate_guitar(wav_path: str, output_dir: str) -> dict[str, str] | None:
    """
    guitar.wav → acoustic_guitar.wav + electric_guitar.wav

    Returns:
        {"acoustic": "/path/acoustic_guitar.wav",
         "electric": "/path/electric_guitar.wav"}
        or None if model not available.
    """
    if not _load_model():
        return None

    import librosa
    import soundfile as sf

    cfg = _cfg or _DEFAULT_CFG
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        y, file_sr = librosa.load(wav_path, sr=None, mono=True)
        if file_sr != cfg["sr"]:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=cfg["sr"])

        ac, el = _chunk_inference(y, cfg)

        ac_path = str(out / "acoustic_guitar.wav")
        el_path = str(out / "electric_guitar.wav")
        sf.write(ac_path, ac, cfg["sr"], subtype="PCM_24")
        sf.write(el_path, el, cfg["sr"], subtype="PCM_24")

        logger.info("Guitar separation done: %s", output_dir)
        return {"acoustic": ac_path, "electric": el_path}

    except Exception as e:
        logger.warning("Guitar separation failed: %s", e)
        return None
