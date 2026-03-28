"""
Extract acoustic + electric guitar directly from a full music mix.
Uses the model trained by train_guitar_extractor.py.

Usage:
    from guitar_extractor_inference import extract_guitars
    paths = extract_guitars("/path/to/song.wav", "/output/dir")
    # {"acoustic": "/.../acoustic_guitar.wav",
    #  "electric": "/.../electric_guitar.wav",
    #  "ac_present": True, "el_present": True}
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "guitar_extractor.pt"

_DEFAULT_CFG = dict(sr=22050, n_fft=2048, hop=512, n_mels=128, t_frames=256, clip_sec=6.0)

_model  = None
_cfg    = None
_device = None


# ---------------------------------------------------------------------------
# Model (must match train_guitar_extractor.py exactly)
# ---------------------------------------------------------------------------

def _build_model():
    import torch.nn as nn
    import torch.nn.functional as F

    class _SE(nn.Module):
        def __init__(self, ch, r=16):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(ch, ch//r, bias=False), nn.ReLU(inplace=True),
                nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())
        def forward(self, x):
            return x * self.fc(self.pool(x).flatten(1)).view(x.shape[0], x.shape[1], 1, 1)

    class _Enc(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
            self.pool = nn.MaxPool2d(2, 2)
        def forward(self, x):
            s = self.conv(x); return self.pool(s), s

    class _Dec(nn.Module):
        def __init__(self, i, sk, o):
            super().__init__()
            self.up = nn.ConvTranspose2d(i, o, 2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(o+sk, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        def forward(self, x, s):
            x = self.up(x)
            if x.shape != s.shape: x = F.interpolate(x, size=s.shape[2:])
            return self.conv(torch.cat([x, s], 1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1=_Enc(1,32); self.enc2=_Enc(32,64)
            self.enc3=_Enc(64,128); self.enc4=_Enc(128,256)
            self.bot=nn.Sequential(
                nn.Conv2d(256,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True),
                _SE(512))
            self.dec4=_Dec(512,256,256); self.dec3=_Dec(256,128,128)
            self.dec2=_Dec(128,64,64);   self.dec1=_Dec(64,32,32)
            self.head=nn.Sequential(nn.Conv2d(32,2,1),nn.Sigmoid())
        def forward(self, x):
            x1,s1=self.enc1(x); x2,s2=self.enc2(x1)
            x3,s3=self.enc3(x2); x4,s4=self.enc4(x3)
            b=self.bot(x4)
            return self.head(self.dec1(self.dec2(self.dec3(self.dec4(b,s4),s3),s2),s1))

    return Net()


def _load_model() -> bool:
    global _model, _cfg, _device
    if _model is not None:
        return True
    if not MODEL_PATH.exists():
        logger.warning("guitar_extractor.pt missing")
        return False
    try:
        ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _cfg    = ck.get("config", _DEFAULT_CFG)
        _model  = _build_model()
        _model.load_state_dict(ck["model_state_dict"])
        _model.eval()
        _device = (torch.device("mps")  if torch.backends.mps.is_available() else
                   torch.device("cuda") if torch.cuda.is_available() else
                   torch.device("cpu"))
        _model  = _model.to(_device)
        logger.info("guitar_extractor loaded  val_loss=%.4f  device=%s",
                    ck.get("val_loss", 0), _device)
        return True
    except Exception as e:
        logger.warning("guitar_extractor load failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Presence threshold: if average mask energy below this, treat as absent
PRESENCE_THRESH = 0.08


def _run(y: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns: ac_audio, el_audio, ac_energy, el_energy
    """
    import librosa
    from scipy.ndimage import uniform_filter1d

    sr     = cfg["sr"]
    n_fft  = cfg["n_fft"]
    hop    = cfg["hop"]
    n_mels = cfg["n_mels"]
    T      = cfg["t_frames"]
    clip_n = int(cfg["clip_sec"] * sr)
    stride = int(clip_n * 0.75)

    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    fb_sum = mel_fb.sum(axis=0, keepdims=True) + 1e-8

    total  = len(y)
    ac_buf = np.zeros(total, dtype=np.float32)
    el_buf = np.zeros(total, dtype=np.float32)
    w_buf  = np.zeros(total, dtype=np.float32)
    ac_energies, el_energies = [], []

    fade_len = min(clip_n // 4, sr // 2)
    fade_i   = np.hanning(fade_len * 2)[:fade_len].astype(np.float32)
    fade_o   = np.hanning(fade_len * 2)[fade_len:].astype(np.float32)

    starts = list(range(0, total - clip_n + 1, stride)) or [0]

    for start in starts:
        chunk = y[start: start + clip_n]
        if len(chunk) < clip_n:
            chunk = np.pad(chunk, (0, clip_n - len(chunk)))

        D     = librosa.stft(chunk, n_fft=n_fft, hop_length=hop)
        mag   = np.abs(D)
        phase = np.angle(D)

        power    = mag ** 2
        mel_pow  = mel_fb @ power
        log_mel  = librosa.power_to_db(mel_pow, ref=mel_pow.max() + 1e-8)
        m, s     = log_mel.mean(), log_mel.std() + 1e-8
        norm_mel = ((log_mel - m) / s).astype(np.float32)

        t_act = norm_mel.shape[1]
        inp   = norm_mel[:, :T] if t_act >= T else np.pad(norm_mel, ((0,0),(0, T - t_act)))

        with torch.no_grad():
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_device)
            masks = _model(x).squeeze(0).cpu().numpy()   # (2, n_mels, T)

        t_use  = min(t_act, T)
        ac_mel = masks[0, :, :t_use]
        el_mel = masks[1, :, :t_use]

        ac_energies.append(float(ac_mel.mean()))
        el_energies.append(float(el_mel.mean()))

        # Smooth in time
        ac_mel = uniform_filter1d(ac_mel.astype(np.float32), size=7, axis=1)
        el_mel = uniform_filter1d(el_mel.astype(np.float32), size=7, axis=1)

        # Back-project to STFT bins
        ac_stft = np.clip((mel_fb.T @ ac_mel) / fb_sum.T, 0.0, 1.0)
        el_stft = np.clip((mel_fb.T @ el_mel) / fb_sum.T, 0.0, 1.0)

        # Spectral noise gate
        ac_stft = np.where(ac_stft > 0.12, ac_stft, 0.0)
        el_stft = np.where(el_stft > 0.12, el_stft, 0.0)

        # Normalise so masks don't exceed 1 together
        tot = ac_stft + el_stft + 1e-8
        ac_stft = ac_stft / np.maximum(tot, 1.0)
        el_stft = el_stft / np.maximum(tot, 1.0)

        sl   = mag[:, :t_use] * np.exp(1j * phase[:, :t_use])
        ac_a = librosa.istft(sl * ac_stft, hop_length=hop, length=clip_n)
        el_a = librosa.istft(sl * el_stft, hop_length=hop, length=clip_n)

        end = min(start + clip_n, total)
        seg = end - start
        w = np.ones(seg, dtype=np.float32)
        w[:min(fade_len, seg)]    *= fade_i[:min(fade_len, seg)]
        w[max(0, seg-fade_len):]  *= fade_o[max(0, fade_len-(seg-max(0,seg-fade_len))):]

        ac_buf[start:end] += ac_a[:seg] * w
        el_buf[start:end] += el_a[:seg] * w
        w_buf [start:end] += w

    ws = np.where(w_buf > 1e-6, w_buf, 1.0)
    ac_out = ac_buf / ws
    el_out = el_buf / ws

    # RMS-match to input
    in_rms = float(np.sqrt(np.mean(y**2))) + 1e-8
    for sig in [ac_out, el_out]:
        rms = float(np.sqrt(np.mean(sig**2))) + 1e-8
        sig *= (in_rms / rms) * 0.85
        np.clip(sig, -0.99, 0.99, out=sig)

    ac_en = float(np.mean(ac_energies))
    el_en = float(np.mean(el_energies))
    return ac_out, el_out, ac_en, el_en


def extract_guitars(wav_path: str, output_dir: str) -> dict | None:
    """
    Extract acoustic + electric guitar directly from original audio.

    Returns:
        {
          "acoustic": "/path/acoustic_guitar.wav",   # only if ac_present
          "electric": "/path/electric_guitar.wav",   # only if el_present
          "ac_present": bool,
          "el_present": bool,
        }
        or None if model unavailable.
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

        ac, el, ac_en, el_en = _run(y, cfg)

        logger.info("Guitar extractor: ac_energy=%.3f  el_energy=%.3f", ac_en, el_en)

        result: dict = {"ac_present": False, "el_present": False}

        if ac_en >= PRESENCE_THRESH:
            ac_path = str(out / "acoustic_guitar.wav")
            sf.write(ac_path, ac, cfg["sr"], subtype="PCM_24")
            result["acoustic"]   = ac_path
            result["ac_present"] = True

        if el_en >= PRESENCE_THRESH:
            el_path = str(out / "electric_guitar.wav")
            sf.write(el_path, el, cfg["sr"], subtype="PCM_24")
            result["electric"]   = el_path
            result["el_present"] = True

        logger.info("Guitar extraction done → ac=%s  el=%s", result.get("ac_present"), result.get("el_present"))
        return result

    except Exception as e:
        logger.warning("Guitar extraction failed: %s", e)
        return None
