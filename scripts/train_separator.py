"""
Train acoustic / electric guitar separation model (v3) — **canonical** training for ``backend/guitar_separator.pt``.

Run: ``python scripts/train_separator.py`` or ``python backend/train_guitar_separator.py``

v3 changes vs v2:
  - Training input switches from "acoustic+electric sum" to full-instrument mix (mixture.wav)
    → mixture.wav from prepare_training_data.py (full-band mix with drums, bass, vocals)
    → Trains closer to real full-song conditions
  - Targets (acoustic.wav / electric.wav) use levels relative to mixture
    → Model learns correct guitar ratios inside the mix
  - Older pairs without mixture.wav fall back to on-the-fly ac+el sum

Unchanged from v2:
  1. STFT magnitude-domain loss (matches inference)
  2. Random crops per pair (N_CROPS)
  3. Early stop patience=15, max 100 epochs

Usage:
    source .venv/bin/activate
    python scripts/train_separator.py
"""

import random
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- Config -------------------------------------------------------------------

ROOT      = Path(__file__).parent.parent
PAIRS_DIR = ROOT / "training_data" / "pairs"
SAVE_PATH = ROOT / "backend" / "guitar_separator.pt"

SR        = 44100
N_FFT     = 2048
HOP       = 512
N_MELS    = 128
CLIP_SEC  = 6.0
CLIP_SAMP = int(CLIP_SEC * SR)
T_FRAMES  = 512
F_BINS    = N_FFT // 2 + 1   # 1025

N_CROPS   = 40     # Random crops per pair (diversity per epoch)
EPOCHS    = 100
BATCH     = 4
LR        = 3e-4
PATIENCE  = 15
VAL_RATIO = 0.15

# Mel filterbank (numpy) — back-project mel masks to STFT frequency axis in loss
_MEL_FB  = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)   # (N_MELS, F_BINS)
_FB_SUM  = _MEL_FB.sum(axis=0, keepdims=True) + 1e-8                 # (1, F_BINS)


# --- U-Net (same as guitar_separator_inference.py) ----------------------------

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation: learn per-frequency importance at the bottleneck."""
    def __init__(self, channels: int, reduction: int = 16):
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
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
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
    """
    Input:  (B, 1, N_MELS, T_FRAMES) — normalized log-mel
    Output: (B, 2, N_MELS, T_FRAMES) — [acoustic_mask, electric_mask] (sigmoid)
    """
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
            _SEBlock(512),   # Re-weight frequency bands at bottleneck
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


# --- Preprocessing helpers ----------------------------------------------------

def _wav_to_mel_norm(y: np.ndarray) -> np.ndarray:
    """waveform → normalized log-mel (N_MELS, T_FRAMES) — same preprocessing as inference."""
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=mel.max() + 1e-8)
    t = log_mel.shape[1]
    if t >= T_FRAMES:
        log_mel = log_mel[:, :T_FRAMES]
    else:
        log_mel = np.pad(log_mel, ((0, 0), (0, T_FRAMES - t)))
    m, s = log_mel.mean(), log_mel.std() + 1e-8
    return ((log_mel - m) / s).astype(np.float32)


def _stft_mag(y: np.ndarray) -> np.ndarray:
    """waveform → STFT magnitude (F_BINS, T_FRAMES) — same as inference."""
    D   = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(D)                # (F_BINS, t)
    t   = mag.shape[1]
    if t >= T_FRAMES:
        return mag[:, :T_FRAMES].astype(np.float32)
    return np.pad(mag, ((0, 0), (0, T_FRAMES - t))).astype(np.float32)


def _augment_electric(y: np.ndarray) -> np.ndarray:
    """Simulate electric guitar distortion (tanh drive)."""
    drive = np.random.uniform(1.5, 6.0)
    return np.tanh(y * drive) / (np.tanh(drive) + 1e-8)


def _augment_reverb(y: np.ndarray) -> np.ndarray:
    """Simulate acoustic guitar reverb (short delay + decay)."""
    delay = int(np.random.uniform(0.008, 0.04) * SR)
    decay = np.random.uniform(0.08, 0.28)
    out   = y.copy()
    if delay < len(y):
        out[delay:] += y[:-delay] * decay
    return out


# --- Dataset ------------------------------------------------------------------

class GuitarPairDataset(Dataset):
    """
    Cambridge MT pair dataset (v3).

    If mixture.wav exists → use full-instrument mix as input (v3)
    Otherwise         → use acoustic+electric sum (legacy fallback)

    Audio is cached in memory at init → random crops in __getitem__.
    Expect ~120 MB per pair (mixture + acoustic + electric).
    """

    def __init__(self, pair_dirs: list[Path], n_crops: int = N_CROPS, augment: bool = False):
        self.n_crops = n_crops
        self.augment = augment
        # (mix, ac, el, has_mixture) tuples
        self._audio: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []

        n_mix = 0
        n_skip = 0
        print(f"  Loading audio cache ({len(pair_dirs)} pairs) ...", flush=True)
        for i, d in enumerate(pair_dirs):
            try:
                ac, _ = librosa.load(str(d / "acoustic.wav"), sr=SR, mono=True)
                el, _ = librosa.load(str(d / "electric.wav"),  sr=SR, mono=True)

                mix_path = d / "mixture.wav"
                if mix_path.exists():
                    mix, _ = librosa.load(str(mix_path), sr=SR, mono=True)
                    # Align three tracks to shortest length
                    min_l = min(len(mix), len(ac), len(el))
                    mix = mix[:min_l]; ac = ac[:min_l]; el = el[:min_l]
                    has_mix = True
                    n_mix += 1
                else:
                    # Legacy pair: on-the-fly sum (fallback)
                    min_l = min(len(ac), len(el))
                    ac = ac[:min_l]; el = el[:min_l]
                    mix_raw = ac + el
                    peak = np.abs(mix_raw).max()
                    mix = mix_raw / peak * 0.9 if peak > 1e-6 else mix_raw
                    has_mix = False

                self._audio.append((mix, ac, el, has_mix))
                print(f"\r  [{i+1}/{len(pair_dirs)}] {d.name[:40]}", end="", flush=True)
            except Exception as e:
                n_skip += 1
                print(f"\r  [{i+1}/{len(pair_dirs)}] skip ({d.name[:30]}): {e}", flush=True)
        print()
        print(f"  Pairs with mixture.wav: {n_mix}/{len(pair_dirs)}  (skipped: {n_skip})", flush=True)

    def __len__(self):
        return len(self._audio) * self.n_crops

    def __getitem__(self, idx):
        mix_full, ac_full, el_full, has_mix = self._audio[idx // self.n_crops]
        min_len = len(mix_full)  # already aligned length

        # Random 6s crop (same position for mix / ac / el)
        if min_len > CLIP_SAMP:
            start = random.randint(0, min_len - CLIP_SAMP)
            mix = mix_full[start:start + CLIP_SAMP].copy()
            ac  = ac_full[start:start + CLIP_SAMP].copy()
            el  = el_full[start:start + CLIP_SAMP].copy()
        else:
            mix = np.pad(mix_full, (0, CLIP_SAMP - min_len))
            ac  = np.pad(ac_full,  (0, CLIP_SAMP - len(ac_full)))[:CLIP_SAMP]
            el  = np.pad(el_full,  (0, CLIP_SAMP - len(el_full)))[:CLIP_SAMP]

        # Augmentation: global volume scale (same factor for mix/ac/el → ratio preserved)
        if self.augment:
            gain = np.random.uniform(0.5, 1.2)
            mix = mix * gain
            ac  = ac  * gain
            el  = el  * gain
            # Per-timbre augmentation only for legacy pairs (safe when mix is on-the-fly sum)
            if not has_mix:
                if np.random.rand() < 0.45:
                    el = _augment_electric(el)
                if np.random.rand() < 0.30:
                    ac = _augment_reverb(ac)
                # Re-sum and re-normalize after augmentation
                mix = ac + el
                peak = np.abs(mix).max()
                if peak > 1e-6:
                    s = min(1.0, 0.95 / peak)
                    mix *= s; ac *= s; el *= s

        # Avoid mixture clipping
        peak = np.abs(mix).max()
        if peak > 0.95:
            s = 0.95 / peak
            mix *= s; ac *= s; el *= s

        return (
            torch.tensor(_wav_to_mel_norm(mix)).unsqueeze(0),       # (1, N_MELS, T_FRAMES)
            torch.tensor(_stft_mag(mix)),                            # (F_BINS, T_FRAMES)
            torch.tensor(_stft_mag(ac)),                             # (F_BINS, T_FRAMES)
            torch.tensor(_stft_mag(el)),                             # (F_BINS, T_FRAMES)
            torch.tensor(mix.astype(np.float32)),                    # (CLIP_SAMP,) for SI-SDR
            torch.tensor(ac.astype(np.float32)),                     # (CLIP_SAMP,) for SI-SDR
            torch.tensor(el.astype(np.float32)),                     # (CLIP_SAMP,) for SI-SDR
        )


# --- Loss — STFT magnitude domain (matches inference exactly) -----------------

def stft_sep_loss(
    masks:   torch.Tensor,   # (B, 2, N_MELS, T_FRAMES)
    mag_mix: torch.Tensor,   # (B, F_BINS, T_FRAMES)
    mag_ac:  torch.Tensor,   # (B, F_BINS, T_FRAMES)
    mag_el:  torch.Tensor,   # (B, F_BINS, T_FRAMES)
    mel_fb:  torch.Tensor,   # (N_MELS, F_BINS) — constant
    fb_sum:  torch.Tensor,   # (1, F_BINS) — constant
) -> tuple[torch.Tensor, float, float]:
    """
    Apply masks to STFT the same way as guitar_separator_inference.py.

    Loss = STFT L1 (linear) + 0.3 × mel L1 (perceptual)

    Returns: (total_loss, ac_l1, el_l1)
      ac_l1 / el_l1 are for logging only (backprop uses total_loss)
    """
    ac_mask_mel = masks[:, 0]   # (B, N_MELS, T)
    el_mask_mel = masks[:, 1]   # (B, N_MELS, T)

    # Mel mask → back-project to STFT frequency axis
    fb_T     = mel_fb.T
    fb_sum_t = fb_sum.T
    ac_stft = torch.einsum('fn,bnt->bft', fb_T, ac_mask_mel) / fb_sum_t
    el_stft = torch.einsum('fn,bnt->bft', fb_T, el_mask_mel) / fb_sum_t

    # Wiener-style normalization
    total   = ac_stft + el_stft + 1e-8
    ac_stft = ac_stft / total
    el_stft = el_stft / total

    # Predicted: mask × mix STFT magnitude
    ac_pred = ac_stft * mag_mix   # (B, F_BINS, T)
    el_pred = el_stft * mag_mix

    # --- Per-stem STFT L1 -----------------------------------------------------
    ac_stft_l1 = F.l1_loss(ac_pred, mag_ac)
    el_stft_l1 = F.l1_loss(el_pred, mag_el)

    # --- Auxiliary mel-domain L1 --------------------------------------------
    mel_ac_pred = torch.einsum('nf,bft->bnt', mel_fb, ac_pred)
    mel_el_pred = torch.einsum('nf,bft->bnt', mel_fb, el_pred)
    mel_ac_tgt  = torch.einsum('nf,bft->bnt', mel_fb, mag_ac)
    mel_el_tgt  = torch.einsum('nf,bft->bnt', mel_fb, mag_el)
    ac_mel_l1 = F.l1_loss(mel_ac_pred, mel_ac_tgt)
    el_mel_l1 = F.l1_loss(mel_el_pred, mel_el_tgt)

    ac_loss = ac_stft_l1 + 0.3 * ac_mel_l1
    el_loss = el_stft_l1 + 0.3 * el_mel_l1

    return ac_loss + el_loss, ac_loss.item(), el_loss.item()


def si_sdr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant SDR loss (waveform level).
    pred, target: (B, T)
    Returns: scalar (negative mean SI-SDR — lower is better)

    SI-SDR = 10 log10( |s_target|² / |e_noise|² )
    Scale-invariant so level differences do not skew separation quality.
    """
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    alpha  = (pred * target).sum(dim=-1) / (target.pow(2).sum(dim=-1) + eps)
    s_tgt  = alpha.unsqueeze(-1) * target
    noise  = pred - s_tgt
    ratio  = s_tgt.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
    return -10.0 * torch.log10(ratio + eps).mean()


def _fold_istft(
    spec: torch.Tensor,   # (B, F_BINS, T) complex
    n_fft: int,
    hop_length: int,
    window: torch.Tensor, # (n_fft,)
    length: int,
) -> torch.Tensor:
    """
    Manual overlap-add ISTFT without torch.istft COLA checks.
    Differentiable via F.fold.
    """
    B, F, T = spec.shape
    # IFFT per frame: (B, T, n_fft)
    frames = torch.fft.irfft(spec.permute(0, 2, 1), n=n_fft, dim=-1)
    frames = frames * window                          # (B, T, n_fft)
    frames = frames.permute(0, 2, 1).contiguous()    # (B, n_fft, T)

    sig_len = (T - 1) * hop_length + n_fft
    output = torch.nn.functional.fold(
        frames, output_size=(1, sig_len),
        kernel_size=(1, n_fft), stride=(1, hop_length),
    ).squeeze(1).squeeze(1)                           # (B, sig_len)

    # Normalize (window OLA)
    win_sq = (window ** 2).unsqueeze(0).unsqueeze(-1).expand(1, n_fft, T).contiguous()
    norm = torch.nn.functional.fold(
        win_sq, output_size=(1, sig_len),
        kernel_size=(1, n_fft), stride=(1, hop_length),
    ).squeeze(1).squeeze(1).clamp(min=1e-8)           # (1, sig_len)

    output = output / norm
    # Match length
    if output.shape[-1] < length:
        output = torch.nn.functional.pad(output, (0, length - output.shape[-1]))
    else:
        output = output[..., :length]
    return output


def si_sdr_from_masks(
    masks:    torch.Tensor,   # (B, 2, N_MELS, T_FRAMES)
    mix_wave: torch.Tensor,   # (B, CLIP_SAMP)
    ac_wave:  torch.Tensor,   # (B, CLIP_SAMP)
    el_wave:  torch.Tensor,   # (B, CLIP_SAMP)
    mel_fb:   torch.Tensor,   # (N_MELS, F_BINS)
    fb_sum:   torch.Tensor,   # (1, F_BINS)
    hann_win: torch.Tensor,   # (N_FFT,)
) -> tuple[torch.Tensor, float, float]:
    """
    Apply masks to complex STFT → ISTFT (manual OLA) → SI-SDR.
    Returns: (total_loss, ac_sdr_db, el_sdr_db)
    """
    # Mel mask → back-project to STFT frequency axis
    ac_mask_mel = masks[:, 0]
    el_mask_mel = masks[:, 1]
    fb_T     = mel_fb.T
    fb_sum_t = fb_sum.T
    ac_stft  = torch.einsum('fn,bnt->bft', fb_T, ac_mask_mel) / fb_sum_t
    el_stft  = torch.einsum('fn,bnt->bft', fb_T, el_mask_mel) / fb_sum_t
    total    = ac_stft + el_stft + 1e-8
    ac_mask_stft = ac_stft / total
    el_mask_stft = el_stft / total

    # Mix STFT (CPU, center=False)
    cpu = torch.device("cpu")
    win_cpu = hann_win.to(cpu)
    D = torch.stft(mix_wave.to(cpu), n_fft=N_FFT, hop_length=HOP,
                   window=win_cpu, return_complex=True, center=False)
    T_full = D.shape[2]
    T_mask = ac_mask_stft.shape[2]

    if T_mask < T_full:
        ac_m = torch.nn.functional.pad(ac_mask_stft, (0, T_full - T_mask)).to(cpu)
        el_m = torch.nn.functional.pad(el_mask_stft, (0, T_full - T_mask)).to(cpu)
    else:
        ac_m = ac_mask_stft[:, :, :T_full].to(cpu)
        el_m = el_mask_stft[:, :, :T_full].to(cpu)

    sig_len = mix_wave.shape[-1]
    ac_pred_wave = _fold_istft(D * ac_m, N_FFT, HOP, win_cpu, sig_len).to(mix_wave.device)
    el_pred_wave = _fold_istft(D * el_m, N_FFT, HOP, win_cpu, sig_len).to(mix_wave.device)

    ac_sdr = si_sdr_loss(ac_pred_wave, ac_wave)
    el_sdr = si_sdr_loss(el_pred_wave, el_wave)

    return ac_sdr + el_sdr, (-ac_sdr).item(), (-el_sdr).item()


# --- Training loop ------------------------------------------------------------

def train():
    pair_dirs = sorted([d for d in PAIRS_DIR.iterdir()
                        if d.is_dir() and (d / "acoustic.wav").exists()])
    if not pair_dirs:
        print(f"No training data: {PAIRS_DIR}")
        print("Run scripts/prepare_training_data.py first.")
        return

    random.shuffle(pair_dirs)
    n_val    = max(1, int(len(pair_dirs) * VAL_RATIO))
    val_dirs = pair_dirs[:n_val]
    trn_dirs = pair_dirs[n_val:]

    print(f"\nTrain pairs: {len(trn_dirs)}  /  Val pairs: {len(val_dirs)}")
    print(f"Samples per epoch: train {len(trn_dirs) * N_CROPS}  /  val {len(val_dirs) * N_CROPS}\n")

    print("[Loading train data]")
    trn_ds = GuitarPairDataset(trn_dirs, n_crops=N_CROPS, augment=True)
    print("[Loading val data]")
    val_ds = GuitarPairDataset(val_dirs, n_crops=N_CROPS, augment=False)

    trn_ld = DataLoader(trn_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"\nDevice: {device}")

    # Mel filterbank tensors (for loss)
    mel_fb   = torch.tensor(_MEL_FB, dtype=torch.float32, device=device)   # (N_MELS, F)
    fb_sum   = torch.tensor(_FB_SUM, dtype=torch.float32, device=device)   # (1, F)
    hann_win = torch.hann_window(N_FFT, device=device)                     # for ISTFT

    model     = GuitarSeparatorUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val = float("inf")
    no_imp   = 0

    hdr = (f"{'Epoch':>6}  {'trn_ac':>7}  {'trn_el':>7}  "
           f"{'val_ac':>7}  {'val_el':>7}  {'ac_SDR':>7}  {'el_SDR':>7}  {'time':>6}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # --- Train
        model.train()
        trn_ac = trn_el = 0.0
        for x_mel, mag_mix, mag_ac, mag_el, mix_wave, ac_wave, el_wave in trn_ld:
            x_mel    = x_mel.to(device)
            mag_mix  = mag_mix.to(device)
            mag_ac   = mag_ac.to(device)
            mag_el   = mag_el.to(device)
            mix_wave = mix_wave.to(device)
            ac_wave  = ac_wave.to(device)
            el_wave  = el_wave.to(device)
            optimizer.zero_grad()
            masks                       = model(x_mel)
            spec_loss, ac_l1, el_l1     = stft_sep_loss(masks, mag_mix, mag_ac, mag_el, mel_fb, fb_sum)
            sdr_loss, _, _              = si_sdr_from_masks(masks, mix_wave, ac_wave, el_wave,
                                                            mel_fb, fb_sum, hann_win)
            loss = spec_loss + 0.1 * sdr_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(x_mel)
            trn_ac += ac_l1 * n
            trn_el += el_l1 * n
        trn_ac /= len(trn_ds)
        trn_el /= len(trn_ds)

        # --- Validation
        model.eval()
        val_ac = val_el = val_sdr_ac = val_sdr_el = 0.0
        with torch.no_grad():
            for x_mel, mag_mix, mag_ac, mag_el, mix_wave, ac_wave, el_wave in val_ld:
                x_mel    = x_mel.to(device)
                mag_mix  = mag_mix.to(device)
                mag_ac   = mag_ac.to(device)
                mag_el   = mag_el.to(device)
                mix_wave = mix_wave.to(device)
                ac_wave  = ac_wave.to(device)
                el_wave  = el_wave.to(device)
                masks                        = model(x_mel)
                _, ac_l1, el_l1              = stft_sep_loss(masks, mag_mix, mag_ac, mag_el, mel_fb, fb_sum)
                _, ac_sdr_db, el_sdr_db      = si_sdr_from_masks(masks, mix_wave, ac_wave, el_wave,
                                                                  mel_fb, fb_sum, hann_win)
                n = len(x_mel)
                val_ac     += ac_l1      * n
                val_el     += el_l1      * n
                val_sdr_ac += ac_sdr_db  * n
                val_sdr_el += el_sdr_db  * n
        val_ac     /= len(val_ds)
        val_el     /= len(val_ds)
        val_sdr_ac /= len(val_ds)
        val_sdr_el /= len(val_ds)
        val_loss = val_ac + val_el

        scheduler.step()
        elapsed = time.time() - t0
        star    = " ★" if val_loss < best_val else ""
        print(
            f"{epoch:>6d}  {trn_ac:>7.4f}  {trn_el:>7.4f}  "
            f"{val_ac:>7.4f}  {val_el:>7.4f}  "
            f"{val_sdr_ac:>6.2f}dB  {val_sdr_el:>6.2f}dB  "
            f"{elapsed:>5.1f}s{star}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            no_imp   = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "sr": SR, "n_fft": N_FFT, "hop": HOP,
                    "n_mels": N_MELS, "t_frames": T_FRAMES,
                    "clip_sec": CLIP_SEC,
                },
                "val_loss": best_val,
                "n_train_pairs": len(trn_dirs),
            }, SAVE_PATH)
            print(f"  ★ saved (ac={val_ac:.4f}  el={val_el:.4f}  "
                  f"SI-SDR  ac={val_sdr_ac:.2f}dB  el={val_sdr_el:.2f}dB)", flush=True)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"\nEarly stop (patience={PATIENCE})")
                break

    print(f"\nBest val_loss: {best_val:.4f}")
    print(f"Model saved: {SAVE_PATH}")


if __name__ == "__main__":
    train()
