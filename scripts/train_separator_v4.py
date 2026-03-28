"""
Guitar acoustic/electric separator — v4

Deprecated for the default workflow: training on ``guitar_demucs.wav`` matches Demucs,
not the "separate from original full mix" pipeline. Prefer ``scripts/train_separator.py``
(v3: ``mixture.wav`` = full-band mix, saved as ``backend/guitar_separator.pt``).

핵심 개선점 (vs v3):
  1. Train/Inference 불일치 수정 (가장 중요)
       v3: 모델 입력 = mixture.wav (풀밴드 믹스)
       v4: 모델 입력 우선순위:
           ① guitar_demucs.wav  (prepare_demucs_stems.py로 생성, 추론과 완전 동일한 분포)
           ② acoustic + electric 합산 + bleed augmentation (guitar_demucs.wav 없을 때)
       → prepare_demucs_stems.py 먼저 실행하면 최고 품질 달성

  2. 멀티 해상도 STFT 손실
       FFT 512 / 1024 / 2048 세 가지 스케일로 L1 계산
       고주파(어택)와 저주파(바디) 모두 커버

  3. SI-SDR 가중치 상향 0.1 → 0.4
       파형 단위 분리 품질 직접 최적화

  4. 중복 페어 제거
       폴더명 " 2" 접미사 스킵

  5. SpecAugment (시간·주파수 마스킹)
       데이터 적을 때 과적합 방지

Usage:
    source .venv/bin/activate
    python scripts/train_separator_v4.py
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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

STRIDE_SEC  = 3.0               # 슬라이딩 윈도우 간격 (초)
STRIDE_SAMP = int(STRIDE_SEC * SR)
MIN_ENERGY  = 1e-4              # 이 이하면 무음 구간으로 스킵
EPOCHS    = 120
BATCH     = 4
LR        = 3e-4
PATIENCE  = 20
VAL_RATIO = 0.15

# Multi-resolution STFT 설정
MR_FFTS   = [512, 1024, 2048]
MR_HOPS   = [128, 256,  512 ]

# Bleed augmentation: Demucs 스템에 남아있는 다른 악기 노이즈 시뮬레이션
BLEED_PROB  = 0.5    # 50% 확률로 믹스 일부를 기타 스템에 섞음
BLEED_RANGE = (0.03, 0.15)  # 믹스의 3~15% 추가

# Mel filterbank
_MEL_FB  = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
_FB_SUM  = _MEL_FB.sum(axis=0, keepdims=True) + 1e-8


# ---------------------------------------------------------------------------
# U-Net (architecture 동일)
# ---------------------------------------------------------------------------

class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
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
    Input:  (B, 1, N_MELS, T_FRAMES) — normalized log-mel of guitar stem
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
            _SEBlock(512),
        )
        self.dec4 = _DecoderBlock(512, 256, 256)
        self.dec3 = _DecoderBlock(256, 128, 128)
        self.dec2 = _DecoderBlock(128, 64,  64)
        self.dec1 = _DecoderBlock(64,  32,  32)
        self.head = nn.Sequential(nn.Conv2d(32, 2, 1), nn.Sigmoid())

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.bottleneck(x4)
        return self.head(self.dec1(self.dec2(self.dec3(self.dec4(b, s4), s3), s2), s1))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _wav_to_mel_norm(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=mel.max() + 1e-8)
    t = log_mel.shape[1]
    if t >= T_FRAMES:
        log_mel = log_mel[:, :T_FRAMES]
    else:
        log_mel = np.pad(log_mel, ((0, 0), (0, T_FRAMES - t)))
    m, s = log_mel.mean(), log_mel.std() + 1e-8
    return ((log_mel - m) / s).astype(np.float32)


def _stft_mag(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    D   = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(D)
    t_frames = (CLIP_SAMP // hop) + 1
    t = mag.shape[1]
    if t >= t_frames:
        return mag[:, :t_frames].astype(np.float32)
    return np.pad(mag, ((0, 0), (0, t_frames - t))).astype(np.float32)


def _spec_augment(mel: np.ndarray, freq_mask_f: int = 15, time_mask_t: int = 40) -> np.ndarray:
    """SpecAugment: random freq/time masking."""
    mel = mel.copy()
    # Frequency masking
    f0 = random.randint(0, N_MELS - freq_mask_f)
    mel[f0:f0 + freq_mask_f, :] = 0.0
    # Time masking
    t0 = random.randint(0, T_FRAMES - time_mask_t)
    mel[:, t0:t0 + time_mask_t] = 0.0
    return mel


def _augment_electric(y: np.ndarray) -> np.ndarray:
    drive = np.random.uniform(1.5, 6.0)
    return np.tanh(y * drive) / (np.tanh(drive) + 1e-8)


def _augment_reverb(y: np.ndarray) -> np.ndarray:
    delay = int(np.random.uniform(0.008, 0.04) * SR)
    decay = np.random.uniform(0.08, 0.28)
    out   = y.copy()
    if delay < len(y):
        out[delay:] += y[:-delay] * decay
    return out


# ---------------------------------------------------------------------------
# Dataset  — v4: 입력 = guitar stem (ac + el), bleed augmentation 포함
# ---------------------------------------------------------------------------

class GuitarPairDataset(Dataset):
    """
    v4 핵심 변경:
      - model input = acoustic + electric (추론 시 Demucs 출력과 동일한 분포)
      - mixture.wav는 bleed augmentation에만 사용 (Demucs 잔류 노이즈 시뮬레이션)
    """

    def __init__(self, pair_dirs: list[Path], augment: bool = False):
        self.augment = augment
        # (ac_clip, el_clip, demucs_clip_or_None, mix_clip_or_None) 모든 유효 윈도우
        self._windows: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = []

        n_skip = n_demucs = 0
        print(f"  Loading {len(pair_dirs)} pairs...", flush=True)
        for i, d in enumerate(pair_dirs):
            try:
                ac, _ = librosa.load(str(d / "acoustic.wav"), sr=SR, mono=True)
                el, _ = librosa.load(str(d / "electric.wav"), sr=SR, mono=True)
                min_l = min(len(ac), len(el))
                ac = ac[:min_l]; el = el[:min_l]

                demucs_full = None
                if (d / "guitar_demucs.wav").exists():
                    g, _ = librosa.load(str(d / "guitar_demucs.wav"), sr=SR, mono=True)
                    demucs_full = g[:min_l]
                    n_demucs += 1

                mix_full = None
                if (d / "mixture.wav").exists():
                    m, _ = librosa.load(str(d / "mixture.wav"), sr=SR, mono=True)
                    mix_full = m[:min_l]

                # 슬라이딩 윈도우로 전체 곡 커버
                starts = range(0, max(1, min_l - CLIP_SAMP + 1), STRIDE_SAMP)
                n_added = 0
                for s in starts:
                    e = s + CLIP_SAMP
                    def w(arr):
                        if arr is None: return None
                        return arr[s:e].copy() if e <= len(arr) else np.pad(arr[s:], (0, e - len(arr)))
                    ac_w = w(ac); el_w = w(el)
                    # 통기타 또는 일렉 소리가 있는 구간만 학습
                    if np.sqrt(np.mean(ac_w**2)) < MIN_ENERGY and np.sqrt(np.mean(el_w**2)) < MIN_ENERGY:
                        continue
                    self._windows.append((ac_w, el_w, w(demucs_full), w(mix_full)))
                    n_added += 1

                print(f"\r  [{i+1}/{len(pair_dirs)}] {d.name[:40]} ({n_added}clips)", end="", flush=True)
            except Exception as e:
                n_skip += 1
                print(f"\r  [{i+1}/{len(pair_dirs)}] skip: {e}", flush=True)
        print(f"\n  Loaded: {len(pair_dirs)-n_skip}곡  총 {len(self._windows)}클립  (guitar_demucs: {n_demucs}/{len(pair_dirs)-n_skip})  skipped: {n_skip}", flush=True)

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        ac, el, demucs, mix = self._windows[idx]
        ac     = ac.copy()
        el     = el.copy()
        demucs = demucs.copy() if demucs is not None else None
        mix    = mix.copy()    if mix    is not None else None

        # 오디오 augmentation (ac/el 타깃에만 적용)
        if self.augment:
            gain = np.random.uniform(0.5, 1.2)
            ac *= gain; el *= gain
            if demucs is not None:
                demucs *= gain   # Demucs 스템도 같은 gain (비율 유지)

            # 음색 augmentation은 guitar_demucs.wav가 없을 때만 (합산 모드)
            if demucs is None:
                if np.random.rand() < 0.45:
                    el = _augment_electric(el)
                if np.random.rand() < 0.30:
                    ac = _augment_reverb(ac)

        # ★ 모델 입력 결정
        if demucs is not None:
            # ① 최우선: 실제 Demucs 출력 (추론과 동일한 분포)
            guitar_stem = demucs.copy()
        else:
            # ② Fallback: ac + el 합산 + bleed augmentation
            guitar_stem = ac + el

        peak = np.abs(guitar_stem).max()
        if peak > 0.95:
            s = 0.95 / peak
            guitar_stem *= s; ac *= s; el *= s

        # Bleed augmentation: guitar_demucs.wav 없을 때만 (demucs 모드는 이미 블리드 포함)
        if self.augment and demucs is None and mix is not None and np.random.rand() < BLEED_PROB:
            bleed = np.random.uniform(*BLEED_RANGE)
            guitar_stem = guitar_stem + mix * bleed
            peak2 = np.abs(guitar_stem).max()
            if peak2 > 0.95:
                guitar_stem *= (0.95 / peak2)

        # STFT 멀티 해상도 준비
        stft_data = {}
        for n, h in zip(MR_FFTS, MR_HOPS):
            stft_data[n] = (
                _stft_mag(guitar_stem, n, h),
                _stft_mag(ac,          n, h),
                _stft_mag(el,          n, h),
            )

        # Mel (모델 입력)
        mel_inp = _wav_to_mel_norm(guitar_stem)
        if self.augment and np.random.rand() < 0.5:
            mel_inp = _spec_augment(mel_inp)

        return (
            torch.tensor(mel_inp).unsqueeze(0),              # (1, N_MELS, T)
            # 멀티 해상도 STFT (각 scale별 mix/ac/el)
            {n: tuple(torch.tensor(x) for x in v) for n, v in stft_data.items()},
            torch.tensor(guitar_stem.astype(np.float32)),     # (CLIP_SAMP,) 파형
            torch.tensor(ac.astype(np.float32)),
            torch.tensor(el.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# 멀티 해상도 STFT 손실
# ---------------------------------------------------------------------------

def mr_stft_loss(
    masks:   torch.Tensor,   # (B, 2, N_MELS, T)
    mel_fb:  torch.Tensor,   # (N_MELS, F_BINS)
    fb_sum:  torch.Tensor,   # (1, F_BINS)
    stft_dict: dict,         # {n_fft: (mag_mix, mag_ac, mag_el)}
) -> tuple[torch.Tensor, float, float]:
    """멀티 해상도 STFT L1 손실 합산."""
    total_loss = torch.tensor(0.0, device=masks.device)
    ac_total = 0.0; el_total = 0.0
    weight_sum = 0.0

    for n_fft, (mag_mix, mag_ac, mag_el) in stft_dict.items():
        mag_mix = mag_mix.to(masks.device)
        mag_ac  = mag_ac.to(masks.device)
        mag_el  = mag_el.to(masks.device)

        F_cur = n_fft // 2 + 1
        T_cur = mag_mix.shape[-1]

        # mel mask → back-project to this FFT's frequency axis
        if F_cur != mel_fb.shape[1]:
            # 작은 FFT에서는 mel 개수 줄여서 empty filter 방지
            n_mels_cur = min(N_MELS, n_fft // 4)
            fb_np = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels_cur)
            fb    = torch.tensor(fb_np, dtype=torch.float32, device=masks.device)
            fbs   = fb.sum(dim=0, keepdim=True) + 1e-8
            # mask를 n_mels_cur 크기로 보간
            masks_cur = F.interpolate(masks, size=(n_mels_cur, masks.shape[-1]),
                                      mode='bilinear', align_corners=False)
        else:
            fb = mel_fb; fbs = fb_sum; masks_cur = masks

        n_mels_use = fb.shape[0]
        # mel mask를 현재 스케일의 mel 개수 + T_cur 크기로 보간
        ac_mask_mel = F.interpolate(
            masks_cur[:, 0:1], size=(n_mels_use, T_cur), mode='bilinear', align_corners=False
        ).squeeze(1)  # (B, n_mels_use, T_cur)
        el_mask_mel = F.interpolate(
            masks_cur[:, 1:2], size=(n_mels_use, T_cur), mode='bilinear', align_corners=False
        ).squeeze(1)

        ac_stft = torch.einsum('fn,bnt->bft', fb.T, ac_mask_mel) / fbs.T
        el_stft = torch.einsum('fn,bnt->bft', fb.T, el_mask_mel) / fbs.T
        denom   = ac_stft + el_stft + 1e-8
        ac_stft = ac_stft / denom
        el_stft = el_stft / denom

        T_use  = min(T_cur, mag_mix.shape[-1])
        ac_pred = ac_stft[:, :, :T_use] * mag_mix[:, :, :T_use]
        el_pred = el_stft[:, :, :T_use] * mag_mix[:, :, :T_use]

        # 가중치: 더 높은 FFT 해상도에 더 많은 비중 (저주파 품질)
        w = np.log2(n_fft / 256)
        ac_l1 = F.l1_loss(ac_pred, mag_ac[:, :, :T_use])
        el_l1 = F.l1_loss(el_pred, mag_el[:, :, :T_use])
        total_loss = total_loss + w * (ac_l1 + el_l1)
        ac_total  += ac_l1.item() * w
        el_total  += el_l1.item() * w
        weight_sum += w

    return total_loss / weight_sum, ac_total / weight_sum, el_total / weight_sum


# ---------------------------------------------------------------------------
# SI-SDR (기존과 동일)
# ---------------------------------------------------------------------------

def si_sdr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    alpha  = (pred * target).sum(dim=-1) / (target.pow(2).sum(dim=-1) + eps)
    s_tgt  = alpha.unsqueeze(-1) * target
    noise  = pred - s_tgt
    ratio  = s_tgt.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
    return -10.0 * torch.log10(ratio + eps).mean()


def _fold_istft(spec, n_fft, hop_length, window, length):
    B, n_freqs, T = spec.shape
    frames = torch.fft.irfft(spec.permute(0, 2, 1), n=n_fft, dim=-1) * window
    frames = frames.permute(0, 2, 1).contiguous()
    sig_len = (T - 1) * hop_length + n_fft
    fold = torch.nn.functional.fold
    output = fold(frames, output_size=(1, sig_len), kernel_size=(1, n_fft), stride=(1, hop_length)).squeeze(1).squeeze(1)
    win_sq = (window ** 2).unsqueeze(0).unsqueeze(-1).expand(1, n_fft, T).contiguous()
    norm = fold(win_sq, output_size=(1, sig_len), kernel_size=(1, n_fft), stride=(1, hop_length)).squeeze(1).squeeze(1).clamp(min=1e-8)
    output = output / norm
    if output.shape[-1] < length:
        output = torch.nn.functional.pad(output, (0, length - output.shape[-1]))
    else:
        output = output[..., :length]
    return output


def _wave_si_sdr(masks, mel_fb, fb_sum, guitar_wave, ac_wave, el_wave, hann_win):
    """masks → ISTFT → SI-SDR."""
    cpu = torch.device("cpu")
    win_cpu = hann_win.to(cpu)
    D = torch.stft(guitar_wave.to(cpu), n_fft=N_FFT, hop_length=HOP,
                   window=win_cpu, return_complex=True, center=False)
    T_full = D.shape[2]

    ac_mask_mel = masks[:, 0]
    el_mask_mel = masks[:, 1]

    ac_stft = torch.einsum('fn,bnt->bft', mel_fb.T, ac_mask_mel) / fb_sum.T
    el_stft = torch.einsum('fn,bnt->bft', mel_fb.T, el_mask_mel) / fb_sum.T
    denom   = ac_stft + el_stft + 1e-8
    ac_mask_stft = ac_stft / denom
    el_mask_stft = el_stft / denom

    T_mask = ac_mask_stft.shape[2]
    if T_mask < T_full:
        ac_m = F.pad(ac_mask_stft, (0, T_full - T_mask)).to(cpu)
        el_m = F.pad(el_mask_stft, (0, T_full - T_mask)).to(cpu)
    else:
        ac_m = ac_mask_stft[:, :, :T_full].to(cpu)
        el_m = el_mask_stft[:, :, :T_full].to(cpu)

    sig_len = guitar_wave.shape[-1]

    ac_pred = _fold_istft(D * ac_m, N_FFT, HOP, win_cpu, sig_len).to(guitar_wave.device)
    el_pred = _fold_istft(D * el_m, N_FFT, HOP, win_cpu, sig_len).to(guitar_wave.device)

    ac_sdr = si_sdr_loss(ac_pred, ac_wave)
    el_sdr = si_sdr_loss(el_pred, el_wave)
    return ac_sdr + el_sdr, (-ac_sdr).item(), (-el_sdr).item()


# ---------------------------------------------------------------------------
# 중복 페어 필터링
# ---------------------------------------------------------------------------

def filter_unique_pairs(pair_dirs: list[Path]) -> list[Path]:
    """
    ' 2', ' 3' 등 접미사 중복 폴더 제거.
    같은 곡의 ' 2' 버전은 제거.
    """
    import re
    unique = []
    seen_bases = set()
    for d in pair_dirs:
        # ' 2', ' 3' 접미사 제거하여 base name 추출
        base = re.sub(r'\s+\d+$', '', d.name)
        if base not in seen_bases:
            seen_bases.add(base)
            unique.append(d)
    print(f"  Unique pairs: {len(unique)} / {len(pair_dirs)} total")
    return unique


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train():
    all_dirs = sorted([d for d in PAIRS_DIR.iterdir()
                       if d.is_dir() and (d / "acoustic.wav").exists()])
    if not all_dirs:
        print(f"학습 데이터 없음: {PAIRS_DIR}")
        print("scripts/prepare_training_data.py 먼저 실행하세요.")
        return

    pair_dirs = filter_unique_pairs(all_dirs)
    random.shuffle(pair_dirs)

    n_val    = max(1, int(len(pair_dirs) * VAL_RATIO))
    val_dirs = pair_dirs[:n_val]
    trn_dirs = pair_dirs[n_val:]

    print(f"\nTrain: {len(trn_dirs)}쌍  /  Val: {len(val_dirs)}쌍\n")

    print("[학습 데이터 로드]")
    trn_ds = GuitarPairDataset(trn_dirs, augment=True)
    print("[검증 데이터 로드]")
    val_ds = GuitarPairDataset(val_dirs, augment=False)
    print(f"에포크당 샘플: train {len(trn_ds)}  /  val {len(val_ds)}\n")

    def collate(batch):
        mels      = torch.stack([b[0] for b in batch])
        stft_keys = list(batch[0][1].keys())
        stft_dict = {
            k: tuple(torch.stack([b[1][k][i] for b in batch]) for i in range(3))
            for k in stft_keys
        }
        guitar_w = torch.stack([b[2] for b in batch])
        ac_w     = torch.stack([b[3] for b in batch])
        el_w     = torch.stack([b[4] for b in batch])
        return mels, stft_dict, guitar_w, ac_w, el_w

    trn_ld = DataLoader(trn_ds, batch_size=BATCH, shuffle=True,  num_workers=0, collate_fn=collate)
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=collate)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"\n디바이스: {device}")

    mel_fb   = torch.tensor(_MEL_FB, dtype=torch.float32, device=device)
    fb_sum   = torch.tensor(_FB_SUM, dtype=torch.float32, device=device)
    hann_win = torch.hann_window(N_FFT, device=device)

    model     = GuitarSeparatorUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 기존 모델이 있으면 이어서 학습 (선택)
    if SAVE_PATH.exists():
        ans = input(f"\n기존 {SAVE_PATH.name} 발견. 이어서 학습? [y/N] ").strip().lower()
        if ans == 'y':
            ck = torch.load(SAVE_PATH, map_location=device, weights_only=False)
            try:
                model.load_state_dict(ck["model_state_dict"])
                print("  기존 가중치 로드 완료, 파인튜닝 시작")
            except Exception as e:
                print(f"  가중치 로드 실패 ({e}), 처음부터 학습")

    best_val = float("inf")
    no_imp   = 0

    hdr = f"{'Ep':>4}  {'통기타%':>6}  {'일렉%':>5}  {'time':>5}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # 학습
        model.train()
        trn_ac = trn_el = 0.0
        n_trn = 0
        for x_mel, stft_dict, gw, aw, ew in trn_ld:
            x_mel = x_mel.to(device)
            gw = gw.to(device); aw = aw.to(device); ew = ew.to(device)
            optimizer.zero_grad()
            masks = model(x_mel)
            # 멀티 해상도 STFT 손실
            mr_loss, ac_l1, el_l1 = mr_stft_loss(masks, mel_fb, fb_sum, stft_dict)
            # SI-SDR 손실 (파형)
            sdr_loss, _, _ = _wave_si_sdr(masks, mel_fb, fb_sum, gw, aw, ew, hann_win)
            loss = mr_loss + 0.4 * sdr_loss   # v4: SI-SDR 가중치 0.1→0.4
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(x_mel)
            trn_ac += ac_l1 * n; trn_el += el_l1 * n; n_trn += n

        trn_ac /= n_trn; trn_el /= n_trn

        # 검증
        model.eval()
        val_ac = val_el = val_sdr_ac = val_sdr_el = 0.0
        n_val_s = 0
        with torch.no_grad():
            for x_mel, stft_dict, gw, aw, ew in val_ld:
                x_mel = x_mel.to(device)
                gw = gw.to(device); aw = aw.to(device); ew = ew.to(device)
                masks = model(x_mel)
                _, ac_l1, el_l1 = mr_stft_loss(masks, mel_fb, fb_sum, stft_dict)
                _, ac_sdr, el_sdr = _wave_si_sdr(masks, mel_fb, fb_sum, gw, aw, ew, hann_win)
                n = len(x_mel)
                val_ac += ac_l1 * n; val_el += el_l1 * n
                val_sdr_ac += ac_sdr * n; val_sdr_el += el_sdr * n
                n_val_s += n

        val_ac /= n_val_s; val_el /= n_val_s
        val_sdr_ac /= n_val_s; val_sdr_el /= n_val_s
        val_loss = val_ac + val_el

        scheduler.step()
        elapsed = time.time() - t0
        ac_pct = max(0.0, min(100.0, (val_sdr_ac + 40) / 40 * 100))
        el_pct = max(0.0, min(100.0, (val_sdr_el + 40) / 40 * 100))
        star = " ★" if val_loss < best_val else ""
        print(
            f"{epoch:>4d}  {ac_pct:>5.1f}%  {el_pct:>4.1f}%  {elapsed:>4.1f}s{star}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            no_imp   = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "sr": SR, "n_fft": N_FFT, "hop": HOP,
                    "n_mels": N_MELS, "t_frames": T_FRAMES, "clip_sec": CLIP_SEC,
                },
                "val_loss": best_val,
                "n_train_pairs": len(trn_dirs),
                "version": "v4",
            }, SAVE_PATH)
            print(f"  ★ 저장 (통기타 {ac_pct:.1f}%  일렉 {el_pct:.1f}%)", flush=True)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"\nEarly stop (patience={PATIENCE})")
                break

    print(f"\n최적 val_loss: {best_val:.4f}")
    print(f"모델 저장: {SAVE_PATH}")


if __name__ == "__main__":
    import librosa  # noqa — ensure import at module level for filterbank cache
    train()
