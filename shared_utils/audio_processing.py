# 音訊處理
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional


def load_audio(file_path: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform


def get_mel_spectrogram(
    # 生成梅爾頻譜圖
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int
) -> torch.Tensor:
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    return mel_spectrogram_transform(waveform)


def convert_to_db(mel_spec: torch.Tensor) -> torch.Tensor:
    # 轉換為分貝刻度
    return T.AmplitudeToDB()(mel_spec)


def normalize_spectrogram(
    # 標準化頻譜圖
    spec: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if mean is not None and std is not None:
        return (spec - mean) / std
    return spec


def pad_or_truncate_spectrogram(spec: torch.Tensor, target_length: int) -> torch.Tensor:
    #統一長度
    current_length = spec.shape[-1]

    if current_length < target_length:
        pad_size = target_length - current_length
        spec = torch.nn.functional.pad(spec, (0, pad_size))
    elif current_length > target_length:
        spec = spec[:, :, :target_length]

    return spec