import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Tuple, Optional, Union

def make_pad_mask(lengths: Union[torch.Tensor, list], max_len: Optional[int] = None) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.
    
    Args:
        lengths: Batch of lengths (B,)
        max_len: Optional maximum length of the sequence
    Returns:
        mask: boolean tensor of shape (B, T) where True indicates padded index
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.long)
    bs = lengths.size(0)
    if max_len is None:
        max_len = int(lengths.max())
    seq_range = torch.arange(0, max_len, dtype=torch.long, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


class STFTProcessor(nn.Module):
    """PyTorch STFT Module replicating ESPnet's STFT behaviour."""
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.window_type = window

    def forward(self, input: torch.Tensor, ilens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Ensure 2D input (Batch, Nsamples)
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        # Instantiate window function
        if self.window_type is not None:
            window_func = getattr(torch, f"{self.window_type}_window")
            window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        else:
            window = None

        stft_kwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": window,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
            "return_complex": True,
        }
        
        # Perform STFT
        output = torch.stft(input.float(), **stft_kwargs)
        # Transpose from (Batch, Freq, Frames) to (Batch, Frames, Freq)
        output = output.transpose(1, 2)
        
        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad
            olens = torch.div(ilens - self.n_fft, self.hop_length, rounding_mode="trunc") + 1
            mask = make_pad_mask(olens, max_len=output.size(1)).to(output.device)
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            olens = None
            
        return output, olens


class LogMelProcessor(nn.Module):
    """PyTorch Log-Mel Filterbank Feature Extractor replicating ESPnet's LogMel."""
    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        htk: bool = False,
        log_base: Optional[float] = None,
    ):
        super().__init__()
        fmin = 0.0 if fmin is None else fmin
        fmax = fs / 2.0 if fmax is None else fmax
        
        self.mel_options = {
            "sr": fs,
            "n_fft": n_fft,
            "n_mels": n_mels,
            "fmin": fmin,
            "fmax": fmax,
            "htk": htk,
        }
        self.log_base = log_base
        
        # Build mel filterbank using librosa
        melmat = librosa.filters.mel(**self.mel_options)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def forward(self, power_spec: torch.Tensor, ilens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Multiply by mel matrix: (Batch, Frames, Freq) x (Freq, n_mels) -> (Batch, Frames, n_mels)
        mel_feat = torch.matmul(power_spec, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)
        
        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / np.log(self.log_base)
            
        if ilens is not None:
            mask = make_pad_mask(ilens, max_len=logmel_feat.size(1)).to(logmel_feat.device)
            logmel_feat = logmel_feat.masked_fill(mask.unsqueeze(-1), 0.0)
            
        return logmel_feat, ilens


class UtteranceMVN(nn.Module):
    """PyTorch Utterance Mean and Variance Normalization replicating ESPnet's UtteranceMVN."""
    def __init__(self, norm_means: bool = True, norm_vars: bool = True, eps: float = 1e-20, compat_espnet: bool = True):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps
        self.compat_espnet = compat_espnet

    def forward(self, x: torch.Tensor, ilens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1), dtype=torch.long)
            
        ilens_ = ilens.to(x.device, x.dtype).view(-1, 1, 1)
        mask = make_pad_mask(ilens, max_len=x.size(1)).to(x.device).unsqueeze(-1)
        
        # Ensure padding contains zeros
        x = x.masked_fill(mask, 0.0)
        
        # Calculate mean
        mean = x.sum(dim=1, keepdim=True) / ilens_
        
        if self.norm_means:
            x = x - mean
            if not self.compat_espnet:
                x = x.masked_fill(mask, 0.0)
            
            if self.norm_vars:
                # If compat_espnet is True, we calculate variance using non-zero padded values (ESPnet behavior)
                var = x.pow(2).sum(dim=1, keepdim=True) / ilens_
                std = torch.clamp(var.sqrt(), min=self.eps)
                x = x / std
                if not self.compat_espnet:
                    x = x.masked_fill(mask, 0.0)
        else:
            if self.norm_vars:
                y = x - mean
                y = y.masked_fill(mask, 0.0)
                var = y.pow(2).sum(dim=1, keepdim=True) / ilens_
                std = torch.clamp(var.sqrt(), min=self.eps)
                x = x / std
                x = x.masked_fill(mask, 0.0)
                
        return x, ilens


class SpecAugment(nn.Module):
    """PyTorch SpecAugment (Time & Frequency Masking) replicating ESPnet's implementation."""
    def __init__(
        self,
        time_mask_width_range: Tuple[int, int] = (0, 30),
        freq_mask_width_range: Tuple[int, int] = (0, 20),
        num_time_mask: int = 2,
        num_freq_mask: int = 2,
        replace_with_zero: bool = True,
    ):
        super().__init__()
        self.time_mask_width_range = time_mask_width_range
        self.freq_mask_width_range = freq_mask_width_range
        self.num_time_mask = num_time_mask
        self.num_freq_mask = num_freq_mask
        self.replace_with_zero = replace_with_zero

    def mask_along_axis(self, spec: torch.Tensor, mask_width_range: Tuple[int, int], dim: int, num_mask: int) -> torch.Tensor:
        B = spec.shape[0]
        D = spec.shape[dim]
        
        if D <= mask_width_range[1]:
            mask_width_range = (mask_width_range[0], max(1, D // 4))

        # mask_length: (B, num_mask, 1)
        mask_length = torch.randint(
            mask_width_range[0],
            mask_width_range[1] + 1,
            (B, num_mask),
            device=spec.device,
        ).unsqueeze(2)

        # mask_pos: (B, num_mask, 1)
        mask_pos = torch.randint(
            0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device
        ).unsqueeze(2)

        aran = torch.arange(D, device=spec.device)[None, None, :]
        # Check coordinates
        mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
        mask = mask.any(dim=1)
        
        if dim == 1:
            mask = mask.unsqueeze(2)
        elif dim == 2:
            mask = mask.unsqueeze(1)

        value = 0.0 if self.replace_with_zero else spec.mean().item()
        return spec.masked_fill(mask, value)

    def forward(self, spec: torch.Tensor, ilens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.training:
            return spec, ilens
            
        if self.num_freq_mask > 0:
            spec = self.mask_along_axis(spec, self.freq_mask_width_range, dim=2, num_mask=self.num_freq_mask)
            
        if self.num_time_mask > 0:
            spec = self.mask_along_axis(spec, self.time_mask_width_range, dim=1, num_mask=self.num_time_mask)
            
        return spec, ilens


class MusicAudioProcessor(nn.Module):
    """Unified, portable audio feature extractor pipeline inspired by ESPnet."""
    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        n_mels: int = 80,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        norm_means: bool = True,
        norm_vars: bool = True,
        apply_spec_augment: bool = False,
        time_mask_width_range: Tuple[int, int] = (0, 30),
        freq_mask_width_range: Tuple[int, int] = (0, 20),
        num_time_mask: int = 2,
        num_freq_mask: int = 2,
        compat_espnet: bool = True,
    ):
        super().__init__()
        self.stft = STFTProcessor(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window="hann",
            center=True,
            normalized=False,
            onesided=True,
        )
        self.logmel = LogMelProcessor(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.mvn = UtteranceMVN(
            norm_means=norm_means,
            norm_vars=norm_vars,
            compat_espnet=compat_espnet,
        )
        self.apply_spec_augment = apply_spec_augment
        self.spec_augment = SpecAugment(
            time_mask_width_range=time_mask_width_range,
            freq_mask_width_range=freq_mask_width_range,
            num_time_mask=num_time_mask,
            num_freq_mask=num_freq_mask,
        )

    def forward(self, waveforms: torch.Tensor, ilens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # waveforms: (Batch, Nsamples)
        complex_spec, olens = self.stft(waveforms, ilens)
        power_spec = complex_spec.real**2 + complex_spec.imag**2
        logmel_feat, olens = self.logmel(power_spec, olens)
        norm_feat, olens = self.mvn(logmel_feat, olens)
        
        if self.apply_spec_augment:
            norm_feat, olens = self.spec_augment(norm_feat, olens)
            
        return norm_feat, olens
