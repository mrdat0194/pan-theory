import torch
import torch.nn as nn
from typing import Tuple, Optional
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.utterance_mvn import UtteranceMVN

class ESPnetAudioProcessor(nn.Module):
    """Audio feature extractor using official ESPnet modules."""
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
    ):
        super().__init__()
        self.frontend = DefaultFrontend(
            fs=fs,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            frontend_conf=None,  # No speech enhancement to match custom pipeline
        )
        self.mvn = UtteranceMVN(
            norm_means=norm_means,
            norm_vars=norm_vars,
        )

    def forward(self, waveforms: torch.Tensor, ilens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if ilens is None:
            ilens = waveforms.new_full([waveforms.size(0)], waveforms.size(1), dtype=torch.long)
        
        feats, feats_lens = self.frontend(waveforms, ilens)
        norm_feats, feats_lens = self.mvn(feats, feats_lens)
        return norm_feats, feats_lens
