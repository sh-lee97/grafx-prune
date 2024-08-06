import torch.nn as nn
from auraloss_opt.freq import SumAndDifferenceSTFTLoss


class MRSTFTLoss(nn.Module):
    def __init__(
        self,
        sr=30000,
        fft_sizes=[512, 1024, 4096],
        hop_sizes=[128, 256, 1024],
        win_lengths=[512, 1024, 4096],
        n_bins=96,
        omit_sec=1,
    ):
        super().__init__()
        self.loss = SumAndDifferenceSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            perceptual_weighting=True,
            sample_rate=sr,
            scale="mel",
            n_bins=n_bins,
            eps=1e-4,
            output="full",
        )
        self.omit = int(sr * omit_sec)

    def forward(self, pred, true):
        audio_len = pred.shape[-1]
        if pred.ndim > 3:
            pred = pred.view(-1, 2, audio_len)
        if true.ndim > 3:
            true = true.view(-1, 2, audio_len)
        pred, true = pred[..., self.omit :], true[..., self.omit :]
        full, lr_loss, sum_loss, diff_loss = self.loss(pred, true)
        loss_dict = {
            "match/full": full,
            "match/lr": lr_loss,
            "match/sum": sum_loss,
            "match/diff": diff_loss,
        }
        return loss_dict
