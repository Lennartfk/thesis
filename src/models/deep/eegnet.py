import torch
from torch import nn


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max-norm constrained weights, as used in EEGNet depthwise filters."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """EEGNet for binary/multiclass EEG epoch classification.

    Expected input shape is one of:
    - (batch, channels, samples)
    - (batch, 1, channels, samples)

    The SEED-VIG defaults match the current epoched data:
    17 EEG channels, 1600 samples per 8-second epoch at 200 Hz.
    """

    def __init__(
        self,
        n_channels=17,
        n_samples=1600,
        n_classes=2,
        f1=8,
        depth_multiplier=2,
        f2=None,
        temporal_kernel_length=64,
        separable_kernel_length=16,
        dropout=0.5,
        pool1_kernel=4,
        pool2_kernel=8,
    ):
        super().__init__()
        if f2 is None:
            f2 = f1 * depth_multiplier

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, temporal_kernel_length),
                padding=(0, temporal_kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )

        self.spatial_block = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=f1,
                out_channels=f1 * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1_kernel)),
            nn.Dropout(dropout),
        )

        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f1 * depth_multiplier,
                kernel_size=(1, separable_kernel_length),
                padding=(0, separable_kernel_length // 2),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2_kernel)),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(self._infer_classifier_features(), n_classes)

    def _infer_classifier_features(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            features = self._forward_features(dummy)
            return int(features.shape[1])

    def _forward_features(self, x):
        x = self.temporal_block(x)
        x = self.spatial_block(x)
        x = self.separable_block(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                "EEGNet expects input with shape (batch, channels, samples) "
                "or (batch, 1, channels, samples)."
            )
        features = self._forward_features(x)
        return self.classifier(features)


def build_eegnet(
    seed=42,
    n_channels=17,
    n_samples=1600,
    n_classes=2,
    dropout=0.5,
    f1=8,
    depth_multiplier=2,
    f2=None,
):
    torch.manual_seed(seed)
    return EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        dropout=dropout,
        f1=f1,
        depth_multiplier=depth_multiplier,
        f2=f2,
    )
