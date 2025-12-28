from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------
# Activation functions
# --------------------------------------------------------------------

@torch.jit.script
def snake(x: Tensor, alpha: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return snake(x, self.alpha)


def activation(act: str, **act_params):
    if act == "ELU":
        return nn.ELU(**act_params)
    elif act == "Snake":
        return Snake1d(**act_params)
    elif act == "Tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {act}")


# --------------------------------------------------------------------
# Convolution layers with normalization
# --------------------------------------------------------------------

def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
    assert norm in ["none", "weight_norm"]
    if norm == "weight_norm":
        return weight_norm(module)
    return module


class NormConv1d(nn.Conv1d):
    """1D Convolution with optional weight normalization and causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",
        causal: bool = False,
        pad_mode: str = "none",
        **kwargs
    ):
        if pad_mode == "none":
            pad = (kernel_size - stride) * dilation // 2
        else:
            pad = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            **kwargs
        )

        if norm == "weight_norm":
            weight_norm(self)

        self.causal = causal
        self.pad_mode = pad_mode
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation

    def _pad(self, x: Tensor) -> Tensor:
        if self.pad_mode == "none":
            return x

        length = x.shape[-1]
        kernel_size = self._kernel_size
        dilation = self._dilation
        stride = self._stride

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel_size - stride
        n_frames = (length - effective_kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        extra_padding = ideal_length - length

        if self.causal:
            pad_x = F.pad(x, (padding_total, extra_padding))
        else:
            padding_right = extra_padding // 2
            padding_left = padding_total - padding_right
            pad_x = F.pad(x, (padding_left, padding_right + extra_padding))

        return pad_x

    def forward(self, x: Tensor) -> Tensor:
        x = self._pad(x)
        return super().forward(x)


class NormConvTranspose1d(nn.ConvTranspose1d):
    """1D Transposed Convolution with optional weight normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",
        causal: bool = False,
        pad_mode: str = "none",
        **kwargs
    ):
        if pad_mode == "none":
            padding = (stride + 1) // 2
            output_padding = 1 if stride % 2 else 0
        else:
            padding = 0
            output_padding = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            output_padding=output_padding,
            **kwargs
        )

        if norm == "weight_norm":
            weight_norm(self)

        self.causal = causal
        self.pad_mode = pad_mode
        self._kernel_size = kernel_size
        self._stride = stride

    def _unpad(self, x: Tensor) -> Tensor:
        if self.pad_mode == "none":
            return x

        length = x.shape[-1]
        kernel_size = self._kernel_size
        stride = self._stride

        padding_total = kernel_size - stride
        if self.causal:
            padding_left = 0
            end = length - padding_total
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            end = length - padding_right

        return x[..., padding_left:end]

    def forward(self, x: Tensor) -> Tensor:
        y = super().forward(x)
        return self._unpad(y)


# --------------------------------------------------------------------
# Residual blocks
# --------------------------------------------------------------------

class ResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        kernel: int = 7,
        dilation: int = 1,
        act: str = "Snake",
        stride: int = 1,
        compress: int = 1,
        pad_mode: str = "none",
        causal: bool = False,
        norm: str = "weight_norm",
        true_skip: bool = False,
    ):
        super().__init__()
        kernels = [kernel, 1]
        dilations = [dilation, 1]
        hidden = dim // compress

        if act == "Snake":
            act_params = {"channels": dim}
        elif act == "ELU":
            act_params = {"alpha": 1.0}
        else:
            raise ValueError(f"Unsupported activation: {act}")

        layers = []
        for i, (kernel_size, dil) in enumerate(zip(kernels, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernels) - 1 else hidden

            layers += [
                activation(act=act, **act_params),
                NormConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dil,
                    norm=norm,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]

        self.block = nn.Sequential(*layers)
        self.true_skip = true_skip

    def shortcut(self, x: Tensor, y: Tensor) -> Tensor:
        if self.true_skip:
            return x
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        return y + self.shortcut(x, y)


# --------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, kernel=7, dilation=1),
            ResidualUnit(dim // 2, kernel=7, dilation=3),
            ResidualUnit(dim // 2, kernel=7, dilation=9),
            Snake1d(dim // 2),
            NormConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                pad_mode="none",
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: List[int] = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        layers = [NormConv1d(1, d_model, kernel_size=7, pad_mode="none")]

        for stride in strides:
            d_model *= 2
            layers += [EncoderBlock(d_model, stride=stride)]

        layers += [
            Snake1d(d_model),
            NormConv1d(d_model, d_latent, kernel_size=3, pad_mode="none"),
        ]

        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# --------------------------------------------------------------------
# Decoder with Watermarking
# --------------------------------------------------------------------

default_decoder_convtr_kwargs = {
    "acts": ["Snake", "ELU"],
    "pad_mode": ["none", "auto"],
    "norm": ["weight_norm", "none"],
}

default_wm_encoder_kwargs = {
    "acts": ["Snake", "Tanh"],
    "pad_mode": ["none", "auto"],
    "norm": ["weight_norm", "none"],
}


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        stride_wm: int = 1,
        acts: Optional[List[str]] = None,
        pad_modes: Optional[List[str]] = None,
        norms: Optional[List[str]] = None,
        downsampling_factor: int = 3,
        last_kernel_size: Optional[int] = None,
    ):
        super().__init__()

        if acts is None:
            acts = default_decoder_convtr_kwargs["acts"]
        if pad_modes is None:
            pad_modes = default_decoder_convtr_kwargs["pad_mode"]
        if norms is None:
            norms = default_decoder_convtr_kwargs["norm"]

        conv_strides = [stride, stride_wm]
        conv_in_dim = input_dim
        conv_out_dim = output_dim
        layers = []

        for act, norm, pad_mode, conv_stride in zip(acts, norms, pad_modes, conv_strides):
            if act == "Snake":
                act_params = {"channels": input_dim}
                causal = False
            else:  # ELU
                act_params = {"alpha": 1.0}
                causal = True
                conv_in_dim //= downsampling_factor
                conv_out_dim //= downsampling_factor
            layers += [
                activation(act=act, **act_params),
                NormConvTranspose1d(
                    conv_in_dim,
                    conv_out_dim,
                    kernel_size=2 * conv_stride,
                    stride=conv_stride,
                    causal=causal,
                    pad_mode=pad_mode,
                    norm=norm,
                ),
            ]

        layers += [
            ResidualUnit(output_dim, dilation=1, act="Snake", compress=1, causal=False, pad_mode="none", norm="weight_norm", true_skip=False),
            ResidualUnit(output_dim, dilation=3, act="Snake", compress=1, causal=False, pad_mode="none", norm="weight_norm", true_skip=False),
            ResidualUnit(output_dim // downsampling_factor, kernel=3, act="ELU", compress=2, causal=True, pad_mode="auto", norm="none", true_skip=True),
            ResidualUnit(output_dim // downsampling_factor, kernel=3, act="ELU", compress=2, causal=True, pad_mode="auto", norm="none", true_skip=True),
            ResidualUnit(output_dim, dilation=9, act="Snake", compress=1, causal=False, pad_mode="none", norm="weight_norm", true_skip=False),
        ]

        if last_kernel_size is not None:
            layers += [ResidualUnit(output_dim, kernel=last_kernel_size, act="Snake", pad_mode="none", norm="weight_norm", causal=False, true_skip=True)]
        else:
            layers += [nn.Identity()]

        layers += [
            nn.ELU(alpha=1.0),
            NormConv1d(conv_out_dim, conv_in_dim, kernel_size=2 * stride_wm, stride=stride_wm, causal=True, pad_mode="auto", norm="none"),
        ]

        self.block = nn.ModuleList(layers)
        self._chunk_size = len(acts)

    def forward(self, x: Tensor) -> Tensor:
        layer_cnt = len(self.block)
        chunks = [self.block[i:i + self._chunk_size] for i in range(0, layer_cnt, self._chunk_size)]
        group = [layer for j, chunk in enumerate(chunks) if j % self._chunk_size == 0 for layer in chunk]
        group = nn.Sequential(*group)
        return group(x)

    def upsample_group(self) -> nn.Sequential:
        layer_cnt = len(self.block)
        chunks = [self.block[i:i + self._chunk_size] for i in range(0, layer_cnt, self._chunk_size)]
        group = [layer for j, chunk in enumerate(chunks) if j % self._chunk_size != 0 for layer in chunk]
        return nn.Sequential(*group[(len(group) // 2):])

    def downsample_group(self) -> nn.Sequential:
        layer_cnt = len(self.block)
        chunks = [self.block[i:i + self._chunk_size] for i in range(0, layer_cnt, self._chunk_size)]
        group = [layer for j, chunk in enumerate(chunks) if j % self._chunk_size != 0 for layer in chunk]
        return nn.Sequential(*group[:(len(group) // 2)])


class LSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T) -> (T, B, C)
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        # (T, B, C) -> (B, C, T)
        return y.permute(1, 2, 0)


class MsgProcessor(nn.Module):
    """Apply the secret message to the encoder output."""

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: Tensor, msg: Tensor) -> Tensor:
        indices = 2 * torch.arange(msg.shape[-1]).to(hidden.device)
        indices = indices.repeat(msg.shape[0], 1)
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)
        msg_aux = msg_aux.sum(dim=-2)
        msg_aux = msg_aux.unsqueeze(-1).repeat(1, 1, hidden.shape[2])
        hidden = hidden + msg_aux
        return hidden


class WatermarkEncoderBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 96,
        out_dim: int = 128,
        wm_channels: int = 32,
        hidden: int = 512,
        lstm_layers: Optional[int] = None,
        acts: Optional[List[str]] = None,
        pad_modes: Optional[List[str]] = None,
        norms: Optional[List[str]] = None,
    ):
        super().__init__()

        if acts is None:
            acts = default_wm_encoder_kwargs["acts"]
        if pad_modes is None:
            pad_modes = default_wm_encoder_kwargs["pad_mode"]
        if norms is None:
            norms = default_wm_encoder_kwargs["norm"]

        # pre: Snake(in_dim) -> Conv(in_dim -> 1) -> Tanh -> Conv(1 -> wm_channels)
        pre_layers = []
        for i, (act, norm, pad_mode) in enumerate(zip(acts, norms, pad_modes)):
            input_dim = in_dim if i == 0 else 1
            output_dim = 1 if i == 0 else wm_channels
            if act == "Snake":
                act_params = {"channels": in_dim}
                causal = False
            else:  # Tanh
                act_params = {}
                causal = True
            pre_layers += [
                activation(act=act, **act_params),
                NormConv1d(
                    input_dim,
                    output_dim,
                    kernel_size=7,
                    causal=causal,
                    pad_mode=pad_mode,
                    norm=norm,
                ),
            ]
        self.pre = nn.Sequential(*pre_layers)

        # post: LSTM(hidden) -> ELU -> Conv(hidden -> out_dim)
        post_layers = []
        if lstm_layers is not None:
            post_layers.append(LSTMBlock(hidden, hidden, lstm_layers))
        post_layers += [
            nn.ELU(alpha=1.0),
            NormConv1d(hidden, out_dim, kernel_size=7, causal=True, norm="none", pad_mode="auto"),
        ]
        self.post = nn.Sequential(*post_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Full forward through pre layers."""
        return self.pre(x)

    def forward_no_conv(self, x: Tensor) -> Tensor:
        """Forward through pre layers, but skip the last conv (output is 1 channel)."""
        # pre[0] = Snake, pre[1] = Conv(in->1), pre[2] = Tanh, pre[3] = Conv(1->wm_channels)
        # forward_no_conv: Snake -> Conv(in->1) -> Tanh, skip last conv
        for i, layer in enumerate(self.pre):
            if i == len(self.pre) - 1:
                break  # skip last conv
            x = layer(x)
        return x

    def post_process(self, x: Tensor) -> Tensor:
        return self.post(x)


class WatermarkDecoderBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        out_dim: int = 1,
        channels: int = 32,
        hidden: int = 512,
        lstm_layers: Optional[int] = None,
    ):
        super().__init__()

        # pre: Conv(in_dim -> hidden) -> LSTM
        pre_layers = [
            NormConv1d(in_dim, hidden, kernel_size=7, causal=True, norm="none", pad_mode="auto"),
        ]
        if lstm_layers is not None:
            pre_layers.append(LSTMBlock(hidden, hidden, lstm_layers))
        self.pre = nn.Sequential(*pre_layers)

        # post: ELU -> Conv(channels -> out_dim)
        self.post = nn.Sequential(
            nn.ELU(alpha=1.0),
            NormConv1d(channels, out_dim, kernel_size=7, causal=True, norm="none", pad_mode="auto"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pre(x)

    def post_process(self, x: Tensor) -> Tensor:
        return self.post(x)


class Watermarker(nn.Module):
    def __init__(
        self,
        dim: int,
        d_out: int = 1,
        d_latent: int = 128,
        channels: int = 32,
        hidden: int = 512,
        nbits: int = 16,
        lstm_layers: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_block = WatermarkEncoderBlock(dim, d_latent, channels, hidden=hidden, lstm_layers=lstm_layers)
        self.msg_processor = MsgProcessor(nbits, d_latent)
        self.decoder_block = WatermarkDecoderBlock(d_latent, d_out, channels, hidden=hidden, lstm_layers=lstm_layers)

    def random_message(self, bsz: int) -> Tensor:
        nbits = self.msg_processor.nbits
        return torch.randint(0, 2, (bsz, nbits), dtype=torch.float32)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        wm_rates: List[int],
        wm_channels: int = 32,
        nbits: int = 16,
        d_out: int = 1,
        d_wm_out: int = 128,
        blending: str = "linear",
    ):
        super().__init__()

        layers = [NormConv1d(input_channel, channels, kernel_size=7, stride=1)]

        output_dim = channels
        for i, (stride, wm_stride) in enumerate(zip(rates, wm_rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride, wm_stride))

        self.model = nn.ModuleList(layers)

        self.wm_model = Watermarker(
            output_dim, d_out, d_wm_out, wm_channels, hidden=512, nbits=nbits, lstm_layers=2,
        )
        self.alpha = wm_channels / d_wm_out
        self.blending = blending

    def forward(self, x: Tensor, message: Optional[Tensor] = None) -> Tensor:
        for layer in self.model:
            x = layer(x)
        return self.watermark(x, message)

    def watermark(self, x: Tensor, message: Optional[Tensor] = None) -> Tensor:
        if self.alpha == 0.0:
            return x

        # Encode through watermark encoder
        h = self.wm_model.encoder_block(x)

        # Upsample through decoder blocks (reversed)
        upsampler = [block.upsample_group() for block in self.model[1:]][::-1]
        for layer in upsampler:
            h = layer(h)

        # Post-process encoder
        h = self.wm_model.encoder_block.post_process(h)

        # Apply message
        if message is None:
            bsz = x.shape[0]
            message = self.wm_model.random_message(bsz)
        message = message.to(x.device)
        h = self.wm_model.msg_processor(h, message)

        # Decode through watermark decoder
        h = self.wm_model.decoder_block(h)

        # Downsample through decoder blocks
        downsampler = [block.downsample_group() for block in self.model[1:]]
        for layer in downsampler:
            h = layer(h)

        # Post-process decoder -> output is (B, 1, T)
        h = self.wm_model.decoder_block.post_process(h)

        # Blend: linear blending uses forward_no_conv which outputs 1 channel
        if self.blending == "conv":
            out = self.wm_model.encoder_block.forward(x) + self.alpha * h
        else:
            out = self.wm_model.encoder_block.forward_no_conv(x) + self.alpha * h

        return out


# --------------------------------------------------------------------
# VAE Bottleneck
# --------------------------------------------------------------------

class VAEBottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 0,
        codebook_dim: int = 512,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = NormConv1d(input_dim, codebook_dim * 2, kernel_size=1)
        self.out_proj = NormConv1d(codebook_dim, input_dim, kernel_size=1)

    def forward(self, z: Tensor, n_quantizers: int = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mean, scale = self.in_proj(z).chunk(2, dim=1)
        z_q, kl = self._vae_sample(mean, scale)
        z_q = self.out_proj(z_q)
        return z_q, torch.zeros(z_q.size(), device=z.device), z_q, kl, torch.tensor(0.0, device=z.device)

    def _vae_sample(self, mean: Tensor, scale: Tensor) -> Tuple[Tensor, Tensor]:
        stdev = F.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        return latents, kl


# --------------------------------------------------------------------
# Main DAC-VAE Model
# --------------------------------------------------------------------

class DACVAE(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 8, 10, 12],
        latent_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [12, 10, 8, 2],
        wm_rates: Optional[List[int]] = None,
        n_codebooks: int = 16,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
        quantizer_dropout: bool = False,
        sample_rate: int = 48000,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim

        if wm_rates is None:
            wm_rates = [8, 5, 4, 2]

        self.hop_length = int(np.prod(encoder_rates))
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.quantizer = VAEBottleneck(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )
        self.codebook_dim = codebook_dim

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            wm_rates,
        )

        self.apply(init_weights)

    def _pad(self, wavs: Tensor) -> Tensor:
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return F.pad(wavs, p1d, "reflect")
        return wavs

    def preprocess(self, audio_data: Tensor, sample_rate: Optional[int] = None) -> Tensor:
        if sample_rate is not None:
            assert sample_rate == self.sample_rate
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(audio_data, (0, right_pad))
        return audio_data

    def encode(self, audio_data: Tensor) -> Tensor:
        """Encode audio to continuous latent representation.

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        Returns
        -------
        Tensor[B x D x T']
            Continuous latent representation (D = codebook_dim = 128)
        """
        z = self.encoder(self._pad(audio_data))
        mean, scale = self.quantizer.in_proj(z).chunk(2, dim=1)
        stdev = F.softplus(scale) + 1e-4
        encoded_frames = torch.randn_like(mean) * stdev + mean
        return encoded_frames

    def decode(self, encoded_frames: Tensor, message: Optional[Tensor] = None) -> Tensor:
        """Decode latent representation to audio.

        Parameters
        ----------
        encoded_frames : Tensor[B x D x T']
            Continuous latent representation
        message : Tensor[B x nbits], optional
            Watermark message to embed

        Returns
        -------
        Tensor[B x 1 x T]
            Decoded audio
        """
        emb = self.quantizer.out_proj(encoded_frames)
        return self.decoder(emb, message=message)

    def forward(
        self,
        audio_data: Tensor,
        sample_rate: Optional[int] = None,
    ) -> dict:
        """Full forward pass: encode and decode."""
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        latent = self.encode(audio_data)
        x = self.decode(latent)
        return {
            "audio": x[..., :length],
            "latent": latent,
        }

    @classmethod
    def load(cls, path: str) -> "DACVAE":
        """Load model from path or HuggingFace hub."""
        if not os.path.exists(path):
            if path.startswith("facebook/"):
                try:
                    from huggingface_hub import hf_hub_download
                    path = hf_hub_download(repo_id=path, filename="weights.pth")
                except ImportError:
                    raise ImportError("huggingface_hub required for HuggingFace downloads")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if "metadata" in checkpoint and "kwargs" in checkpoint["metadata"]:
            kwargs = checkpoint["metadata"]["kwargs"]
            model = cls(**kwargs)
        else:
            model = cls()

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


# --------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------

def load_dacvae(
    path: str = "facebook/dacvae-watermarked",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> DACVAE:
    """Load DAC-VAE model."""
    model = DACVAE.load(path)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


@torch.inference_mode()
def encode_audio(model: DACVAE, audio: Tensor) -> Tensor:
    """Encode audio to latent representation."""
    if audio.ndim == 2:
        audio = audio.unsqueeze(1)
    return model.encode(audio)


@torch.inference_mode()
def decode_latent(model: DACVAE, latent: Tensor, message: Optional[Tensor] = None) -> Tensor:
    """Decode latent representation to audio."""
    return model.decode(latent, message=message)


def get_model_info() -> dict:
    """Get information about the DAC-VAE model configuration."""
    return {
        "sample_rate": 48000,
        "encoder_rates": [2, 8, 10, 12],
        "decoder_rates": [12, 10, 8, 2],
        "hop_length": 2 * 8 * 10 * 12,  # 1920
        "latent_rate_hz": 48000 / 1920,  # 25 Hz
        "latent_dim": 128,  # codebook_dim
        "encoder_latent_dim": 1024,  # before VAE projection
        "30s_latent_frames": int(30 * 48000 / 1920),  # 750
    }


if __name__ == "__main__":
    print("DAC-VAE Model Info:")
    info = get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nTesting with random audio...")
    model = DACVAE()
    audio = torch.randn(1, 1, 48000)

    latent = model.encode(audio)
    print(f"  Input audio shape: {audio.shape}")
    print(f"  Latent shape: {latent.shape}")

    recon = model.decode(latent)
    print(f"  Reconstructed audio shape: {recon.shape}")
