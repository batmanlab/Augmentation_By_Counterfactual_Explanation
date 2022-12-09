import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from style_model import ModulatedConv2d, StyledConv, ConstantInput, PixelNorm, Upsample, Downsample, Blur, EqualLinear, ConvLayer, EqualConv2d

num_channel = 1
def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
    
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        
        return torch.cat((ll, lh, hl, hh), 1)
    
class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        
        return ll + lh + hl + hh


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.iwt = InverseHaarTransform(3)
            self.upsample = Upsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = ModulatedConv2d(in_channel, 3 * 4, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3 * 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.iwt(skip)
            skip = self.upsample(skip)
            skip = self.dwt(skip)

            out = out + skip

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        #print("Input: ", input.shape)
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        #print("out resblock: ", out.shape)
        return out
    
class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        num_class,
        n_mlp=1,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        num_channel = 1
    ):
        super().__init__()

        self.size = size  #256
        self.style_dim_ = style_dim
        self.num_class = num_class
        self.style_dim = style_dim  + int(style_dim/2) #512 + 256 = 768
        
        #Encode condition
        layers = [PixelNorm()]
        layers.append(
                EqualLinear(
                    num_class, int(style_dim/2), lr_mul=0.01, activation="fused_lrelu"
                ))

        self.encode_class = nn.Sequential(*layers)
        
        
        #Encoder
        self.encoder_channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            224: 64,
            512: 32,
            1024: 16
        } 
        log_size = int(math.log(size, 2))
        self.e_n_latents = log_size*2 - 4
        encoder_convs = [ConvLayer(3, self.encoder_channels[size], 1)]

        in_channel = self.encoder_channels[size]
        for i in range(log_size, 2, -1):
            out_channel = self.encoder_channels[2 ** (i - 1)]
            encoder_convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel
        encoder_convs.append(EqualConv2d(in_channel, self.e_n_latents*self.style_dim_, 4, padding=0, bias=False))    
        self.encoder_convs = nn.Sequential(*encoder_convs)
        
        
        #Generator
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier, #512
            128: 128 * channel_multiplier, #256
            256: 64 * channel_multiplier, #128
            224: 64 * channel_multiplier, #128
            512: 32 * channel_multiplier, #64
            1024: 16 * channel_multiplier, #32
        }
        
        self.input = ConstantInput(self.channels[4])  #Input [n , 512]
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, self.style_dim, blur_kernel=blur_kernel  #conv: 512 x 512 x 3
        )
        self.to_rgb1 = ToRGB(self.channels[4], self.style_dim, upsample=False)

        self.log_size = int(math.log(size, 2)) - 1   #7
        self.num_layers = (self.log_size - 2) * 2 + 1  #11

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]           
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    self.style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(ToRGB(out_channel, self.style_dim)) #512, 12, 1, upsample=False

            in_channel = out_channel

        self.iwt = InverseHaarTransform(3)

        self.n_latent = self.log_size * 2 - 2  #12

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises



    def get_latent(self, input):
        out = self.encoder_convs(input)
        latent = out.view(len(input), self.e_n_latents, self.style_dim_)
        return latent
    
    
    def forward(
        self,
        input,
        labels,
        return_latents=False,
    ):
        #encode condition
        #labels: [N, num_class]
        #Input [N, 3, 256, 256]
        latent_label = self.encode_class(labels) #[N, 256]
        latent_label = latent_label.unsqueeze(1).repeat(1, self.e_n_latents, 1) #[N, 12 , 256]
        
        out = self.encoder_convs(input) #[N, 6144, 1, 1]
        latent = out.view(len(input), self.e_n_latents, self.style_dim_) #[N, 12, 512]
        
        #concatenate latent + latent_label
        latent = torch.cat((latent, latent_label),2)  #[N, 12, 512 + 256]
        
        noise = [None] * self.num_layers #11 layers
        
        out = self.input(latent) #constant [N, 512, 4, 4]
        out = self.conv1(out, latent[:, 0], noise=noise[0])  #[N, 512, 4, 4]; 

        skip = self.to_rgb1(out, latent[:, 1])  #[N, 12, 4, 4]

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1) # [upsample] [N, 512, 8, 8],[N, 512, 16, 16]
            out = conv2(out, latent[:, i + 1], noise=noise2) #[N, 512, 8, 8],[N, 512, 16, 16]
            skip = to_rgb(out, latent[:, i + 2], skip) #[N, 12, 8, 8], [N, 12, 16, 16]
            i += 2
        #out: [N, 256, 128, 128]
        #skip: [N, 12, 128, 128]
        image = self.iwt(skip) #[N, 3, 256, 256]
        if return_latents:
            return image, latent 
        else:
            return image, None
        
        
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class FromRGB(nn.Module):
    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.downsample = downsample

        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = ConvLayer(3 * 4, out_channel, 1)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out


class Discriminator(nn.Module):
    def __init__(self, size, concate_size = 0, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        # size: 256
        self.concate_size = concate_size
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier, #512
            128: 128 * channel_multiplier, #256
            256: 64 * channel_multiplier, #128
            224: 64 * channel_multiplier, #128
            512: 32 * channel_multiplier, #64
            1024: 16 * channel_multiplier, #32
        }
        
        self.dwt = HaarTransform(3)

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1  #7

        in_channel = channels[size]  #512

        for i in range(log_size, 2, -1): #7,6,5,4,3
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel))
            
            in_channel = out_channel
            '''
            i: 7 RGB:12, 128, 1                Conv: 128, 128, 3; 128, 512, 3
            i: 6 RGB:12, 512, 1 downsample     Conv: 512, 512, 3; 512, 512, 3
            i: 5 RGB:12, 512, 1 downsample     Conv: 512, 512, 3; 512, 512, 3
            i: 4 RGB:12, 512, 1 downsample     Conv: 512, 512, 3; 512, 512, 3
            i: 3 RGB:12, 512, 1 downsample     Conv: 512, 512, 3; 512, 512, 3
            '''

        self.from_rgbs.append(FromRGB(channels[4]))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)  #513 512 3
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4 + concate_size, channels[4], activation="fused_lrelu"),  #8192 , 512
            EqualLinear(channels[4], 1),  # 512, 1
        )
        

    def forward(self, input, cls_feature):
        input = self.dwt(input)
        out = None

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out) # [n, 512, 64, 64], [n, 512, 32, 32], [n, 512, 8, 8], [n, 512, 4, 4]

        _, out = self.from_rgbs[-1](input, out) #[n, 512, 4, 4]

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1) #[n, 513, 4, 4]

        out = self.final_conv(out) #[n, 512, 4, 4]

        out = out.view(batch, -1) #[n, 8192]
        if self.concate_size != 0:
            out = torch.cat([out, cls_feature],1) #[n, 9856]
        out = self.final_linear(out) #[n, 1]

        return out

