import torch
import torch.nn as nn


# import resnet3D

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


def deep_reg(in_channels=2, out_channels=3):
    return nn.Sequential(
        nn.MaxPool3d(2),
        nn.Conv3d(kernel_size=3, in_channels=in_channels, out_channels=8),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.BatchNorm3d(8),

        nn.Conv3d(kernel_size=3, in_channels=8, out_channels=32),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.BatchNorm3d(32),

        nn.Conv3d(kernel_size=3, in_channels=32, out_channels=32),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm3d(32),

        nn.Conv3d(kernel_size=3, in_channels=32, out_channels=64),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm3d(64),

        nn.Conv3d(kernel_size=3, in_channels=64, out_channels=64),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm3d(64),

        nn.Flatten(),

        nn.Linear(in_features=512, out_features=512),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm1d(512),

        nn.Linear(in_features=512, out_features=512),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm1d(512),

        nn.Linear(in_features=512, out_features=256),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm1d(256),

        nn.Linear(in_features=256, out_features=out_channels)
    )


class Loupe(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, eps=0.01):
        super(Loupe, self).__init__()
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        self.device = device

        # UNet
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv3d(64, image_dims[-1], 1)

        # regression
        self.reg_rot = deep_reg()

        self.reg_trans = deep_reg()

        # Mask
        self.pmask = nn.Parameter(
            torch.FloatTensor(*self.image_dims))  # Mask is same dimension as image plus complex domain
        self.pmask.requires_grad = True
        self.pmask.data.uniform_(eps, 1 - eps)  # uniform sparsity, 均匀分布U~[0,1)
        self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
        self.pmask.data = self.pmask.data.to(self.device)

    def squash_mask(self, mask):
        return torch.sigmoid(self.pmask_slope * mask)

    def sparsify(self, mask):
        xbar = mask.mean()
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = (r <= 1).float()
        return le * mask * r + (1 - le) * (1 - (1 - mask) * beta)

    def threshold(self, mask):
        random_uniform = torch.empty(*self.image_dims).uniform_(0, 1).to(self.device)
        return torch.sigmoid(self.sample_slope * (mask - random_uniform))

    def undersample(self, x, prob_mask):
        x_complex = x.contiguous().view(self.image_dims[3], -1, self.image_dims[0], self.image_dims[1],
                                        self.image_dims[2])
        x_real = x_complex.real
        x_imag = x_complex.imag
        mask = prob_mask.expand(x.shape[1], -1, -1, -1)  # former: x.shape[1], -1, -1, -1

        # print("x_complex's device:" + x.deivce)
        # print("x_real's device:" + x_real.deivce)
        # print("mask's device:" + mask.device)

        x_real = torch.mul(x_real, mask[:, :, :, 0])
        x_imag = torch.mul(x_imag, mask[:, :, :, 0])
        x_comp = torch.complex(x_real, x_imag)
        x_comp = x_comp.view(-1, self.image_dims[0], self.image_dims[1],
                             self.image_dims[2], self.image_dims[3])
        return x_comp

    def unet(self, x):
        x = x.float()
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        low_rank_rep = x

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        return x, low_rank_rep

    def regression(self, x):
        conv1_1 = self.down1(x)
        conv1_2 = self.dconv_main1(conv1_1)
        res1 = torch.cat([conv1_1, conv1_2], dim=1)

        conv2_1 = self.down2(res1)
        conv2_2 = self.dconv_main2(conv2_1)
        res2 = torch.cat([conv2_1, conv2_2], dim=1)

        conv3_1 = self.down3(res2)
        conv3_2 = self.dconv_main3(conv3_1)
        res3 = torch.cat([conv3_1, conv3_2], dim=1)

        conv4_1 = self.down4(res3)
        conv4_2 = self.dconv_main4(conv4_1)
        res4 = torch.cat([conv4_1, conv4_2], dim=1)

        return res4

    def forward(self, data):
        # FFT into k space
        x = data[:, :, :, :, 0]
        x = torch.unsqueeze(x, -1)
        x_gt = x[:, 3:76, 7:72, 5:74, :]

        ref = data[:, :, :, :, 1]
        ref = torch.unsqueeze(ref, -1)
        x = torch.fft.fftn(x)

        # Apply probabilistic mask
        probmask = self.squash_mask(self.pmask)

        # Sparsify
        sparse_mask = self.sparsify(probmask)

        # Threshold
        mask = self.threshold(sparse_mask)

        # Under sample
        x = self.undersample(x, mask)

        # iFFT into image space
        x = torch.fft.ifftn(x)

        # x_c = x
        # Through unet

        # print(x == x.real)

        x_in = x.real.contiguous().view(-1, 1, self.image_dims[0], self.image_dims[1],
                                        self.image_dims[2])  # Reshape for convolution

        unet_tensor, low_rank_in = self.unet(x_in)
        unet_tensor = unet_tensor.view(-1, *self.image_dims)  # Reshape for convolution

        # residual connection
        out_recon = unet_tensor + x.real  # [2,80,80,80,1]

        out = out_recon[:, 3:76, 7:72, 5:74, :]

        reg_in = torch.cat([out_recon, ref], dim=-1)

        reg_in = torch.squeeze(reg_in)

        reg_in = reg_in.contiguous().view(-1, 2, self.image_dims[0], self.image_dims[1], self.image_dims[2])

        param_rot = self.reg_rot(reg_in)
        param_trans = self.reg_trans(reg_in)

        # print(out == unet_tensor)
        return x_gt, out, param_rot, param_trans
