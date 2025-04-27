
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn

class CleanCholesky(nn.Module):
    """
    PyTorch 神经网络层：用于从输入矩阵中提取 Cholesky 分解矩阵 T
    """
    def __init__(self):
        super(CleanCholesky, self).__init__()

    def forward(self, img):
        """
        Args:
            img (torch.Tensor): 形状 (batch_size,2, hilbert_size, hilbert_size)
                                表示神经网络输出的随机矩阵，其中最后一个维度是实部和虚部。

        Returns:
            T (torch.Tensor): 形状 (batch_size, hilbert_size, hilbert_size)
                              代表 Cholesky 分解矩阵 T
        """
        real = img[:, 0, :,:]#.reshape(-1,4,4)
        imag = img[:, 1, :,:]#.reshape(-1,4,4)
        # 确保虚部的对角线为 0
        diag_all = torch.diagonal(imag, dim1=-2, dim2=-1)
        imag = imag - torch.diag_embed(diag_all)

        # 取下三角部分
        imag = torch.tril(imag)
        real = torch.tril(real)

        # 组合为复数矩阵
        T = torch.complex(real, imag)
        return T


class DensityMatrixFromT(nn.Module):
    """
    PyTorch 神经网络层：用于从 Cholesky 矩阵 T 计算密度矩阵 rho
    """
    def __init__(self):
        super(DensityMatrixFromT, self).__init__()

    def forward(self, tmatrix):
        """
        Args:
            tmatrix (torch.Tensor): 形状 (batch_size, hilbert_size, hilbert_size)
                                    代表 Cholesky 分解矩阵 T

        Returns:
            rho (torch.Tensor): 形状 (batch_size, hilbert_size, hilbert_size)
                                代表归一化的密度矩阵
        """
        # T_dagger = tmatrix.transpose(-1, -2).conj()  # 计算共轭转置
        # proper_dm = torch.matmul(T_dagger, tmatrix)  # 计算 T†T
        #
        # # 计算迹，并确保保持归一化
        # all_traces = torch.trace(proper_dm).view(-1, 1, 1)  # 计算每个矩阵的迹并调整形状
        # rho = proper_dm / all_traces  # 归一化密度矩阵

        batch_size = tmatrix.shape[0]
        T = tmatrix
        # 计算共轭转置 T†
        T_dagger = T.transpose(-1, -2).conj()

        # 计算 T_dagger @ T
        proper_dm = torch.matmul(T_dagger, T)

        # 计算每个矩阵的迹
        traces = torch.einsum('bii->b', proper_dm)
        # 计算归一化因子 1/trace，并调整形状便于广播
        inv_traces = 1.0 / traces
        inv_traces = inv_traces.view(-1, 1, 1)

        # 归一化密度矩阵
        rho = proper_dm * inv_traces

        # 分离实部和虚部
        real_part = rho.real  # 取实部，形状 (32, 4, 4)
        imag_part = rho.imag  # 取虚部，形状 (32, 4, 4)

        # 堆叠成 (32, 2, 4, 4)，2 表示 real 和 imag 两个通道
        separated_rho = torch.stack([real_part, imag_part], dim=1)

        return separated_rho#separated_rho.view(batch_size, 2, -1)




# 定义 VAE 结构
class VAE(nn.Module):
    def __init__(self, latent_dim=2,input=16):
        super(VAE, self).__init__()
        self.input = input
        # Encoder 网络
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # (batch, 32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 4, 4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch, 128, 2, 2)
            nn.ReLU(),
            nn.Flatten(),  # (batch, 128 * 2 * 2)
        )

        # 计算均值和方差
        self.fc_mu = nn.Linear(128 * 2 * 2, latent_dim)  # 输出均值
        self.fc_logvar = nn.Linear(128 * 2 * 2, latent_dim)  # 输出 log 方差

        # Decoder 网络
        self.fc_decoder = nn.Linear(latent_dim, 128 * 2 * 2)  # 变换到 (batch, 128*2*2)
        self.decoder = nn.Sequential(
            #nn.Unflatten(1, (128, 2, 2)),  # 还原形状 (batch, 128, 2, 2)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 2, 16, 16)
        )

        self.clean_cholesky_layer = CleanCholesky()
        self.density_matrix_layer = DensityMatrixFromT()

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder(z).view(-1, 128, 2, 2)
        img = self.decoder(h)
        #print(img.shape)
        img1 = self.clean_cholesky_layer(img)
        #print(img1.shape)
        img2 = self.density_matrix_layer(img1)#img#
        #print(img2.shape)
        return img2

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
# class VAE2q(nn.Module):
#     def __init__(self, latent_dim=4):
#         super(VAE2q, self).__init__()
#         # Encoder
#         # Encoder
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # (16,16) -> (8,8)
#             #nn.BatchNorm2d(32),
#             #nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (8,8) -> (4,4)
#             #nn.BatchNorm2d(64),
#             #nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (4,4) -> (2,2)
#             #nn.BatchNorm2d(128),
#             #nn.ReLU(),
#             nn.Flatten()
#         )
#
#         self.fc_mu = nn.Linear(128 * 2 * 2, latent_dim)  # Mean vector
#         self.fc_logvar = nn.Linear(128 * 2 * 2, latent_dim)  # Log variance vector
#
#         # Decoder
#         self.fc_decode = nn.Linear(latent_dim, 128 * 2 * 2)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (2,2) -> (4,4)
#             #nn.BatchNorm2d(64),
#             #nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (4,4) -> (8,8)
#             #nn.BatchNorm2d(32),
#             #nn.ReLU(),
#             nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8,8) -> (16,16)
#             #nn.Sigmoid()
#         )
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         x = self.encoder(x)
#         mu, logvar = self.fc_mu(x), self.fc_logvar(x)
#         z = self.reparameterize(mu, logvar)
#         x_recon = self.fc_decode(z).view(-1, 128, 2, 2)
#         x_recon = self.decoder(x_recon)
#         return x_recon, mu, logvar


# class VAE2q(nn.Module):
#     def __init__(self, latent_dim=16):
#         super(VAE2q, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(2, 32, kernel_size=3, stride=2, padding=1),  # (16) -> (8)
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (8) -> (4)
#             nn.ReLU(),
#             nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # (4) -> (2)
#             nn.ReLU(),
#             nn.Flatten()
#         )
#
#         self.fc_mu = nn.Linear(128 * 2, latent_dim)  # Mean vector
#         self.fc_logvar = nn.Linear(128 * 2, latent_dim)  # Log variance vector
#
#         # Decoder
#         self.fc_decoder = nn.Linear(latent_dim, 128 * 2)
#         self.decoder = nn.Sequential(
#             #nn.Unflatten(1, (128, 2)),
#             nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (2) -> (4)
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (4) -> (8)
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8) -> (16)
#             #nn.Sigmoid()
#         )
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#     def decode(self, z):
#         h = self.fc_decoder(z)
#         return self.decoder(h)
#     def forward(self, x):
#         x = self.encoder(x)
#         mu, logvar = self.fc_mu(x), self.fc_logvar(x)
#         z = self.reparameterize(mu, logvar)
#         x_recon = self.fc_decoder(z).view(-1, 128, 2)
#         x_recon = self.decoder(x_recon)
#         return x_recon, mu, logvar
#
#
# class VAE2q_1(nn.Module):
#     def __init__(self, latent_dim=16):
#         super(VAE2q_1, self).__init__()
#         # Encoder
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # (15,15) -> (8,8)
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (8,8) -> (4,4)
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (4,4) -> (2,2)
#             nn.ReLU(),
#             nn.Flatten()
#         )
#
#         self.fc_mu = nn.Linear(128 * 2 * 2, latent_dim)  # Mean vector
#         self.fc_logvar = nn.Linear(128 * 2 * 2, latent_dim)  # Log variance vector
#
#         # Decoder
#         self.fc_decoder = nn.Linear(latent_dim, 128 * 2 * 2)
#         self.decoder = nn.Sequential(
#             #nn.Unflatten(1, (128, 2, 2)),
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (2,2) -> (4,4)
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (4,4) -> (8,8)
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=0),  # (8,8) -> (15,15)
#             #nn.Sigmoid()
#         )
#     def decode(self, z):
#         h = self.fc_decoder(z)
#         return self.decoder(h)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         x = self.encoder(x)
#         mu, logvar = self.fc_mu(x), self.fc_logvar(x)
#         z = self.reparameterize(mu, logvar)
#         x_recon = self.fc_decoder(z).view(-1, 128, 2, 2)
#         x_recon = self.decoder(x_recon)
#         return x_recon, mu, logvar
