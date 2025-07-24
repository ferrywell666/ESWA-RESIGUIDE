import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from compressai.layers import subpel_conv3x3
from modules.layers.conv import conv1x1, conv3x3, conv, deconv
from modules.layers.res_blk import *

class FGCM(nn.Module):
    def __init__(self, dim, out_dim, K=4, sigma_s=0.01):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.K = K
        self.sigma_s = sigma_s
        self.intermediates = {}
        
        self.rho_generator = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.mlp_low = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, K)
        )
        
        self.mlp_high = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(), 
            nn.Linear(dim // 2, K)
        )
        
        self.filter_bases_low = nn.Parameter(
            torch.randn(K, dim, 2) * 0.02
        )
        
        self.filter_bases_high = nn.Parameter(
            torch.randn(K, dim, 2) * 0.02
        )

    def forward(self, y_hat_prev):
        B, C, H, W = y_hat_prev.shape
        
        x_proc = y_hat_prev.to(torch.float32)
        F_raw = torch.fft.rfft2(x_proc, dim=(2, 3), norm='ortho')
        
        F_real = F_raw.real
        F_imag = F_raw.imag
        Psi = torch.cat([
            F.adaptive_avg_pool2d(F_real, (1, 1)).flatten(1),
            F.adaptive_avg_pool2d(F_imag, (1, 1)).flatten(1)
        ], dim=-1)
        
        rho = self.rho_generator(Psi) * 0.5
        
        freq_y = torch.fft.fftfreq(H, device=y_hat_prev.device)
        freq_x = torch.fft.rfftfreq(W, device=y_hat_prev.device)
        yy, xx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        distance = torch.sqrt(yy.pow(2) + xx.pow(2))
        max_distance = torch.sqrt(torch.tensor(0.5, device=y_hat_prev.device))
        d = distance / max_distance
        
        d_expanded = d.unsqueeze(0).unsqueeze(0)
        rho_expanded = rho.view(B, 1, 1, 1)
        
        Mask_low = 1.0 / (1.0 + torch.exp((d_expanded - rho_expanded) / self.sigma_s))
        Mask_high = 1.0 - Mask_low
        
        alpha_low = F.softmax(self.mlp_low(Psi), dim=1)
        alpha_high = F.softmax(self.mlp_high(Psi), dim=1)
        
        filter_bases_low = torch.view_as_complex(
            self.filter_bases_low.view(self.K, C, 2).permute(0, 2, 1)
        ).unsqueeze(-1).unsqueeze(-1)
        
        filter_bases_high = torch.view_as_complex(
            self.filter_bases_high.view(self.K, C, 2).permute(0, 2, 1)
        ).unsqueeze(-1).unsqueeze(-1)
        
        W_low = torch.zeros(B, C, 1, 1, dtype=torch.complex64, device=y_hat_prev.device)
        W_high = torch.zeros(B, C, 1, 1, dtype=torch.complex64, device=y_hat_prev.device)
        
        for k in range(self.K):
            alpha_k_low = alpha_low[:, k].view(B, 1, 1, 1)
            alpha_k_high = alpha_high[:, k].view(B, 1, 1, 1)
            W_low += alpha_k_low * filter_bases_low[k].unsqueeze(0)
            W_high += alpha_k_high * filter_bases_high[k].unsqueeze(0)
        
        W_low = W_low.expand(-1, -1, H, W//2+1)
        W_high = W_high.expand(-1, -1, H, W//2+1)
        
        F_filter_low = (F_raw * Mask_low) * W_low
        F_filter_high = (F_raw * Mask_high) * W_high
        
        self.intermediates['F_raw'] = F_raw.detach()
        self.intermediates['F_filter_low'] = F_filter_low.detach()
        self.intermediates['F_filter_high'] = F_filter_high.detach()
        
        F_filter = F_filter_low + F_filter_high
        Phi_global = torch.fft.irfft2(F_filter, s=(H, W), dim=(2, 3), norm='ortho')
        
        return Phi_global

class RLCM(nn.Module):
    def __init__(self, dim, out_dim, max_displacement=0.25):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.max_displacement = max_displacement
        self.intermediates = {}
        
        wavelet = pywt.Wavelet('haar')
        dec_lo, dec_hi, _, _ = wavelet.filter_bank
        
        self.register_buffer('wavelet_low', torch.Tensor(dec_lo).unsqueeze(0).unsqueeze(0))
        self.register_buffer('wavelet_high', torch.Tensor(dec_hi).unsqueeze(0).unsqueeze(0))
        
        self.offset_generator = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 6, kernel_size=3, padding=1)
        )
        
        nn.init.zeros_(self.offset_generator[-1].weight)
        nn.init.zeros_(self.offset_generator[-1].bias)
        
        self.conv_ll = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_lh = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_hl = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_hh = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.fusion = nn.Conv2d(dim * 4, out_dim, kernel_size=1)
    
    def _dwt_decompose(self, x):
        B, C, H, W = x.shape
        
        pad_h = (self.wavelet_low.shape[-2] - 1) // 2
        pad_w = (self.wavelet_low.shape[-1] - 1) // 2
        
        lo_filter_h = self.wavelet_low.view(1, 1, 1, -1)
        hi_filter_h = self.wavelet_high.view(1, 1, 1, -1)
        lo_filter_v = self.wavelet_low.view(1, 1, -1, 1)
        hi_filter_v = self.wavelet_high.view(1, 1, -1, 1)
        
        ll, lh, hl, hh = [], [], [], []
        
        for c in range(C):
            x_c = x[:, c:c+1]
            
            x_l = F.conv2d(
                F.pad(x_c, (pad_w, pad_w, 0, 0), mode='reflect'),
                lo_filter_h,
                stride=(1, 2)
            )
            
            x_h = F.conv2d(
                F.pad(x_c, (pad_w, pad_w, 0, 0), mode='reflect'),
                hi_filter_h,
                stride=(1, 2)
            )
            
            x_ll = F.conv2d(
                F.pad(x_l, (0, 0, pad_h, pad_h), mode='reflect'),
                lo_filter_v,
                stride=(2, 1)
            )
            
            x_lh = F.conv2d(
                F.pad(x_l, (0, 0, pad_h, pad_h), mode='reflect'),
                hi_filter_v,
                stride=(2, 1)
            )
            
            x_hl = F.conv2d(
                F.pad(x_h, (0, 0, pad_h, pad_h), mode='reflect'),
                lo_filter_v,
                stride=(2, 1)
            )
            
            x_hh = F.conv2d(
                F.pad(x_h, (0, 0, pad_h, pad_h), mode='reflect'),
                hi_filter_v,
                stride=(2, 1)
            )
            
            if c == 0:
                ll = x_ll
                lh = x_lh
                hl = x_hl
                hh = x_hh
            else:
                ll = torch.cat([ll, x_ll], dim=1)
                lh = torch.cat([lh, x_lh], dim=1)
                hl = torch.cat([hl, x_hl], dim=1)
                hh = torch.cat([hh, x_hh], dim=1)
        
        return ll, lh, hl, hh
    
    def _apply_deformable_sampling(self, x, offset):
        B, C, H, W = x.shape
        
        offset = torch.tanh(offset) * self.max_displacement
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        offset_grid = offset.permute(0, 2, 3, 1)
        deformed_grid = grid + offset_grid
        
        Z_B = F.grid_sample(
            x, 
            deformed_grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True
        )
        
        return Z_B
    
    def _process_feature(self, x, conv):
        Z_conv = conv(x)
        B, C, H, W = Z_conv.shape
        
        Z_reshaped = Z_conv.permute(0, 2, 3, 1).contiguous()
        Z_norm = self.norm(Z_reshaped)
        Z_ffn = self.ffn(Z_norm)
        
        Z_out = (Z_reshaped + Z_ffn).permute(0, 3, 1, 2)
        return Z_out
    
    def forward(self, R):
        self.intermediates['R'] = R.detach()
        
        ll, lh, hl, hh = self._dwt_decompose(R)
        
        offsets = self.offset_generator(R)
        offset_lh, offset_hl, offset_hh = torch.split(offsets, 2, dim=1)
        
        Z_ll = ll
        Z_lh = self._apply_deformable_sampling(lh, offset_lh)
        Z_hl = self._apply_deformable_sampling(hl, offset_hl)
        Z_hh = self._apply_deformable_sampling(hh, offset_hh)
        
        feat_ll = self._process_feature(Z_ll, self.conv_ll)
        feat_lh = self._process_feature(Z_lh, self.conv_lh)
        feat_hl = self._process_feature(Z_hl, self.conv_hl)
        feat_hh = self._process_feature(Z_hh, self.conv_hh)
        
        combined = torch.cat([feat_ll, feat_lh, feat_hl, feat_hh], dim=1)
        Z_conv = self.fusion(combined)
        
        B, C, H, W = Z_conv.shape
        Z_conv_reshaped = Z_conv.permute(0, 2, 3, 1).contiguous()
        Z_norm = self.norm(Z_conv_reshaped)
        Z_ffn = self.ffn(Z_norm)
        
        Phi_local = Z_conv + Z_ffn.permute(0, 3, 1, 2)
        
        return Phi_local

class ResiGuideModule(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.fgcm = FGCM(dim, dim)
        self.rlcm = RLCM(dim, dim)
        self.fusion = nn.Conv2d(dim * 2, out_dim, kernel_size=1)
        self.intermediates = {}
        
    def forward(self, y_hat_prev):
        Phi_global = self.fgcm(y_hat_prev)
        self.intermediates['Phi_global'] = Phi_global.detach()
        
        R = y_hat_prev - Phi_global
        self.intermediates['R'] = R.detach()
        
        Phi_local = self.rlcm(R)
        self.intermediates['Phi_local'] = Phi_local.detach()
        
        if hasattr(self.fgcm, 'intermediates'):
            for key, value in self.fgcm.intermediates.items():
                self.intermediates[key] = value
        
        if hasattr(self.rlcm, 'intermediates'):
            for key, value in self.rlcm.intermediates.items():
                if key not in self.intermediates:
                    self.intermediates[key] = value
        
        context_features = torch.cat([Phi_global, Phi_local], dim=1)
        output = self.fusion(context_features)
        
        return output


