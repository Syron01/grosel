"""
Anti-AI Artifact & Radiant Manifold Physics Kernel (PT-RMF)
FLUX.2 Klein mimarisi 3D latent tensör formatı kullanır: [B, Tokens, Channels] veya [B, N, C].
Burada tokenlar uzamsal grid'e (2D manifold) dönüştürülüp fiziksel optik işlem yapılır,
ardından geri paketlenir.
"""

import math
import torch
import torch.nn.functional as F

class RadiantManifoldPhysics:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
        self.sigma_epidermis = 1.2
        self.sigma_dermis = 2.8
        self._init_gaussian_kernels()

    def _init_gaussian_kernels(self):
        kernel_size = 5
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        
        g_epi = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma_epidermis**2))
        g_epi = g_epi / g_epi.sum()
        
        g_der = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma_dermis**2))
        g_der = g_der / g_der.sum()
        
        self.kernel_epi = g_epi.view(1, 1, kernel_size, kernel_size).to(self.device, self.dtype)
        self.kernel_der = g_der.view(1, 1, kernel_size, kernel_size).to(self.device, self.dtype)

    def _to_spatial_4d(self, tensor):
        """
        Gelen tensör 3D [B, N, C] ise [B, C, H, W] uzamsal grid formatına çevirir.
        """
        is_3d = (tensor.dim() == 3)
        if is_3d:
            B, N, C = tensor.shape
            H = int(math.isqrt(N))
            W = N // H
            tensor_4d = tensor.transpose(1, 2).contiguous().view(B, C, H, W)
            return tensor_4d, is_3d, (B, N, C)
        return tensor, False, tensor.shape

    def _from_spatial_4d(self, tensor_4d, is_3d, orig_shape):
        if is_3d:
            B, N, C = orig_shape
            return tensor_4d.view(B, C, N).transpose(1, 2).contiguous()
        return tensor_4d

    def apply_subsurface_scattering(self, latent_tensor, strength=0.15):
        """
        BSSRDF Deri Altı Işık Saçılımı: Sıcak ve gerçekçi ten geçirgenliği.
        """
        tensor_4d, is_3d, orig_shape = self._to_spatial_4d(latent_tensor)
        B, C, H, W = tensor_4d.shape
        
        weight_epi = self.kernel_epi.repeat(C, 1, 1, 1)
        weight_der = self.kernel_der.repeat(C, 1, 1, 1)
        
        epi = F.conv2d(tensor_4d, weight_epi, padding=2, groups=C)
        der = F.conv2d(tensor_4d, weight_der, padding=2, groups=C)
        
        scattered = tensor_4d + (epi - der) * strength
        return self._from_spatial_4d(scattered, is_3d, orig_shape)

    def apply_path_traced_radiance(self, latent_tensor, light_direction=(0.6, -0.8), ao_strength=0.10):
        """
        RTX Path-Tracing Benzetimli Yönlü Işıklandırma ve Ortam Kapanması (AO).
        """
        tensor_4d, is_3d, orig_shape = self._to_spatial_4d(latent_tensor)
        
        dx = tensor_4d[:, :, :, 2:] - tensor_4d[:, :, :, :-2]
        dx = F.pad(dx, (1, 1, 0, 0))
        
        dy = tensor_4d[:, :, 2:, :] - tensor_4d[:, :, :-2, :]
        dy = F.pad(dy, (0, 0, 1, 1))
        
        lx, ly = light_direction
        norm = (lx**2 + ly**2)**0.5
        lx, ly = lx / norm, ly / norm
        
        directional_shading = lx * dx + ly * dy
        
        laplacian = (dx[:, :, :, 1:] - dx[:, :, :, :-1])
        laplacian = F.pad(laplacian, (1, 0, 0, 0))
        
        radiant_correction = (directional_shading * 0.12) - (torch.clamp(laplacian, min=0.0) * ao_strength)
        refined = tensor_4d + radiant_correction
        return self._from_spatial_4d(refined, is_3d, orig_shape)

    def apply_poisson_skin_microtexture(self, latent_tensor, texture_intensity=0.05):
        """
        Poisson-Disk Organik Doku Süzgeci: Plastik AI hissini yok eden mikro detay.
        """
        tensor_4d, is_3d, orig_shape = self._to_spatial_4d(latent_tensor)
        B, C, H, W = tensor_4d.shape
        
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=self.dtype, device=self.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        
        high_freq = F.conv2d(tensor_4d, kernel, padding=1, groups=C)
        noise = torch.randn_like(tensor_4d)
        modulator = torch.sigmoid(torch.abs(high_freq) * 3.5)
        
        organic_grain = noise * modulator * texture_intensity
        refined = tensor_4d + organic_grain
        return self._from_spatial_4d(refined, is_3d, orig_shape)
