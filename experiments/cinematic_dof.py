"""
Cinematic Depth of Field & Bokeh Engine (Lens Focus & Optical Realism)
- Çift Katmanlı Odak Derinliği (DoF): Arka planı gerçek 50mm f/1.4 lens bokeh'i ile eritir.
- Portre/Odak Alanı Koruması: Ana karakterin yüzünü, saç tellerini ve bakışlarını jilet gibi net tutar.
- Yarı saydam ateş/şömine ışığı difüzyonu ve sıcak ortam aydınlatması ekler.
"""

import torch
import torch.nn.functional as F

class CinematicLensBokeh:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
        # 7x7 Dairesel Bokeh Çekirdeği (Optik Diyafram Taklidi)
        k_size = 7
        y, x = torch.meshgrid(torch.linspace(-1, 1, k_size), torch.linspace(-1, 1, k_size), indexing='ij')
        dist = torch.sqrt(x**2 + y**2)
        # Dairesel diyafram bıçakları yumuşak geçişi
        bokeh_circle = torch.exp(- (dist**2) / 0.5)
        bokeh_circle[dist > 1.0] = 0.0
        bokeh_circle = bokeh_circle / bokeh_circle.sum()
        
        self.bokeh_kernel = bokeh_circle.view(1, 1, k_size, k_size).to(device, dtype)

    def apply_focus_and_bokeh(self, latent_tensor, focus_center=(0.55, 0.40), depth_falloff=1.8):
        """
        latent_tensor: [B, C, H, W] (32 channels)
        focus_center: (x, y) normalize koordinat - karakterin göz/yüz odak noktası.
        depth_falloff: Odak dışı alanların ne kadar eriyeceği (DoF derinliği).
        """
        B, C, H, W = latent_tensor.shape
        
        # Odak Haritası (Depth of Field Map)
        y = torch.linspace(0, 1, H, device=self.device, dtype=self.dtype).view(1, 1, H, 1).repeat(B, 1, 1, W)
        x = torch.linspace(0, 1, W, device=self.device, dtype=self.dtype).view(1, 1, 1, W).repeat(B, 1, H, 1)
        
        cx, cy = focus_center
        # Eliptik portre odak çemberi (karakterin yüzü ve gövdesi odakta kalır)
        dist_from_focus = torch.sqrt(((x - cx) / 0.45)**2 + ((y - cy) / 0.65)**2)
        
        # 0 = tam net (odak merkezi), 1 = tam bokeh (arka plan / şömine / kitaplık)
        blur_mask = torch.clamp((dist_from_focus - 0.35) * depth_falloff, 0.0, 1.0)
        
        # Arka plana optik bokeh evrişimi uygula
        weight = self.bokeh_kernel.repeat(C, 1, 1, 1)
        blurred_background = F.conv2d(latent_tensor, weight, padding=3, groups=C)
        
        # Odaklanan karakteri ve mikro detayları koruyarak arka planı erit
        rendered = latent_tensor * (1.0 - blur_mask * 0.45) + blurred_background * (blur_mask * 0.45)
        
        # Şömine ve ortam ışığı saçılımı (Warm Ambient Diffusion):
        # Sol alt/orta alandaki ateş ışığı için sıcak ton ışıma alanı
        fire_glow_y = (1.0 - y) * torch.exp(-((x - 0.15)**2) / 0.1)
        warm_ambience = fire_glow_y * 0.04
        rendered = rendered + warm_ambience
        
        return rendered
