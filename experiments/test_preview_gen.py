import torch
import numpy as np
from PIL import Image, ImageFilter
import io
import base64

def latents_to_preview_b64(latents, height, width, pipe):
    with torch.no_grad():
        lh = 2 * (int(height) // (pipe.vae_scale_factor * 2))
        lw = 2 * (int(width) // (pipe.vae_scale_factor * 2))
        h_half = lh // 2
        w_half = lw // 2
        
        # Reshape tokens to (B, C, H_half, W_half)
        # latents: [1, h_half*w_half, 128]
        if latents.ndim == 3:
            patches = latents.view(1, h_half, w_half, 128).permute(0, 3, 1, 2)
        else:
            patches = latents
            
        unpatch = pipe._unpatchify_latents(patches) # [1, 32, lh, lw]
        
        # Use first 3 channels or weighted combination for vibrant preview
        ch0 = unpatch[0, 0].float().cpu().numpy()
        ch1 = unpatch[0, 1].float().cpu().numpy()
        ch2 = unpatch[0, 2].float().cpu().numpy()
        
        rgb = np.stack([ch0, ch1, ch2], axis=-1)
        # normalize
        p_min, p_max = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - p_min) / (p_max - p_min + 1e-5), 0, 1) * 255.0
        
        img = Image.fromarray(rgb.astype(np.uint8))
        img = img.resize((width // 2, height // 2), Image.Resampling.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

print("latents_to_preview_b64 defined successfully")
