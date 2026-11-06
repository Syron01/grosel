"""Karo sözleşmesi her VAE API'sine kurulur — motoru açılışta çökmeden.

diffusers'ta iki yüzey var: Qwen `enable_tiling(min, stride)` alıyor, FLUX2 ise
argsız `enable_tiling()` + `tile_sample_min_size`/`tile_overlap_factor` özniteliği
kullanıyor. `enable_tiling(**kwargs)` FLUX2'de TypeError ile motoru çökertiyordu;
bu test gerçek sınıfları ağırlıksız, küçük konfigürasyonla kurduğu için GPU'suz
yakalar. Telemetri de kurulandan yazılır: sözleşme değeri ile fiilî değer ayrı.
"""

import pytest

diffusers = pytest.importorskip("diffusers")

from web_app import quality_math as qm  # noqa: E402


def _flux2_vae():
    return diffusers.AutoencoderKLFlux2(
        block_out_channels=(32, 64, 128, 128), latent_channels=4, sample_size=128,
        layers_per_block=1, norm_num_groups=8,
        use_quant_conv=False, use_post_quant_conv=False,
    )


def _qwen_vae():
    return diffusers.AutoencoderKLQwenImage21(
        base_dim=8, dim_mult=[1, 1], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[False],
    )


class _BareVae:
    """Karo geometrisi olmayan VAE: iddia kurulmaz, None döner."""

    def enable_tiling(self):
        self.use_tiling = True


@pytest.mark.parametrize("factory", [_flux2_vae, _qwen_vae], ids=["flux2", "qwen"])
def test_karo_onusmesi_sozlesmeye_cekilir(factory):
    vae = factory()
    blend = qm.apply_tiling(vae)
    assert vae.use_tiling is True
    assert blend == qm.blend_px() == qm.TILE_MIN_PX - qm.TILE_STRIDE_PX
    assert qm.blend_px_of(vae) == blend, "telemetri nesneden okunmalı"
    assert not qm.plan_invariants(vae_blend_px=blend)[0], "kurulan geometri sözleşmeyi ihlal etmez"


def test_latent_izgara_ornekliden_yenilir():
    """Karo küçülürken örnek/piksel oranı bozulursa diki değil, kayma çıkar."""
    vae = _flux2_vae()
    ratio = vae.tile_sample_min_size / vae.tile_latent_min_size
    qm.apply_tiling(vae)
    assert vae.tile_sample_min_size / vae.tile_latent_min_size == ratio


def test_qwen_varsayilani_diki_esiginde_diye_bilinir():
    """diffusers'ın Qwen varsayılanı 256/192 = 64 px; ölçülen diki kusurunun kaynağı."""
    vae = _qwen_vae()
    assert qm.blend_px_of(vae) == vae.tile_sample_min_height - vae.tile_sample_stride_height
    assert qm.blend_px_of(vae) < qm.MIN_BLEND_PX
    assert qm.plan_invariants(vae_blend_px=qm.blend_px_of(vae))[0], "dar örtüşme ihlal sayılmalı"


def test_geometrisi_bilinmeyen_vaede_iddia_kurulmaz():
    assert qm.apply_tiling(_BareVae()) is None
    assert qm.blend_px_of(_BareVae()) is None
