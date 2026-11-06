"""
HyperFlux LowVRAM Engine — blok akitmali (block-streamed) FP8 inference.

Iki model de (transformer 3.9 GB + text encoder 4.0 GB) VRAM'de tutulmaz.
Agirliklar pinli host bellekte durur; her blok calismadan hemen once tek bir
yeniden kullanilan GPU tamponuna kopyalanir. Boylece tepe VRAM blok boyutu +
aktivasyonlarla sinirli kalir (~1 GB bandi), ama 1344x768 gibi yuksek taban
cozunurlukler ondeki modellerin anatomi/perspektif hatalarini duzeltir.

Text encoder'in 778 MB'lik vocabulary embedding'i de CPU'da kalir; sadece
kullanilan token satirlari GPU'ya tasinir.
"""

import gc
import torch
import torch.nn as nn

from web_app.fast_engine import FastKleinEngine, free_vram_gb
from web_app.fp8_layers import convert_to_fp8_native, FP8Linear, FP8NativeLinear


def _align(n, a=8):
    return (n + a - 1) // a * a


class BlockStreamer:
    """Bir ModuleList'i tek GPU tamponu uzerinden akitir (sadece H2D, geri kopya yok)."""

    def __init__(self, blocks, device="cuda"):
        self.device = device
        self.entries = []
        max_bytes = 0
        for blk in blocks:
            items = []
            for _, t in list(blk.named_parameters()) + list(blk.named_buffers()):
                host = t.detach().to("cpu").pin_memory()
                items.append((t, host))
            size = 0
            for _, host in items:
                size = _align(size) + host.numel() * host.element_size()
            max_bytes = max(max_bytes, _align(size) + 64)
            self.entries.append((blk, items))
        self.buf = torch.empty(max_bytes, dtype=torch.uint8, device=device)
        for blk, items in self.entries:
            blk.register_forward_pre_hook(self._pre_hook(items))

    def _pre_hook(self, items):
        def hook(_module, _args):
            off = 0
            for t, host in items:
                nb = host.numel() * host.element_size()
                view = self.buf[off:off + nb].view(host.dtype).view(host.shape)
                view.copy_(host, non_blocking=True)
                t.data = view
                off = _align(off + nb)
        return hook

    def bytes(self):
        return self.buf.numel()


class _DummyLMHead(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)


class CpuEmbedding(nn.Module):
    """Buyuk vocabulary embedding'i CPU'da tutar, yalnizca secilen satirlari tasir."""

    def __init__(self, weight):
        super().__init__()
        self.weight_cpu = weight.detach().to("cpu").pin_memory()
        self.num_embeddings, self.embedding_dim = self.weight_cpu.shape

    def forward(self, input_ids):
        ids = input_ids.flatten().to("cpu")
        rows = torch.index_select(self.weight_cpu, 0, ids)
        out = rows.to("cuda", non_blocking=True)
        return out.view(*input_ids.shape, self.embedding_dim)


class LowVramEngine(FastKleinEngine):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # FastKleinEngine.__init__'i bilerek cagirmiyoruz: yerlesim plani farkli.
        from diffusers import Flux2KleinPipeline
        from web_app.config import FLUX_MODEL

        torch.cuda.empty_cache()
        gc.collect()

        self.pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL, torch_dtype=torch.bfloat16, local_files_only=True
        )
        self.transformer = self.pipe.transformer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.native_fp8 = True
        self.n_layers = convert_to_fp8_native(self.transformer)
        self.n_te_layers = convert_to_fp8_native(self.text_encoder)

        # Buyuk vocabulary tablosunu CPU'da birak (778 MB). Pipeline sadece ara
        # katman hidden state'lerini kullaniyor; logits'e ihtiyac yok.
        te = self.text_encoder
        if hasattr(te, "lm_head") and te.lm_head is not None:
            te.lm_head = _DummyLMHead()
        if hasattr(te, "model") and hasattr(te.model, "embed_tokens"):
            emb = te.model.embed_tokens
            te.model.embed_tokens = CpuEmbedding(emb.weight)
        self._emb_cache = {}
        self._dual_resident = True
        self._te_resident = True
        self._tf_resident = True

        # Kucuk moduller VRAM'de, bloklar akitilir
        tf_blocks = list(self.transformer.transformer_blocks) + list(self.transformer.single_transformer_blocks)
        for _, child in self.transformer.named_children():
            if child not in (self.transformer.transformer_blocks, self.transformer.single_transformer_blocks):
                child.to("cuda")
        te_layers = list(te.model.layers)
        for name, child in te.named_children():
            if name != "model":
                child.to("cuda")
        for name, child in te.model.named_children():
            if name not in ("layers", "embed_tokens"):
                child.to("cuda")
        self.vae.to("cuda")

        self.tf_stream = BlockStreamer(tf_blocks)
        self.te_stream = BlockStreamer(te_layers)

        self._warmup()
        print(
            f"[lowvram] FP8 native tf={self.n_layers} te={self.n_te_layers} | "
            f"blok tamponu tf={self.tf_stream.bytes()/1024**2:.0f}MB te={self.te_stream.bytes()/1024**2:.0f}MB | "
            f"serbest VRAM {free_vram_gb():.2f} GB"
        )

    def _te_up(self):
        pass

    def _te_down(self):
        pass

    def _tf_up(self):
        pass

    def _tf_down(self):
        pass
