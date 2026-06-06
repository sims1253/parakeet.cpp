#!/usr/bin/env python3
"""Generate the DETERMINISTIC brute-force band-attention reference fixture used by
``test_relpos_attention_local``.

NeMo's ``RelPositionMultiHeadAttentionLongformer.forward`` is non-deterministic
on short sequences (``sliding_chunks_matmul_pv`` reads uninitialized memory at
sequence boundaries via ``F.pad(value=-1)`` + ``as_strided`` — two identical
forward() calls differ by >1e3). So a hook-captured ``l0_attn_out`` baseline is
unusable for bit-parity on a short clip. Instead we recompute ``l0_attn_out`` as
plain band attention (the well-defined math the longformer approximates), which
the C++ ``forward_local`` matches to ~1e-3. End-to-end NeMo quality is anchored
separately by the long-audio WER capstone, where the boundary noise is moot.

Reads ``l0_attn_in`` and ``pos_emb`` from an existing local baseline (produced by
``gen_nemo_baseline.py --att-context-size W``) so the inputs are the real NeMo
ones; only the reference output is recomputed deterministically.

Usage:
    python scripts/gen_band_ref.py --model nvidia/parakeet-tdt_ctc-110m \
        --in-baseline baseline_110m_local8.gguf --att-context 8 \
        --output baseline_110m_local8_ref.gguf
"""
import argparse
import math

import gguf
import numpy as np
import torch

import nemo.collections.asr as nemo_asr


def read_tensor(path, name):
    r = gguf.GGUFReader(path)
    t = {x.name: x for x in r.tensors}[name]
    return np.array(t.data, dtype=np.float32).reshape(
        tuple(int(d) for d in reversed(t.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt_ctc-110m")
    ap.add_argument("--in-baseline", required=True,
                    help="local baseline gguf with l0_attn_in + pos_emb")
    ap.add_argument("--att-context", type=int, required=True, help="window W")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    li_np = read_tensor(args.in_baseline, "l0_attn_in")     # (T, D)
    pos_np = read_tensor(args.in_baseline, "pos_emb")        # (2W+1, D)
    li = torch.tensor(li_np)[None]
    pos = torch.tensor(pos_np)[None]
    T, D = li.shape[1], li.shape[2]
    w = args.att_context
    vlen = T - 1  # last frame is a center-pad/padding frame for the fixture clip

    m = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu")
    m.eval()
    m.change_attention_model("rel_pos_local_attn", [w, w])
    a0 = m.encoder.layers[0].self_attn
    h, dk = a0.h, a0.d_k
    s = math.sqrt(dk)
    P = 2 * w + 1

    with torch.no_grad():
        q = a0.linear_q(li).view(1, T, h, dk).transpose(1, 2)
        k = a0.linear_k(li).view(1, T, h, dk).transpose(1, 2)
        v = a0.linear_v(li).view(1, T, h, dk).transpose(1, 2)
        p = a0.linear_pos(pos).view(1, -1, h, dk).transpose(1, 2)
        qu = q + a0.pos_bias_u.unsqueeze(1)
        qv = q + a0.pos_bias_v.unsqueeze(1)
        sc = torch.full((1, h, T, P), -1e30)
        for t in range(T):
            for c in range(P):
                key = t - w + c
                if 0 <= key < vlen:
                    sc[0, :, t, c] = ((qu[0, :, t] * k[0, :, key]).sum(-1)
                                      + (qv[0, :, t] * p[0, :, c]).sum(-1)) / s
        at = torch.softmax(sc, dim=-1)
        ctx = torch.zeros(1, h, T, dk)
        for t in range(T):
            for c in range(P):
                key = t - w + c
                if 0 <= key < vlen:
                    ctx[0, :, t] += at[0, :, t, c:c + 1] * v[0, :, key]
        om = a0.linear_out(ctx.transpose(1, 2).reshape(1, T, h * dk))[0].clone()
        om[vlen:] = a0.linear_out(torch.zeros(1, h * dk))[0]  # padded query rows -> bias

    ref = om.numpy().astype(np.float32)
    W = gguf.GGUFWriter(args.output, "pk-band-ref")
    W.add_tensor("l0_attn_in", np.ascontiguousarray(li_np))
    W.add_tensor("pos_emb", np.ascontiguousarray(pos_np))
    W.add_tensor("l0_attn_out", np.ascontiguousarray(ref))
    W.write_header_to_file()
    W.write_kv_data_to_file()
    W.write_tensors_to_file()
    W.close()
    print(f"wrote {args.output}: T={T} D={D} W={w}")


if __name__ == "__main__":
    main()
