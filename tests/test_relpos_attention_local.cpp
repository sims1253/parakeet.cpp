#include "relpos_attention.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// Parity test for NeMo LOCAL (Longformer) attention —
// RelPositionMultiHeadAttentionLongformer, i.e.
// change_attention_model("rel_pos_local_attn", [W, W]).
//
// Under local attention pos_emb is [2W+1, d_model] (NOT the full 2T-1), and
// each query attends only to keys in [t-W, t+W]; forward_local computes this in
// O(T*window) instead of O(T^2).
//
// IMPORTANT — the reference is a DETERMINISTIC brute-force band attention (same
// inputs + model weights), NOT NeMo's raw longformer output. NeMo's
// sliding_chunks_matmul_pv reads uninitialized memory at sequence boundaries
// (F.pad value=-1 + as_strided), so on a short clip its output is
// non-deterministic (verified: two identical forward() calls differ by >1e3).
// The C++ banded math was verified to match NeMo's deterministic pieces directly
// (sliding_chunks_matmul_qk key=t-W+c to 1e-6; scores to 1e-4; pv key map to 0).
// End-to-end NeMo quality is anchored separately by the long-audio WER capstone,
// where the boundary noise is negligible.
//
// Env:
//   PARAKEET_TEST_GGUF            model weights (skip 77 if unset)
//   PARAKEET_TEST_BASELINE_LOCAL  local-attention baseline gguf (skip 77 if unset)
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_LOCAL");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // Attention input: baseline "l0_attn_in" is [T, d_model] row-major.
    std::vector<float> xin; std::vector<int64_t> xshape;
    if (!pktest::load_baseline(base, "l0_attn_in", xin, xshape)) return 1;
    if (xshape.size() != 2) { std::fprintf(stderr, "l0_attn_in rank=%zu\n", xshape.size()); return 1; }
    const int T       = (int)xshape[0];
    const int d_model = (int)xshape[1];

    // Local relative positional encoding: baseline "pos_emb" is [2W+1, d_model].
    std::vector<float> pos; std::vector<int64_t> pshape;
    if (!pktest::load_baseline(base, "pos_emb", pos, pshape)) return 1;
    if (pshape.size() != 2 || (int)pshape[1] != d_model) {
        std::fprintf(stderr, "pos_emb shape=[%lld,%lld] expected [*, %d]\n",
                     (long long)pshape[0], (long long)pshape[1], d_model);
        return 1;
    }
    const int pos_len = (int)pshape[0];
    if (pos_len % 2 == 0) {
        std::fprintf(stderr, "local pos_len=%d is not odd (expected 2W+1)\n", pos_len);
        return 1;
    }
    const int W = (pos_len - 1) / 2;
    if (W >= T) {
        std::fprintf(stderr, "W=%d >= T=%d: window covers the full sequence, "
                     "banding is not exercised — regenerate the baseline with a "
                     "smaller --att-context-size\n", W, T);
        return 1;
    }

    // Last frame is a center-pad/padding frame (same clip + preprocessing as the
    // full-attention baseline), so valid frames are 0..T-2.
    const int valid_len = T - 1;

    pk::RelPosAttention attn(ml, /*layer_idx*/0);
    std::vector<float> out;
    attn.forward_local(xin, T, pos, pos_len, valid_len, /*att_left*/W, /*att_right*/W, out);

    // Reference: l0_attn_out is [T, d_model] row-major.
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "l0_attn_out", ref, rshape)) return 1;
    if (rshape.size() != 2 || (int)rshape[0] != T || (int)rshape[1] != d_model) {
        std::fprintf(stderr, "l0_attn_out shape=[%lld,%lld] expected [%d,%d]\n",
                     (long long)rshape[0], (long long)rshape[1], T, d_model);
        return 1;
    }

    bool ok = pktest::compare(out, ref, "relpos_attention_local", /*atol*/2e-2f, /*rtol*/2e-2f);
    return ok ? 0 : 1;
}
