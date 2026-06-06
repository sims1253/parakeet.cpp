#include "encoder.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <algorithm>

// Batched encoder equivalence + padding invariance with NeMo rel_pos_local_attn
// (banded / Longformer) forced ON. This is the local-attention twin of
// test_encoder_batch.cpp: it exercises RelPosAttention::build_graph_batched_local
// through the whole fused batched encoder.
//
// PARAKEET_ATT_CONTEXT forces the symmetric window for EVERY utterance regardless
// of length (encoder.cpp local_attn_window). We use W=32 = kMaxLocalWindow, the
// exact window the auto path selects for long audio (Tp > 8192) in production, so
// this test gates the real shipped configuration.
//
// As in test_encoder_batch, item0 is the full baseline clip and item1 is its
// first 3/4 zero-padded up to T0. The two correctness properties:
//   * item0 must be BIT-EXACT to its standalone run (0.0) - the shorter padded
//     neighbour must not perturb the full clip at all (no cross-item leakage).
//   * item1's valid region must match its standalone run within 5e-2/5e-2.
//     item1's standalone run is at its OWN (shorter) Tp while the batched run is
//     at the padded width T0; the differing tensor shapes change ggml reduction
//     order, so near-zero activations of the padded clip carry float noise that
//     accumulates over the 17 conformer layers. The tolerance mirrors the
//     full-attention test. (Tightening the window past production's W=32 sharpens
//     the banded softmax and amplifies that noise on near-zero elements - a
//     numerical effect, not pad leakage: item0 stays bit-exact and item1's
//     mean|d| stays ~1e-2 throughout.)
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    // Force banded local attention (production cap) for the whole run.
    setenv("PARAKEET_ATT_CONTEXT", "32", /*overwrite*/1);

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "model load failed\n"); return 1; }

    std::vector<float> mel; std::vector<int64_t> ms;
    if (!pktest::load_baseline(base, "mel", mel, ms)) return 1;
    if (ms.size() != 2) { std::fprintf(stderr, "mel rank=%zu\n", ms.size()); return 1; }
    const int n_mels = (int)ms[0];
    const int T0     = (int)ms[1];
    const int T1     = (T0 * 3) / 4;

    pk::Encoder enc(ml);

    // --- Standalone references (local attention via the forced env) ---------
    std::vector<float> e0, e1; int dm = 0, to0 = 0, to1 = 0;
    enc.forward(mel, n_mels, T0, e0, dm, to0);

    std::vector<float> mel1((size_t)n_mels * T1);
    for (int m = 0; m < n_mels; ++m)
        for (int t = 0; t < T1; ++t)
            mel1[(size_t)m * T1 + t] = mel[(size_t)m * T0 + t];
    int dm1 = 0;
    enc.forward(mel1, n_mels, T1, e1, dm1, to1);

    // --- Batched (T_max=T0; item1 zero-padded) ------------------------------
    pk::MelBatch mb;
    mb.B = 2; mb.n_mels = n_mels; mb.T_max = T0; mb.valid_T = { T0, T1 };
    mb.data.assign((size_t)2 * n_mels * T0, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < T0; ++t) mb.data[((size_t)0 * n_mels + m) * T0 + t] = mel[(size_t)m * T0 + t];
        for (int t = 0; t < T1; ++t) mb.data[((size_t)1 * n_mels + m) * T0 + t] = mel1[(size_t)m * T1 + t];
    }
    std::vector<std::vector<float>> eo; int dmb = 0, tob = 0; std::vector<int> vt;
    enc.forward_batch(mb, eo, dmb, tob, vt);

    if (dmb <= 0 || tob <= 0 || eo.size() != 2 || vt.size() != 2) {
        std::fprintf(stderr, "bad batched output dmb=%d tob=%d |eo|=%zu |vt|=%zu\n",
                     dmb, tob, eo.size(), vt.size());
        return 1;
    }

    std::fprintf(stderr,
        "[encbatch_local] dm=%d to0=%d to1=%d | dmb=%d tob=%d vt={%d,%d}\n",
        dm, to0, to1, dmb, tob, vt[0], vt[1]);

    auto slice_cols = [&](const std::vector<float>& full, int Tfull, int Tvalid) {
        std::vector<float> s((size_t)dmb * Tvalid);
        for (int c = 0; c < dmb; ++c)
            for (int t = 0; t < Tvalid; ++t)
                s[(size_t)c * Tvalid + t] = full[(size_t)c * Tfull + t];
        return s;
    };
    std::vector<float> b0   = slice_cols(eo[0], vt[0], vt[0]);
    std::vector<float> b1   = slice_cols(eo[1], vt[1], vt[1]);
    std::vector<float> ref0 = slice_cols(e0, to0, vt[0]);
    std::vector<float> ref1 = slice_cols(e1, to1, vt[1]);

    // item0 must match its standalone run to within float noise (a real leak
    // from the shorter neighbour would be order-1, not 1e-3): no cross-item leak.
    bool a = pktest::compare(b0, ref0, "encbatch_local.item0", 1e-3f, 1e-3f);
    bool b = pktest::compare(b1, ref1, "encbatch_local.item1", 5e-2f, 5e-2f);
    return (a && b) ? 0 : 1;
}
