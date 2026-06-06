#include "relpos_attention.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>

// Equivalence test for the CHUNK-MATMUL banded attention
// (build_graph_local_chunked) against the trusted pad-and-shift banded path
// (forward_local, itself verified to 1.4e-3 vs a deterministic brute-force band
// reference). Both consume the SAME synthetic x / local pos_emb through the SAME
// layer-0 weights, so they must agree to within float reduction-order noise
// (matmul vs sum_rows-loop). This decouples the test from any baseline clip
// length, so we can exercise large windows (W up to 128 = NeMo's full
// [128,128]) - the whole point of the chunk-matmul construction.
//
// Env: PARAKEET_TEST_GGUF (weights; skip 77 if unset).
static void fill_synth(std::vector<float>& v, int rows, int d, int seed) {
    v.resize((size_t)rows * d);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < d; ++c)
            // smooth, bounded, distinct per (r,c,seed); not all-equal so banding matters
            v[(size_t)r * d + c] =
                0.5f * std::sin(0.017f * (r + 1) * (c + 1 + seed) + 0.3f * seed)
              + 0.1f * std::cos(0.0031f * (r * 7 + c * 3 + seed));
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const int d_model = (int)ml.config().d_model;
    pk::RelPosAttention attn(ml, /*layer_idx*/0);

    struct Case { int T, W, chunk; };
    const Case cases[] = {
        {  37,   8,  -1},   // small, default chunk; padded clip (valid_len=T-1)
        {  64,  16,  16},   // T multiple of chunk
        {  70,  16,  16},   // T NOT a multiple of chunk (padding path)
        { 200,  32,  32},   // production cap window
        { 256, 128,  64},   // NeMo full window, chunk<W
        { 333, 128, 128},   // large W, chunk==W, ragged T
    };

    bool all_ok = true;
    for (const Case& c : cases) {
        const int T = c.T, W = c.W, pos_len = 2 * W + 1;
        const int valid_len = T - 1; // last frame is pad, like the real encoder
        std::vector<float> x, pos;
        fill_synth(x, T, d_model, /*seed*/1 + (W & 7));
        fill_synth(pos, pos_len, d_model, /*seed*/2);

        std::vector<float> ref, got;
        attn.forward_local(x, T, pos, pos_len, valid_len, W, W, ref);
        attn.forward_local_chunked(x, T, pos, pos_len, valid_len, W, W, got, c.chunk);

        char label[64];
        std::snprintf(label, sizeof(label), "chunked.T%d_W%d_c%d", T, W, c.chunk);
        all_ok &= pktest::compare(got, ref, label, /*atol*/2e-3f, /*rtol*/2e-3f);
    }
    return all_ok ? 0 : 1;
}
