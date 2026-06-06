#include "relpos_attention.hpp"
#include "model_loader.hpp"
#include "ggml_graph.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>

// Memory-scaling test for banded local attention.
//
// NeMo rel_pos_local_attn / forward_local must use O(T * window) memory, NOT the
// O(T^2) of full attention (the bug that OOM'd long-audio transcription). We run
// forward_local at T and 2T with a fixed window and assert the gallocr buffer
// grows ~linearly (ratio < 3) and stays far below a single full [T, T, H] score
// tensor — neither of which a quadratic implementation could satisfy. Values are
// irrelevant (graph allocation depends only on shapes), so zero inputs are fine.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "PARAKEET_TEST_GGUF not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    const int D = (int)ml.config().d_model;
    const int H = (int)ml.config().n_heads;
    const int W = 8, P = 2 * W + 1;

    pk::RelPosAttention attn(ml, /*layer_idx*/0);
    auto measure = [&](int T) -> size_t {
        std::vector<float> x((size_t)T * D, 0.0f), pos((size_t)P * D, 0.0f), out;
        attn.forward_local(x, T, pos, P, /*valid_len*/T, W, W, out);
        return pk::last_graph_alloc_bytes();
    };

    const int T0 = 512, T1 = 1024;
    const size_t a0 = measure(T0);
    const size_t a1 = measure(T1);  // persistent gallocr grows to the T1 high-water
    const double ratio = a0 > 0 ? (double)a1 / (double)a0 : 1e9;

    // A single full [T1, T1, H] f32 score tensor — the floor of any O(T^2) path.
    const size_t full_scores = (size_t)T1 * T1 * H * sizeof(float);

    std::fprintf(stderr,
        "alloc(T=%d)=%zu  alloc(T=%d)=%zu  ratio=%.2f  (full T^2 scores=%zu)\n",
        T0, a0, T1, a1, ratio, full_scores);

    bool ok = (a0 > 0) && (ratio < 3.0) && (a1 < full_scores);
    std::fprintf(stderr, "[banded memory-scaling] %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
