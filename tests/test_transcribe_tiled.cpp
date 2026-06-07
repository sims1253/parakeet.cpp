#include "model.hpp"
#include "audio_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

// End-to-end faithfulness test for the long-audio subsampling tiling path.
//
// Tiling the subsampling stage (Encoder::forward_batch_tiled) is engineered to be
// numerically faithful to the fused encoder (Encoder::forward_batch). This test
// runs a real speech clip through the FULL batched pipeline twice:
//
//   1. fused   (PARAKEET_SUBSAMPLING_TILE unset -> auto, short clip stays fused)
//   2. tiled   (PARAKEET_SUBSAMPLING_TILE=17 -> forces many small subsampling tiles)
//
// and asserts the transcripts are IDENTICAL. A faithful tiling must decode a clean
// clip to exactly the same tokens. Self-contained: only needs PARAKEET_TEST_GGUF
// plus the committed tests/fixtures/speech.wav.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "test_transcribe_tiled: PARAKEET_TEST_GGUF not set; skip\n");
        return 77;
    }

    std::unique_ptr<pk::Model> model = pk::Model::load(gguf);
    if (!model) {
        std::fprintf(stderr, "test_transcribe_tiled: failed to load model %s\n", gguf);
        return 1;
    }

    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
        std::fprintf(stderr, "test_transcribe_tiled: failed to load tests/fixtures/speech.wav\n");
        return 1;
    }

    auto run = [&]() -> std::string {
        std::vector<std::vector<float>> batch{ audio.samples };
        return model->transcribe_pcm_batch(batch, 16000).at(0);
    };
    // Single-clip path (CLI / path-based C-API route through transcribe_path ->
    // transcribe_16k). Same clip, same fused-vs-tiled comparison.
    auto run_single = [&]() -> std::string {
        return model->transcribe_path("tests/fixtures/speech.wav");
    };

    std::string fused, tiled, sfused, stiled;
    try {
        // 1) fused (env unset -> auto; short clip below threshold stays fused)
        unsetenv("PARAKEET_SUBSAMPLING_TILE");
        fused = run();
        sfused = run_single();

        // 2) forced tiled path (small tile -> many tiles even on a short clip)
        setenv("PARAKEET_SUBSAMPLING_TILE", "17", 1);
        tiled = run();
        stiled = run_single();
        unsetenv("PARAKEET_SUBSAMPLING_TILE");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe_tiled: transcribe threw: %s\n", e.what());
        return 1;
    }

    if (fused.empty()) {
        std::fprintf(stderr,
            "test_transcribe_tiled: fused transcript is EMPTY (vacuous test)\n");
        return 1;
    }

    if (fused != tiled) {
        std::fprintf(stderr, "tiled transcript differs:\n fused=[%s]\n tiled=[%s]\n",
                     fused.c_str(), tiled.c_str());
        return 1;
    }

    if (sfused.empty()) {
        std::fprintf(stderr,
            "test_transcribe_tiled: single-clip fused transcript is EMPTY (vacuous)\n");
        return 1;
    }

    if (sfused != stiled) {
        std::fprintf(stderr,
            "single-clip tiled transcript differs:\n fused=[%s]\n tiled=[%s]\n",
            sfused.c_str(), stiled.c_str());
        return 1;
    }

    std::printf("transcribe tiled==fused: [%s]\n", fused.c_str());
    std::printf("single-clip tiled==fused: [%s]\n", sfused.c_str());
    return 0;
}
