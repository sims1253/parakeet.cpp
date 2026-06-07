#include "encoder.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
// Self-consistency: forward_batch_tiled (subsampling done per-item via the tiled
// path, then the post-subsampling graph) must match forward_batch (one fused
// graph) for B=1, within the valid region. The metric is per-frame RELATIVE
// (benign float reorder across tiled subsampling + 24 conformer layers is O(1e-3);
// a layout/injection/valid-length bug is O(1)).
//
// NOTE on input: we use a SMOOTH, log-mel-shaped signal (sinusoids), not uniform
// random noise. A uniform-random [-1,1] "mel" is wildly out-of-distribution for
// the encoder: full self-attention + 24 layers amplify even the inherent
// batched-vs-single graph float reorder to ~5e-2 at a few edge frames. That
// amplification is NOT a tiling bug -- the pre-existing forward_batch(B=1) vs
// forward() differ by the same ~5e-2 at the same frame on random input. A
// realistic (smooth) input keeps the amplification benign (~1e-3), so the gate
// below cleanly separates a correct refactor from an O(1) layout bug.
int main(){
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if(!gguf){ std::fprintf(stderr,"env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if(!ml.load(gguf)) return 1;
    pk::Encoder enc(ml);
    const int n_mels = (int)ml.config().n_mels;
    const int T = 4000;
    // build a 1-item MelBatch with a deterministic, smooth log-mel-shaped signal
    pk::MelBatch mb; mb.B=1; mb.n_mels=n_mels; mb.T_max=T; mb.valid_T={T};
    mb.data.assign((size_t)n_mels*T, 0.f);
    for (int m=0;m<n_mels;++m)
        for (int t=0;t<T;++t)
            mb.data[(size_t)m*T+t] = -4.0f + 3.0f*std::sin(0.01f*t + 0.2f*m)
                                          + 0.5f*std::sin(0.003f*t*(m+1));
    std::vector<std::vector<float>> e_ref, e_tiled; int dmr=0,Tr=0,dmt=0,Tt=0;
    std::vector<int> vr, vt;
    enc.forward_batch(mb, e_ref, dmr, Tr, vr);
    enc.forward_batch_tiled(mb, e_tiled, dmt, Tt, vt, /*tile_out_frames*/17);
    if (vr.size()!=vt.size() || vr[0]!=vt[0] || dmr!=dmt){ std::fprintf(stderr,"shape/valid mismatch v=%d/%d dm=%d/%d\n",vr.empty()?-1:vr[0],vt.empty()?-1:vt[0],dmr,dmt); return 1; }
    const int tv = vr[0];
    // enc_outs[0] is channels-first [d_model, tv]: index c*tv + t
    double maxabs=0, worstrel=0;
    for (int t=0;t<tv;++t){ double dmax=0,fmax=0;
        for (int c=0;c<dmr;++c){ double ref=e_ref[0][(size_t)c*tv+t];
            double d=std::fabs((double)e_tiled[0][(size_t)c*tv+t]-ref);
            if(d>maxabs)maxabs=d; if(d>dmax)dmax=d; if(std::fabs(ref)>fmax)fmax=std::fabs(ref);}
        double rel=dmax/(fmax+1e-6); if(rel>worstrel)worstrel=rel; }
    std::printf("forward_batch_tiled maxabs=%.3e worstrel=%.3e (tv=%d dm=%d)\n",maxabs,worstrel,tv,dmr);
    if (worstrel > 2e-2){ std::fprintf(stderr,"encoder tiled parity FAIL rel=%.3e\n",worstrel); return 1; }
    return 0;
}
