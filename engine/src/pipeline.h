// Pre/post-processing for depth inference. Runs on the caller (render) thread; only the model run
// itself is handed to the inference worker.
#pragma once

#include "dms/engine.h"

#include <cstdint>
#include <vector>

namespace dms {

struct FrameView {
    const float* rgb = nullptr;  // interleaved RGB
    int width = 0;
    int height = 0;
    int stride = 0;  // floats per row
};

// A source-space crop of a frame (tiling). x0,y0 inclusive, x1,y1 exclusive.
struct Crop {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    int width() const { return x1 - x0; }
    int height() const { return y1 - y0; }
};

// Test-time augmentation for sample k of n: horizontal flip and sub-pixel shift (in inference pixels).
struct Augment {
    bool flip = false;
    float shift_x = 0.f;
    float shift_y = 0.f;
};
Augment augment_for_sample(int k, int n) noexcept;

// Writes one NCHW-normalized image (3*ih*iw floats) for a crop of the frame.
void preprocess_crop(const FrameView& f, const Crop& c, int iw, int ih, const Augment& a, float* dst);

// Undo augmentation on an inference-resolution depth map (in place).
void deaugment(float* depth, int w, int h, const Augment& a);

}  // namespace dms
