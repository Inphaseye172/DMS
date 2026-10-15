#pragma once
#include "world_view.h"

#include <vector>

namespace dms_plugin {

// Converts a world into interleaved RGB float [0,1], compositing straight-alpha color over 50% gray
// so transparent regions do not inject arbitrary hidden RGB into the depth estimate.
// rgb receives width*height*3 floats; alpha (optional) receives width*height floats.
bool world_to_rgb(const WorldView& w, std::vector<float>& rgb, std::vector<float>* alpha);

// Bilinear resize of an interleaved RGB float image.
void resize_rgb(const std::vector<float>& src, int sw, int sh, std::vector<float>& dst, int dw, int dh);

// Fast 64-bit content hash of a world's pixels (used as the cache "layer checksum").
uint64_t hash_world(const WorldView& w) noexcept;
uint64_t hash_combine(uint64_t a, uint64_t b) noexcept;

// Fills a rect of the output world with a constant gray value and given alpha.
void fill_world(const WorldView& out, float value, float alpha) noexcept;

}  // namespace dms_plugin
