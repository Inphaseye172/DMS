#pragma once
#include <string>
#include <vector>

namespace depthcli {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> rgb;  // interleaved RGB [0,1]
};

bool load_image(const std::string& path, Image& out, std::string& err);
bool write_png_gray8(const std::string& path, const float* v, int w, int h);
bool write_png_rgb8(const std::string& path, const float* rgb, int w, int h);
bool write_png_rgba8(const std::string& path, const float* rgb, const float* alpha, int w, int h);
bool write_pgm16(const std::string& path, const float* v, int w, int h);
bool write_pfm(const std::string& path, const float* v, int w, int h);
bool read_pfm(const std::string& path, std::vector<float>& v, int& w, int& h);

}  // namespace depthcli
