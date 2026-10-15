#include "sha256.h"
#include "test_helpers.h"

#include <cstring>
#include <fstream>
#include <string>

using dms::Sha256;

static std::string hex_of(const std::string& s) {
    Sha256 h;
    h.update(s.data(), s.size());
    return h.finish_hex();
}

TEST(Sha256, KnownVectors) {
    EXPECT_EQ(hex_of(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(hex_of("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(hex_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    EXPECT_EQ(hex_of(std::string(1000000, 'a')), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, StreamingMatchesOneShotAndStateRestores) {
    std::string data;
    for (int i = 0; i < 10007; ++i) data.push_back(static_cast<char>(i * 31 + 7));
    Sha256 a;
    a.update(data.data(), data.size());
    const auto one = a.finish_hex();

    Sha256 b;
    size_t pos = 0;
    size_t step = 1;
    while (pos < data.size()) {
        const size_t n = std::min(step, data.size() - pos);
        b.update(data.data() + pos, n);
        pos += n;
        step = step * 3 % 97 + 1;
        if (pos > 5000 && pos < 5100) {  // snapshot + restore mid-stream
            const auto st = b.state();
            Sha256 c;
            c.restore(st);
            b = c;
        }
    }
    EXPECT_EQ(b.finish_hex(), one);
}

TEST(Sha256, FileAndAbiAgree) {
    const auto dir = dms_test::tmp_dir("sha");
    const auto p = dir / "f.bin";
    {
        std::ofstream f(p, std::ios::binary);
        f << "hello world";
    }
    EXPECT_EQ(dms::sha256_file(p), "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    char hex[65];
    ASSERT_EQ(dms_sha256_hex("hello world", 11, hex), DMS_OK);
    EXPECT_STREQ(hex, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    EXPECT_EQ(dms::sha256_file(dir / "missing.bin"), "");
    EXPECT_EQ(dms_sha256_hex(nullptr, 5, hex), DMS_ERR_INVALID_ARG);
}
