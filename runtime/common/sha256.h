#pragma once

#include "io.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace xinfer {

namespace detail {

inline constexpr std::array<uint32_t, 64> kSha256K{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline void sha256_transform_scalar(std::array<uint32_t, 8> &h, const unsigned char *blocks,
                                    size_t count) {
  for (size_t b = 0; b < count; ++b, blocks += 64) {
    std::array<uint32_t, 64> w{};
    for (size_t i = 0; i < 16; ++i)
      w[i] = uint32_t(blocks[4 * i]) << 24 | uint32_t(blocks[4 * i + 1]) << 16 |
             uint32_t(blocks[4 * i + 2]) << 8 | uint32_t(blocks[4 * i + 3]);

    for (size_t i = 16; i < 64; ++i) {
      const uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    auto a = h[0], b0 = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], z = h[7];
    for (size_t i = 0; i < 64; ++i) {
      const uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t t1 = z + s1 + choice + kSha256K[i] + w[i];
      const uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const uint32_t majority = (a & b0) ^ (a & c) ^ (b0 & c);
      const uint32_t t2 = s0 + majority;

      z = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b0;
      b0 = a;
      a = t1 + t2;
    }

    h[0] += a;
    h[1] += b0;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += z;
  }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sha,ssse3"))) inline void
sha256_transform_x86_sha(std::array<uint32_t, 8> &h, const unsigned char *blocks, size_t count) {
  const __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
  __m128i state0 = _mm_set_epi32(static_cast<int>(h[0]), static_cast<int>(h[1]),
                                 static_cast<int>(h[4]), static_cast<int>(h[5]));
  __m128i state1 = _mm_set_epi32(static_cast<int>(h[2]), static_cast<int>(h[3]),
                                 static_cast<int>(h[6]), static_cast<int>(h[7]));

  for (size_t b = 0; b < count; ++b, blocks += 64) {
    const __m128i abef_save = state0;
    const __m128i cdgh_save = state1;

    __m128i msg0 =
        _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(blocks)), shuf_mask);
    __m128i msg1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(blocks + 16)),
                                    shuf_mask);
    __m128i msg2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(blocks + 32)),
                                    shuf_mask);
    __m128i msg3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(blocks + 48)),
                                    shuf_mask);

    __m128i wk =
        _mm_add_epi32(msg0, _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data())));
    state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
    state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));

    wk = _mm_add_epi32(msg1,
                       _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data() + 4)));
    state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
    state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));

    wk = _mm_add_epi32(msg2,
                       _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data() + 8)));
    state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
    state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));

    wk = _mm_add_epi32(msg3,
                       _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data() + 12)));
    state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
    state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));

    for (size_t r = 16; r < 64; r += 16) {
      msg0 = _mm_sha256msg2_epu32(
          _mm_add_epi32(_mm_sha256msg1_epu32(msg0, msg1), _mm_alignr_epi8(msg3, msg2, 4)), msg3);
      wk = _mm_add_epi32(msg0,
                         _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data() + r)));
      state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
      state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));

      msg1 = _mm_sha256msg2_epu32(
          _mm_add_epi32(_mm_sha256msg1_epu32(msg1, msg2), _mm_alignr_epi8(msg0, msg3, 4)), msg0);
      wk = _mm_add_epi32(
          msg1, _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data() + r + 4)));
      state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
      state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));

      msg2 = _mm_sha256msg2_epu32(
          _mm_add_epi32(_mm_sha256msg1_epu32(msg2, msg3), _mm_alignr_epi8(msg1, msg0, 4)), msg1);
      wk = _mm_add_epi32(
          msg2, _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data() + r + 8)));
      state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
      state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));

      msg3 = _mm_sha256msg2_epu32(
          _mm_add_epi32(_mm_sha256msg1_epu32(msg3, msg0), _mm_alignr_epi8(msg2, msg1, 4)), msg2);
      wk = _mm_add_epi32(
          msg3, _mm_loadu_si128(reinterpret_cast<const __m128i *>(kSha256K.data() + r + 12)));
      state1 = _mm_sha256rnds2_epu32(state1, state0, wk);
      state0 = _mm_sha256rnds2_epu32(state0, state1, _mm_shuffle_epi32(wk, 0x0E));
    }

    state0 = _mm_add_epi32(state0, abef_save);
    state1 = _mm_add_epi32(state1, cdgh_save);
  }

  alignas(16) uint32_t s0[4], s1[4];
  _mm_storeu_si128(reinterpret_cast<__m128i *>(s0), state0);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(s1), state1);
  h[0] = s0[3];
  h[1] = s0[2];
  h[2] = s1[3];
  h[3] = s1[2];
  h[4] = s0[1];
  h[5] = s0[0];
  h[6] = s1[1];
  h[7] = s1[0];
}
#endif

inline void sha256_transform_blocks(std::array<uint32_t, 8> &h, const unsigned char *blocks,
                                    size_t count) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool use_sha = __builtin_cpu_supports("sha") && __builtin_cpu_supports("ssse3");
  if (use_sha) {
    sha256_transform_x86_sha(h, blocks, count);
    return;
  }
#endif
  sha256_transform_scalar(h, blocks, count);
}

} // namespace detail

// Streaming SHA-256 keeps archive verification independent of Python/OpenSSL,
// using runtime-dispatched x86_64 SHA-NI when available and portable scalar code elsewhere.
inline std::string sha256_file(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  require(file.is_open(), "Shared weights cannot open file for checksum: " + path.string());

  std::array<uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

  std::vector<unsigned char> buffer(256 * 1024);
  uint64_t bytes = 0;
  size_t tail = 0;
  while (file) {
    file.read(reinterpret_cast<char *>(buffer.data() + tail),
              static_cast<std::streamsize>(buffer.size() - tail));
    const size_t count = static_cast<size_t>(file.gcount());
    if (count == 0)
      break;
    bytes += count;
    const size_t total = tail + count;
    const size_t blocks = total / 64;
    if (blocks > 0)
      detail::sha256_transform_blocks(h, buffer.data(), blocks);
    tail = total - blocks * 64;
    if (tail > 0)
      std::memmove(buffer.data(), buffer.data() + blocks * 64, tail);
  }
  require(file.eof(), "Shared weights checksum read failed");

  buffer[tail++] = 0x80;
  if (tail > 56) {
    std::fill(buffer.begin() + static_cast<std::ptrdiff_t>(tail), buffer.begin() + 64, 0);
    detail::sha256_transform_blocks(h, buffer.data(), 1);
    tail = 0;
  }
  std::fill(buffer.begin() + static_cast<std::ptrdiff_t>(tail), buffer.begin() + 56, 0);
  const uint64_t bits = bytes * uint64_t{8};
  for (size_t i = 0; i < 8; ++i)
    buffer[63 - i] = static_cast<unsigned char>(bits >> (8 * i));
  detail::sha256_transform_blocks(h, buffer.data(), 1);

  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (auto word : h)
    result << std::setw(8) << word;
  return result.str();
}

} // namespace xinfer
