#pragma once
#include "io.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace xinfer {

// Streaming SHA-256 keeps archive verification independent of Python/OpenSSL.
inline std::string sha256_file(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  require(file.is_open(), "Shared weights cannot open file for checksum: " + path.string());
  constexpr std::array<uint32_t, 64> k{
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  std::array<uint32_t, 8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                           0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  auto transform = [&](const unsigned char* block) {
    std::array<uint32_t, 64> w{};
    for (size_t i = 0; i < 16; ++i)
      w[i] = uint32_t(block[4*i]) << 24 | uint32_t(block[4*i+1]) << 16 |
             uint32_t(block[4*i+2]) << 8 | uint32_t(block[4*i+3]);
    for (size_t i = 16; i < 64; ++i) {
      const uint32_t s0 = std::rotr(w[i-15],7) ^ std::rotr(w[i-15],18) ^ (w[i-15] >> 3);
      const uint32_t s1 = std::rotr(w[i-2],17) ^ std::rotr(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    auto a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], z=h[7];
    for (size_t i = 0; i < 64; ++i) {
      const uint32_t s1 = std::rotr(e,6) ^ std::rotr(e,11) ^ std::rotr(e,25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t t1 = z + s1 + choice + k[i] + w[i];
      const uint32_t s0 = std::rotr(a,2) ^ std::rotr(a,13) ^ std::rotr(a,22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + majority;
      z=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=z;
  };
  std::array<unsigned char, 65536> buffer{};
  uint64_t bytes = 0;
  size_t tail = 0;
  while (file) {
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const size_t count = static_cast<size_t>(file.gcount());
    bytes += count;
    for (size_t offset = 0; offset + 64 <= count; offset += 64) transform(buffer.data() + offset);
    tail = count % 64;
    if (tail) std::copy_n(buffer.data() + count - tail, tail, buffer.data());
  }
  require(file.eof(), "Shared weights checksum read failed");
  buffer[tail++] = 0x80;
  if (tail > 56) {
    std::fill(buffer.begin() + tail, buffer.begin() + 64, 0);
    transform(buffer.data());
    tail = 0;
  }
  std::fill(buffer.begin() + tail, buffer.begin() + 56, 0);
  const uint64_t bits = bytes * uint64_t{8};
  for (size_t i = 0; i < 8; ++i) buffer[63-i] = static_cast<unsigned char>(bits >> (8*i));
  transform(buffer.data());
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (auto word : h) result << std::setw(8) << word;
  return result.str();
}

}  // namespace xinfer
