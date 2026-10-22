#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Loader = torch::inductor::AOTIModelPackageLoader;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

int64_t integer(const std::string& text) {
  int64_t value;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
          "Invalid integer: " + text);
  return value;
}

Json read_json(const fs::path& path) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open metadata: " + path.string());
  return Json::parse(file);
}

struct Spec {
  int64_t layers, cache_length, max_prompt, vocab;
  std::vector<int64_t> cache_shape, eos;

  explicit Spec(const fs::path& directory) {
    const auto prefill = read_json(directory / "prefill.json");
    const auto decode = read_json(directory / "decode.json");
    for (const auto* key : {"model_revision", "torch_version", "torch_cuda_version",
                           "cxx11_abi", "cuda_compute_capability", "batch_size", "dtype",
                           "layers", "kv_heads", "head_dim", "cache_length", "vocab_size",
                           "cache_shape", "cache_stride", "flat_input_count", "eos_token_ids"}) {
      require(prefill.at(key) == decode.at(key), std::string("Package metadata differs: ") + key);
    }
    const auto version = decode.at("torch_version").get<std::string>();
    require(version.substr(0, version.find('+')) == TORCH_VERSION, "LibTorch version differs from package");
    require(decode.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), "LibTorch C++ ABI differs from package");
    require(decode.at("batch_size") == 1 && decode.at("dtype") == "bfloat16", "Expected batch=1 and BF16");
    require(prefill.at("entry_point") == "prefill" && decode.at("entry_point") == "decode",
            "Incorrect package entry points");
    require(decode.at("token_length") == Json::array({1, 1}) && prefill.at("token_length").at(0) == 2,
            "Unsupported token-length contract");
    layers = decode.at("layers");
    cache_length = decode.at("cache_length");
    max_prompt = prefill.at("token_length").at(1);
    vocab = decode.at("vocab_size");
    cache_shape = decode.at("cache_shape").get<std::vector<int64_t>>();
    eos = decode.at("eos_token_ids").get<std::vector<int64_t>>();
    const int64_t heads = decode.at("kv_heads"), dim = decode.at("head_dim");
    require(layers > 0 && cache_length > 0 && max_prompt >= 2 && max_prompt <= cache_length && vocab > 0,
            "Invalid model dimensions");
    require(cache_shape == std::vector<int64_t>{1, heads, cache_length, dim} && heads > 0 && dim > 0,
            "Invalid KV cache shape");
    require(decode.at("cache_stride") == Json::array({heads * cache_length * dim, cache_length * dim, dim, 1}),
            "Expected contiguous KV cache");
    require(decode.at("flat_input_count") == 2 + 2 * layers, "Invalid native input count");
    require(!eos.empty() && std::all_of(eos.begin(), eos.end(), [this](int64_t id) { return id >= 0 && id < vocab; }),
            "Invalid EOS token IDs");
  }
};

std::vector<int64_t> read_ids(const fs::path& path, const Spec& spec) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open token IDs: " + path.string());
  std::vector<int64_t> ids;
  for (std::string word; file >> word;) {
    const int64_t id = integer(word);
    require(id >= 0 && id < spec.vocab, "Token ID out of range: " + word);
    ids.push_back(id);
    require(static_cast<int64_t>(ids.size()) <= spec.max_prompt, "Prompt exceeds maximum token length");
  }
  require(file.eof(), "Failed to read token IDs: " + path.string());
  require(!ids.empty(), "Prompt must not be empty");
  return ids;
}

std::ofstream output_file(const fs::path& path, bool binary = false) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode{}));
  require(file.is_open(), "Cannot open output: " + path.string());
  return file;
}

class Runtime {
 public:
  Runtime(const fs::path& directory, const Spec& spec)
      : spec_(spec), prefill_((directory / "prefill.pt2").string()),
        decode_((directory / "decode.pt2").string()) {
    // LibTorch allocations are aligned; keep these exact buffers for both loaders.
    for (int64_t i = 0; i < 2 * spec.layers; ++i)
      cache_.push_back(at::zeros(spec.cache_shape, at::TensorOptions().device(device_).dtype(at::kBFloat16)));
  }

  std::vector<int64_t> generate(const std::vector<int64_t>& prompt, int64_t limit, std::ostream* trace) {
    for (auto& tensor : cache_) tensor.zero_();
    const auto options = at::TensorOptions().device(device_).dtype(at::kLong);
    std::vector<at::Tensor> inputs{
        at::tensor(prompt, at::kLong).unsqueeze(0).to(device_),
        at::arange(static_cast<int64_t>(prompt.size()), options)};
    inputs.insert(inputs.end(), cache_.begin(), cache_.end());
    int64_t position = prompt.size();
    std::vector<int64_t> generated;
    for (int64_t step = 0; step < limit; ++step) {
      // T=1 also handles a one-token prompt at position zero.
      auto outputs = (inputs[0].size(1) == 1 ? decode_ : prefill_).run(inputs);
      require(outputs.size() == 1 && outputs[0].sizes() == at::IntArrayRef({1, spec_.vocab}) &&
              outputs[0].scalar_type() == at::kBFloat16 && outputs[0].device() == device_,
              "Unexpected compiled output");
      const auto& logits = outputs[0];
      if (trace) {
        // Row i predicts generated[i]; same raw float32 format as the Python fixtures.
        const auto row = logits.to(at::kCPU, at::kFloat).contiguous();
        trace->write(reinterpret_cast<const char*>(row.const_data_ptr<float>()), row.numel() * sizeof(float));
        require(bool(*trace), "Failed to write logits trace");
      }
      const int64_t token = logits.argmax(-1).item<int64_t>();
      generated.push_back(token);
      if (std::find(spec_.eos.begin(), spec_.eos.end(), token) != spec_.eos.end() || step + 1 == limit) break;
      inputs[0] = at::full({1, 1}, token, options);
      inputs[1] = at::full({1}, position++, options);
    }
    return generated;
  }

 private:
  const Spec& spec_;
  const c10::Device device_{c10::kCUDA, 0};
  Loader prefill_, decode_;
  std::vector<at::Tensor> cache_;
};

int main(int argc, char** argv) {
  try {
    if (argc < 4 || std::string(argv[1]) == "--help") {
      std::cerr << "Usage: qwen_aoti PACKAGE_DIR PROMPT_IDS OUTPUT_IDS [--max-new-tokens N] [--trace LOGITS_F32]\n";
      return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
    }
    int64_t limit = 64;
    fs::path trace_path;
    for (int i = 4; i < argc; i += 2) {
      require(i + 1 < argc, "Missing option value");
      const std::string option = argv[i];
      if (option == "--max-new-tokens") limit = integer(argv[i + 1]);
      else if (option == "--trace") trace_path = argv[i + 1];
      else throw std::runtime_error("Unknown option: " + option);
    }
    const Spec spec(argv[1]);
    const auto prompt = read_ids(argv[2], spec);
    require(limit > 0 && limit <= spec.cache_length - static_cast<int64_t>(prompt.size()),
            "max-new-tokens must be positive and prompt + budget must fit the KV cache");
    const auto input_path = fs::weakly_canonical(argv[2]);
    const auto output_path = fs::weakly_canonical(argv[3]);
    require(input_path != output_path, "Input and output paths must differ");
    require(trace_path.empty() || (fs::weakly_canonical(trace_path) != input_path &&
                                  fs::weakly_canonical(trace_path) != output_path),
            "Trace path must differ from input and output");
    // Validate IDs and the complete requested budget before loading model weights.
    require(torch::cuda::is_available(), "CUDA device 0 is required");
    c10::InferenceMode inference;
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const auto load_start = Clock::now();
    Runtime runtime(argv[1], spec);
    torch::cuda::synchronize();
    const auto run_start = Clock::now();
    std::ofstream trace;
    if (!trace_path.empty()) trace = output_file(trace_path, true);
    const auto generated = runtime.generate(prompt, limit, trace_path.empty() ? nullptr : &trace);
    torch::cuda::synchronize();
    const auto run_end = Clock::now();
    auto output = output_file(argv[3]);
    for (const auto id : generated) output << id << ' ';
    output << '\n';
    output.flush();
    if (trace.is_open()) { trace.flush(); require(bool(trace), "Failed to flush logits trace"); }
    require(bool(output), "Failed to write generated IDs");
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::cerr << "prompt_tokens=" << prompt.size() << " generated_tokens=" << generated.size()
              << " load_ms=" << ms(load_start, run_start) << " generation_ms=" << ms(run_start, run_end)
              << " trace=" << !trace_path.empty() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
