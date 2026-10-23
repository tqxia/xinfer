#include "runtime.h"
#include <chrono>

using namespace qwen;
using Clock = std::chrono::steady_clock;

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
