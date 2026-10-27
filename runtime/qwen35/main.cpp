#include "runtime.h"
#include <iostream>

using namespace qwen35;

int main(int argc, char** argv) {
  try {
    if (argc < 4 || std::string(argv[1]) == "--help") {
      std::cerr << "Usage: qwen35_aoti PACKAGE_DIR PROMPT_IDS OUTPUT_IDS [--max-new-tokens N] [--trace LOGITS_F32]\n";
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
    spec.check_request(prompt, limit);  // Reject the whole request before loading weights.
    const fs::path output_path(argv[3]);
    require(!fs::exists(output_path), "Output already exists");
    require(trace_path.empty() || (!fs::exists(trace_path) &&
            fs::weakly_canonical(trace_path) != fs::weakly_canonical(output_path)), "Trace path already exists or aliases output");
    c10::InferenceMode inference;
    configure(spec);
    Runtime runtime(argv[1], spec);
    if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    require(output.is_open(), "Cannot create output");
    std::ofstream trace;
    if (!trace_path.empty()) {
      if (!trace_path.parent_path().empty()) fs::create_directories(trace_path.parent_path());
      trace.open(trace_path, std::ios::binary);
      require(trace.is_open(), "Cannot create trace");
    }
    const auto generated = runtime.generate(prompt, limit, trace_path.empty() ? nullptr : &trace);
    for (auto id : generated) output << id << ' ';
    output << '\n';
    output.close();
    require(bool(output), "Failed to write output");
    if (trace.is_open()) { trace.close(); require(bool(trace), "Failed to write trace"); }
    const bool eos = std::find(spec.eos.begin(), spec.eos.end(), generated.back()) != spec.eos.end();
    std::cerr << "prompt_tokens=" << prompt.size() << " generated_tokens=" << generated.size()
              << " consumed_positions=" << runtime.position() << " stop=" << (eos ? "eos" : "limit") << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
