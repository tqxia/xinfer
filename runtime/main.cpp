#include "common/model.h"
#include <chrono>
#include <iostream>

using namespace xinfer;
using Clock = std::chrono::steady_clock;

int main(int argc, char **argv) {
  try {
    if (argc < 4 || std::string(argv[1]) == "--help") {
      std::cerr << "Usage: xinfer_aoti PACKAGE_DIR PROMPT_IDS OUTPUT_IDS [--max-output-tokens N] "
                   "[--trace LOGITS_F32]\n";
      return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
    }
    int64_t limit = 64;
    fs::path trace_path;
    for (int i = 4; i < argc; i += 2) {
      require(i + 1 < argc, "Missing option value");
      const std::string option = argv[i];
      if (option == "--max-output-tokens")
        limit = integer(argv[i + 1]);
      else if (option == "--trace")
        trace_path = argv[i + 1];
      else
        throw std::runtime_error("Unknown option: " + option);
    }
    return with_model(argv[1], [&](const auto &spec, auto make_runtime) {
      const auto prompt = read_ids(argv[2], spec);
      spec.check_request(prompt, limit);
      const fs::path output_path(argv[3]);
      require(!fs::exists(output_path), "Output already exists");
      const auto input_path = fs::weakly_canonical(argv[2]);
      const auto canonical_output = fs::weakly_canonical(output_path);
      require(input_path != canonical_output, "Input and output paths must differ");
      require(trace_path.empty() || (fs::weakly_canonical(trace_path) != input_path &&
                                     fs::weakly_canonical(trace_path) != canonical_output),
              "Trace path must differ from input and output");
      require(trace_path.empty() || !fs::exists(trace_path), "Trace path already exists");
      c10::InferenceMode inference;
      const auto load_start = Clock::now();
      auto runtime = make_runtime();
      torch::cuda::synchronize();
      const auto run_start = Clock::now();
      std::ofstream trace;
      if (!trace_path.empty())
        trace = output_file(trace_path, true);
      const auto generated =
          runtime->generate(prompt, limit, trace_path.empty() ? nullptr : &trace);
      torch::cuda::synchronize();
      const auto run_end = Clock::now();
      auto output = output_file(output_path);
      for (auto id : generated)
        output << id << ' ';
      output << '\n';
      output.close();
      require(bool(output), "Failed to write generated IDs");
      if (trace.is_open()) {
        trace.close();
        require(bool(trace), "Failed to write logits trace");
      }
      const bool eos =
          std::find(spec.eos.begin(), spec.eos.end(), generated.back()) != spec.eos.end();
      const auto ms = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
      };
      std::cerr << "prompt_tokens=" << prompt.size() << " generated_tokens=" << generated.size()
                << " consumed_positions=" << runtime->position()
                << " stop=" << (eos ? "eos" : "limit") << " load_ms=" << ms(load_start, run_start)
                << " generation_ms=" << ms(run_start, run_end) << " trace=" << !trace_path.empty()
                << " decode_backend=aoti"
                << " compiled_decode_backend=" << (spec.compiled_megaqwen ? "megaqwen" : "aoti")
                << '\n';
      return 0;
    });
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
