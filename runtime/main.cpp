#include "common/model.h"

#include <chrono>
#include <iostream>

using namespace xinfer;
using Clock = std::chrono::steady_clock;

namespace {
struct Options {
  fs::path packages, input, output, trace;
  int64_t limit = 64;
};

Options parse_options(int argc, char **argv) {
  Options options{argv[1], argv[2], argv[3], {}};
  for (int i = 4; i < argc; i += 2) {
    require(i + 1 < argc, "Missing option value");
    const std::string option = argv[i];
    if (option == "--max-output-tokens")
      options.limit = integer(argv[i + 1]);
    else if (option == "--trace")
      options.trace = argv[i + 1];
    else
      throw std::runtime_error("Unknown option: " + option);
  }

  return options;
}

std::vector<int64_t> validate_request(const Options &options, const SpecBase &spec) {
  const auto prompt = read_ids(options.input, spec);
  spec.check_request(prompt, options.limit);

  require(!path_exists(options.output), "Output already exists");
  const auto input_path = fs::weakly_canonical(options.input);
  const auto canonical_output = fs::weakly_canonical(options.output);
  require(input_path != canonical_output, "Input and output paths must differ");
  require(options.trace.empty() || (fs::weakly_canonical(options.trace) != input_path &&
                                    fs::weakly_canonical(options.trace) != canonical_output),
          "Trace path must differ from input and output");
  require(options.trace.empty() || !path_exists(options.trace), "Trace path already exists");

  return prompt;
}

void write_generated_ids(const fs::path &path, const std::vector<int64_t> &generated) {
  auto output = output_file(path);
  for (auto id : generated)
    output << id << ' ';
  output << '\n';

  output.close();
  require(bool(output), "Failed to write generated IDs");
}

struct GenerationResult {
  std::vector<int64_t> ids;
  int64_t position;
  double load_ms, generation_ms;
};

void report_generation(const Options &options, const SpecBase &spec, size_t prompt_tokens,
                       const GenerationResult &result) {
  const bool eos = std::find(spec.eos.begin(), spec.eos.end(), result.ids.back()) != spec.eos.end();
  std::cerr << "prompt_tokens=" << prompt_tokens << " generated_tokens=" << result.ids.size()
            << " consumed_positions=" << result.position << " stop=" << (eos ? "eos" : "limit")
            << " load_ms=" << result.load_ms << " generation_ms=" << result.generation_ms
            << " trace=" << !options.trace.empty() << '\n';
}

void run_generation(const Options &options) {
  with_model(options.packages, [&](const auto &spec, auto make_runtime) {
    const auto prompt = validate_request(options, spec);
    c10::InferenceMode inference;

    const auto load_start = Clock::now();
    auto runtime = make_runtime(1);
    torch::cuda::synchronize();

    const auto run_start = Clock::now();
    std::ofstream trace;
    if (!options.trace.empty())
      trace = output_file(options.trace, true);

    auto generated =
        runtime->generate(prompt, options.limit, options.trace.empty() ? nullptr : &trace);
    torch::cuda::synchronize();
    const auto run_end = Clock::now();

    write_generated_ids(options.output, generated);
    if (trace.is_open()) {
      trace.close();
      require(bool(trace), "Failed to write logits trace");
    }

    const auto ms = [](auto start, auto end) {
      return std::chrono::duration<double, std::milli>(end - start).count();
    };
    const GenerationResult result{std::move(generated), runtime->position(),
                                  ms(load_start, run_start), ms(run_start, run_end)};
    report_generation(options, spec, prompt.size(), result);
  });
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 4 || std::string(argv[1]) == "--help") {
      std::cerr << "Usage: xinfer_aoti PACKAGE_DIR PROMPT_IDS OUTPUT_IDS [--max-output-tokens N] "
                   "[--trace LOGITS_F32]\n";
      return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
    }

    run_generation(parse_options(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
