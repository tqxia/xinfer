// Native test driver: all cases share the production Runtime and its two loaders.
#include "../runtime.h"

using namespace qwen;

std::vector<at::Tensor> snapshot(const Runtime& runtime) {
  std::vector<at::Tensor> result;
  for (const auto& tensor : runtime.cache()) result.push_back(tensor.clone());
  return result;
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen_validate PACKAGE_DIR CASES_JSON OUTPUT_DIR");
    const Spec spec(argv[1]);
    const auto manifest = read_json(argv[2]);
    const fs::path output_dir(argv[3]);
    c10::InferenceMode inference;
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    Runtime runtime(argv[1], spec);
    std::vector<const void*> pointers;
    for (const auto& tensor : runtime.cache()) pointers.push_back(tensor.const_data_ptr());
    const auto check_addresses = [&] {
      for (size_t i = 0; i < pointers.size(); ++i)
        require(runtime.cache()[i].const_data_ptr() == pointers[i], "KV cache storage changed");
    };
    Json report = {{"cases", Json::array()}, {"loader_pairs", 1}, {"cache_buffers", pointers.size()}};
    for (const auto& item : manifest.at("cases")) {
      const std::string name = item.at("name");
      require(!name.empty() && fs::path(name).filename() == name, "Invalid case name");
      auto trace = output_file(output_dir / (name + ".f32"), true);
      Json entry = {{"name", name}};
      if (item.at("mode") == "greedy") {
        const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
        const auto ids = runtime.generate(prompt, item.at("limit"), &trace);
        entry["ids"] = ids;
        entry["rows"] = ids.size();
      } else {
        const double fill = item.value("fill", 0.0);
        runtime.reset(fill);
        for (const auto& tensor : runtime.cache())
          require(tensor.eq(fill).all().item<bool>(), "Cache reset left stale data");
        int64_t rows = 0;
        for (const auto& chunk : item.at("chunks")) {
          const auto tokens = chunk.get<std::vector<int64_t>>();
          const int64_t start = runtime.position(), end = start + tokens.size();
          const auto before = snapshot(runtime);
          write_logits(trace, runtime.step(tokens));
          check_addresses();
          require(runtime.position() == end, "Incorrect position advance");
          for (size_t i = 0; i < before.size(); ++i) {
            const auto& after = runtime.cache()[i];
            require(at::equal(before[i].slice(2, 0, start), after.slice(2, 0, start)), "Cache prefix changed");
            require(at::equal(before[i].slice(2, end), after.slice(2, end)), "Cache suffix changed");
            require(!at::equal(before[i].slice(2, start, end), after.slice(2, start, end)), "Cache update missing");
          }
          ++rows;
        }
        entry["rows"] = rows;
        entry["write_region_checks"] = true;
        entry["reset_check"] = true;
        if (item.value("dump_cache", false)) {
          auto file = output_file(output_dir / (name + ".kv.bf16"), true);
          for (const auto& tensor : runtime.cache()) {
            const auto cpu = tensor.to(at::kCPU).contiguous();
            file.write(static_cast<const char*>(cpu.const_data_ptr()), cpu.nbytes());
          }
          file.flush();
          require(bool(file), "Failed to write cache dump");
        }
        if (item.value("check_overflow", false)) {
          require(runtime.position() == spec.cache_length, "Overflow case must fill the cache");
          const auto before = snapshot(runtime);
          bool rejected = false;
          try { runtime.step({1}); }
          catch (const std::exception& error) {
            rejected = std::string(error.what()).find("exceeds KV cache capacity") != std::string::npos;
          }
          require(rejected && runtime.position() == spec.cache_length, "Cache overflow was not rejected");
          for (size_t i = 0; i < before.size(); ++i)
            require(at::equal(before[i], runtime.cache()[i]), "Rejected step mutated cache");
          entry["overflow_rejected_without_mutation"] = true;
        }
      }
      trace.flush();
      require(bool(trace), "Failed to flush logits trace");
      check_addresses();
      entry["position"] = runtime.position();
      entry["stable_storage"] = true;
      report["cases"].push_back(entry);
      std::cerr << name << ": " << entry.at("rows") << " rows, position " << runtime.position() << '\n';
    }
    auto file = output_file(output_dir / "native.json");
    file << report.dump(2) << '\n';
    file.flush();
    require(bool(file), "Failed to write native validation report");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
