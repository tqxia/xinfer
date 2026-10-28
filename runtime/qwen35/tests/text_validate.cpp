#include "../runtime.h"
#include <iostream>

using namespace qwen35;

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen35_validate PACKAGE_DIR CASES_JSON OUTPUT_DIR");
    Spec spec(argv[1]);
    const auto cases = read_json(argv[2]).at("cases");
    const fs::path output(argv[3]);
    require(!fs::exists(output), "Output already exists");
    c10::InferenceMode inference;
    configure(spec);
    Runtime runtime(argv[1], spec);
    fs::create_directories(output);
    const auto original_eos = spec.eos;
    Json report = {{"cases", Json::array()}, {"loader_pairs", 1}};
    for (const auto& item : cases) {
      const auto name = item.at("name").get<std::string>();
      require(!name.empty() && name != "." && name != ".." && fs::path(name).filename() == name, "Invalid case name");
      spec.eos = item.value("eos", original_eos);  // A test-only override exercises stopping deterministically.
      Json row = {{"name", name}};
      std::ofstream states(output / (name + ".states"), std::ios::binary);
      if (item.contains("prompt")) {
        std::ofstream logits(output / (name + ".f32"), std::ios::binary);
        const auto ids = runtime.generate(item.at("prompt").get<std::vector<int64_t>>(), item.at("limit"), &logits);
        row["ids"] = ids;
        row["stop"] = std::find(spec.eos.begin(), spec.eos.end(), ids.back()) != spec.eos.end() ? "eos" : "limit";
        for (const auto& state : runtime.state()) write_tensor(states, state);
      } else {
        runtime.reset();
        const auto poison = item.value("poison_kv", 0.0);
        if (poison != 0.0)
          for (size_t i = 0; i < runtime.state().size(); ++i) {
            const auto kind = spec.metadata.at("state").at(i).at("kind");
            if (kind == "key" || kind == "value") runtime.state()[i].fill_(poison);
          }
        size_t count = 0;
        for (const auto& ids : item.at("steps")) {
          const auto inputs = runtime.state();
          std::vector<at::Tensor> before;
          for (const auto& tensor : inputs) before.push_back(tensor.clone());
          const auto logits = runtime.step(ids.get<std::vector<int64_t>>());
          require(at::isfinite(logits).all().item<bool>(), "Non-finite logits");
          write_tensor(states, logits);
          for (size_t i = 0; i < inputs.size(); ++i) {
            require(at::equal(inputs[i], before[i]), "Input state mutated");
            require(at::isfinite(runtime.state()[i]).all().item<bool>(), "Non-finite state");
            write_tensor(states, runtime.state()[i]);
          }
          ++count;
        }
        row["steps"] = count;
        row["inputs_unchanged"] = true;
      }
      if (item.value("check_overflow", false)) {
        require(runtime.position() == spec.capacity, "Context not full");
        const auto before = runtime.state();
        bool rejected = false;
        try { runtime.step({0}); } catch (const std::runtime_error& e) {
          rejected = std::string(e.what()) == "Context capacity exceeded";
        }
        require(rejected && runtime.position() == spec.capacity, "Overflow not rejected");
        for (size_t i = 0; i < before.size(); ++i)
          require(before[i].data_ptr() == runtime.state()[i].data_ptr(), "Overflow changed state");
        row["overflow_rejected"] = true;
      }
      states.close();
      require(bool(states), "Failed to close state dump");
      row["position"] = runtime.position();
      row["prefill_calls"] = runtime.prefill_calls();
      row["decode_calls"] = runtime.decode_calls();
      report["cases"].push_back(row);
    }
    std::ifstream maps("/proc/self/maps");
    require(maps.is_open(), "Cannot inspect process mappings");
    for (std::string line; std::getline(maps, line);)
      require(line.find("libpython") == std::string::npos && line.find("libtorch_python") == std::string::npos &&
              line.find("libxinfer_") == std::string::npos, "Python or custom-op library loaded");
    report["no_python_library"] = true;
    std::ofstream file(output / "native.json");
    file << report.dump(2) << '\n';
    file.close();
    require(bool(file), "Failed to write report");
    std::cout << "Native validation completed " << cases.size() << " cases\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
