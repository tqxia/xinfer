#include "../runtime.h"
#include <iostream>

using namespace qwen35;

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen35_validate PACKAGE_DIR CASES_JSON OUTPUT_DIR");
    const Spec spec(argv[1]);
    const auto cases = read_json(argv[2]).at("cases");
    const fs::path output(argv[3]);
    require(!fs::exists(output), "Output already exists");
    c10::InferenceMode inference;
    configure(spec);
    Runtime runtime(argv[1], spec);
    fs::create_directories(output);
    Json report = {{"cases", Json::array()}, {"loader_pairs", 1}};
    for (const auto& item : cases) {
      const auto name = item.at("name").get<std::string>();
      require(!name.empty() && name != "." && name != ".." && fs::path(name).filename() == name, "Invalid case name");
      const auto eos = item.value("eos", spec.eos);
      Json row = {{"name", name}};
      std::ofstream states(output / (name + ".states"), std::ios::binary);
      if (item.contains("prompt")) {
        std::ofstream logits(output / (name + ".f32"), std::ios::binary);
        const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
        const int64_t limit = item.at("limit");
        spec.check_request(prompt, limit);
        // Exercise a test-only stop set through the same production loop.
        const auto ids = item.contains("eos") ? xinfer::generate(runtime, prompt, limit, eos, &logits)
                                              : runtime.generate(prompt, limit, &logits);
        row["ids"] = ids;
        row["stop"] = std::find(eos.begin(), eos.end(), ids.back()) != eos.end() ? "eos" : "limit";
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
          const auto position = runtime.position();
          const auto length = static_cast<int64_t>(ids.size());
          std::vector<at::Tensor> before;
          for (const auto& tensor : inputs) before.push_back(tensor.clone());
          const auto logits = runtime.step(ids.get<std::vector<int64_t>>());
          require(at::isfinite(logits).all().item<bool>(), "Non-finite logits");
          write_tensor(states, logits);
          for (size_t i = 0; i < inputs.size(); ++i) {
            const auto kind = spec.metadata.at("state").at(i).at("kind");
            if (kind == "key" || kind == "value") {
              require(inputs[i].data_ptr() == runtime.state()[i].data_ptr(), "KV output lost caller-owned storage");
              require(inputs[i].strides() == runtime.state()[i].strides(), "KV output layout changed");
              require(at::equal(inputs[i].narrow(2, 0, position), before[i].narrow(2, 0, position)),
                      "Past KV overwritten");
              const auto end = position + length;
              require(at::equal(inputs[i].narrow(2, end, spec.capacity - end),
                                before[i].narrow(2, end, spec.capacity - end)), "Future KV overwritten");
            } else {
              require(at::equal(inputs[i], before[i]), "DeltaNet input state mutated");
            }
            require(at::isfinite(runtime.state()[i]).all().item<bool>(), "Non-finite state");
            write_tensor(states, runtime.state()[i]);
          }
          ++count;
        }
        row["steps"] = count;
        row["kv_inputs_updated_in_place"] = true;
        row["non_kv_inputs_unchanged"] = true;
        row["kv_write_regions_preserved"] = true;
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
    bool attention_library = false;
    for (std::string line; std::getline(maps, line);) {
      require(line.find("libpython") == std::string::npos && line.find("libtorch_python") == std::string::npos,
              "Python library loaded");
      const bool expected = line.find("libxinfer_attention_ops.so") != std::string::npos;
      require(line.find("libxinfer_") == std::string::npos || expected, "Unexpected custom-op library loaded");
      attention_library = attention_library || expected;
    }
    require(attention_library, "Native attention library was not loaded");
    report["no_python_library"] = true;
    report["attention_library"] = true;
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
