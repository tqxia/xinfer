// Diagnostic ablation only: persistent inputs, direct dispatch versus CUDA Graph.
#include "runtime.h"
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>

using namespace qwen;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen_graph_probe PACKAGE_DIR WORKLOAD_JSON OUTPUT_JSON");
    require(!fs::exists(argv[3]), "Output already exists");
    const Spec spec(argv[1]);
    const auto workload = read_json(argv[2]);
    c10::InferenceMode inference;
    at::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const auto stream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    // The default runner queries completion events, which is not capture-safe.
    // This probe has one calling thread and uses the supported single-threaded path.
    Loader prefill((fs::path(argv[1])/"prefill.pt2").string(), "model", true);
    Loader decode((fs::path(argv[1])/"decode.pt2").string(), "model", true);
    const auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    std::vector<at::Tensor> cache;
    for (int64_t i=0; i<spec.layers*2; ++i)
      cache.push_back(at::zeros(spec.cache_shape, options.dtype(at::kBFloat16)));
    auto ids = at::zeros({1,1}, options), position = at::zeros({1}, options);
    std::vector<at::Tensor> inputs{ids,position};
    inputs.insert(inputs.end(),cache.begin(),cache.end());
    for (int i=0; i<3; ++i) decode.run(inputs)[0].argmax(-1);
    torch::cuda::synchronize();
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    auto graph_logits = decode.run(inputs)[0];
    auto graph_prediction = graph_logits.argmax(-1);
    graph.capture_end();
    graph.replay();
    torch::cuda::synchronize();
    Json report = {{"kind","diagnostic_direct_vs_graph"}, {"cases",Json::array()}};
    for (const auto& item : workload.at("cases")) {
      const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
      const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
      require(prompt.size() >= 2 && prompt.size() <= static_cast<size_t>(spec.max_prompt) &&
              !tokens.empty() && prompt.size()+tokens.size() <= static_cast<size_t>(spec.cache_length),
              "Invalid probe workload");
      for (const auto* values : {&prompt, &tokens})
        require(std::all_of(values->begin(), values->end(), [&](auto v){return v>=0 && v<spec.vocab;}), "Invalid token ID");
      Json result = {{"name",item.at("name")}, {"direct",Json::array()}, {"graph",Json::array()}};
      for (int round=-2; round<3; ++round) {
        std::vector<at::Tensor> reference_cache;
        at::Tensor reference_logits;
        std::vector<int64_t> reference_predictions;
        for (int mode_index=0; mode_index<2; ++mode_index) {
          const bool replay = ((round+2)%2 ? 1-mode_index : mode_index) != 0;
          for (auto& c : cache) c.zero_();
          std::vector<at::Tensor> pin{at::tensor(prompt,at::kLong).unsqueeze(0).to(at::kCUDA),
                                    at::arange(static_cast<int64_t>(prompt.size()),options)};
          pin.insert(pin.end(),cache.begin(),cache.end());
          std::vector<int64_t> predictions{prefill.run(pin)[0].argmax(-1).item<int64_t>()};
          at::Tensor logits;
          torch::cuda::synchronize();
          const auto start = Clock::now();
          for (size_t i=0; i<tokens.size(); ++i) {
            ids.fill_(tokens[i]);
            position.fill_(prompt.size()+i);
            if (replay) {
              graph.replay();
              logits = graph_logits;
              predictions.push_back(graph_prediction.item<int64_t>());
            } else {
              logits = decode.run(inputs)[0];
              predictions.push_back(logits.argmax(-1).item<int64_t>());
            }
          }
          torch::cuda::synchronize();
          const auto end = Clock::now();
          if (round >= 0) {
            const double elapsed = std::chrono::duration<double,std::milli>(end-start).count();
            result[replay?"graph":"direct"].push_back({{"decode_ms_per_token",elapsed/tokens.size()},
                                                        {"prediction_ids",predictions}});
            if (mode_index==0) {
              for (const auto& c : cache) reference_cache.push_back(c.clone());
              reference_logits=logits.clone();
              reference_predictions=predictions;
            } else {
              require(predictions == reference_predictions, "Graph predictions differ");
              require(at::equal(logits, reference_logits), "Graph final logits differ");
              for (size_t i=0;i<cache.size();++i)
                require(at::equal(cache[i],reference_cache[i]), "Graph final cache differs");
            }
          }
        }
      }
      result["direct_graph_predictions_logits_full_cache_bitwise"] = true;
      report["cases"].push_back(std::move(result));
      std::cout << item.at("name") << " direct/graph parity passed\n";
    }
    auto output = output_file(argv[3]);
    output << report.dump(2) << '\n';
    output.flush();
    require(bool(output), "Failed to write graph probe report");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
