#include "algorithm/sgps.h"
#include "algorithm/train_bootstrap.h"
#include "common/init.h"
#include "common/signaller.h"
#include "env/boulderdash.h"
#include "env/boxworld.h"
#include "env/craftworld.h"
#include "env/env_loader.h"
#include "env/sokoban.h"
#include "env/tsp.h"
#include "model/twoheaded_convnet_wrapper.h"
#include "model/vsc_wrapper.h"
#include "util/replay_buffer.h"
#include "util/stop_token.h"
#include "util/utility.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>

#include <filesystem>
#include <memory>
#include <random>
#include <ranges>
#include <string>
#include <vector>

using namespace hpts;
using namespace hpts::algorithm;
using namespace hpts::env;
using namespace hpts::model::wrapper;
using json = nlohmann::json;

constexpr std::size_t INF_SIZE_T = std::numeric_limits<std::size_t>::max();
constexpr double INF_D = std::numeric_limits<double>::max();
constexpr int INF_I = std::numeric_limits<int>::max();
constexpr long long int INF_LLI = std::numeric_limits<long long int>::max();
constexpr double MAX_TIME = 60 * 60 * 24 * 365;

// NOLINTBEGIN
ABSL_FLAG(std::string, environment, "", "String name of the environment");
ABSL_FLAG(std::string, problems_path, "", "Path to problems file");
ABSL_FLAG(std::string, output_dir, "/opt/hpts/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, model_vsc_path, "", "Path for input vsc model JSON definition.");
ABSL_FLAG(std::string, model_low_path, "", "Path for input low model JSON definition.");
ABSL_FLAG(int, search_budget, -1, "Maximum number of expanded nodes before termination");
ABSL_FLAG(int, inference_batch_size, 32, "Number of search expansions to batch per inference query");
ABSL_FLAG(double, mix_epsilon, 0.0, "Percentage to mix with uniform policy");
ABSL_FLAG(double, mix_low_alpha, 0.25, "Mixing alpha for low level policy with conditional mixture");
ABSL_FLAG(double, rho, 1.0, "Rho parameter for Louvain algorithm which controls important of in edges");
ABSL_FLAG(int, num_cluster_samples, 1, "Number of samples to create training input on the Louvain clustering");
ABSL_FLAG(int, cluster_level, -1, "Level in the Louvain clustering to sample from, -1 for half-level");
ABSL_FLAG(int, cluster_buffer_size, 10000, "Size of the cluster trajectory distance buffer");
ABSL_FLAG(int, seed, 0, "Seed for all sources of RNG");
ABSL_FLAG(std::size_t, num_train, INF_SIZE_T, "Number of instances of the max to use for training");
ABSL_FLAG(std::size_t, num_validate, INF_SIZE_T, "Number of instances of the max to use for validation");
ABSL_FLAG(int, max_iterations, INF_I, "Maximum number of iterations of running the bootstrap process");
ABSL_FLAG(int, max_budget, INF_I, "Maximum search budget before terminating the bootstrap process");
ABSL_FLAG(double, time_budget, INF_D, "Budget in seconds before terminating");
ABSL_FLAG(double, validation_solved_ratio, 0.99, "Percentage of validation set to solve before checkpointing");
ABSL_FLAG(int, extra_iterations, -1, "Extra iterations of the training process before stopping after validation.");
ABSL_FLAG(int, bootstrap_policy, 0, "Bootstrap policy, 0 for Double, 1 for LTS_CM");
ABSL_FLAG(double, bootstrap_factor, 0.1, "Bootstrap increase factor if using LTS_CM policy");
ABSL_FLAG(int, learning_batch_size, 256, "Batch size used for model updates");
ABSL_FLAG(int, num_threads, 1, "Number of threads to run in the search thread pool");
ABSL_FLAG(int, num_problems_per_batch, 32, "Number of problems per bootstrap iteration");
ABSL_FLAG(int, grad_steps, 10, "Number of gradient updates per batch iteration");
ABSL_FLAG(bool, use_unsolved_data, true, "Use unsolved tree data to train networks");
ABSL_FLAG(bool, use_heuristic, true, "True to use a heuristic for PHS, false for LevinTS");
ABSL_FLAG(bool, resume, false, "True preload model from checkpoint (for diagnostic)");
ABSL_FLAG(int, device_num, 0, "Torch cuda device number to use (defaults to 1)");
// NOLINTEND

namespace {
template <typename T>
class CircularBuffer {
public:
    CircularBuffer(int max_size)
        : max_size(max_size) {}

    void add(T d) {
        if (data.size() < max_size) {
            data.push_back(T{});
        }
        data[idx] = d;
        idx = (idx + 1) % max_size;
    }

    [[nodiscard]] auto size() -> int {
        return static_cast<int>(data.size());
    }

    [[nodiscard]] auto mean() -> double {
        const auto N = static_cast<double>(data.size());
        return std::reduce(data.begin(), data.end()) / N;
    }

    [[nodiscard]] auto stdev() -> double {
        const auto N = static_cast<double>(data.size());
        double mean = std::reduce(data.begin(), data.end()) / N;
        double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        return std::sqrt(sq_sum / N - mean * mean);
    }

private:
    int max_size;
    std::size_t idx = 0;
    std::vector<T> data;
};

template <typename EnvT, hphs_mix::VSCWrapper VSCWrapperT>
auto create_search_inputs(
    const std::vector<EnvT> &problems,
    std::shared_ptr<StopToken> stop_token,
    std::shared_ptr<VSCWrapperT> model_vsc,
    std::shared_ptr<TwoHeadedConvNetWrapper> model_low
) {
    using SearchInputT = hphs_mix::SearchInput<EnvT, VSCWrapperT>;
    std::vector<SearchInputT> search_inputs;
    for (std::size_t i = 0; i < problems.size(); ++i) {
        search_inputs.emplace_back(
            absl::StrFormat("puzzle_%d", i),
            problems[i],
            absl::GetFlag(FLAGS_use_heuristic),
            absl::GetFlag(FLAGS_search_budget),
            absl::GetFlag(FLAGS_inference_batch_size),
            absl::GetFlag(FLAGS_mix_epsilon),
            absl::GetFlag(FLAGS_mix_low_alpha),
            absl::GetFlag(FLAGS_rho),
            absl::GetFlag(FLAGS_num_cluster_samples),
            absl::GetFlag(FLAGS_cluster_level),
            1,
            0,
            static_cast<unsigned long long int>(i) * search_inputs.size() + absl::GetFlag(FLAGS_seed),
            absl::GetFlag(FLAGS_use_unsolved_data),
            stop_token,
            model_vsc,
            model_low
        );
    }
    return split_train_validate(
        search_inputs,
        absl::GetFlag(FLAGS_num_train),
        absl::GetFlag(FLAGS_num_validate),
        absl::GetFlag(FLAGS_seed)
    );
}

template <typename EnvT, hphs_mix::VSCWrapper VSCWrapperT>
class HPHSLearner {
public:
    HPHSLearner(
        std::shared_ptr<VSCWrapperT> model_vsc,
        std::shared_ptr<TwoHeadedConvNetWrapper> model_low,
        int learning_batch_size,
        int grad_steps
    )
        : model_vsc(std::move(model_vsc)),
          model_low(std::move(model_low)),
          learning_batch_size(learning_batch_size),
          grad_steps(grad_steps),
          cluster_distance_buffer(absl::GetFlag(FLAGS_cluster_buffer_size)),
          replay_buffer(10000, 1024) {}

    void checkpoint() {
        model_vsc->SaveCheckpointWithoutOptimizer(-1);
        model_low->SaveCheckpointWithoutOptimizer(-1);
    }

    void preprocess([[maybe_unused]] std::vector<hphs_mix::SearchInput<EnvT, VSCWrapperT>> &batch) {
        double mean = cluster_distance_buffer.size() == 0 ? 5 : cluster_distance_buffer.mean();
        double stdev = cluster_distance_buffer.size() == 0 ? 1 : cluster_distance_buffer.stdev();
        SPDLOG_INFO("Cluster Buffer: N={:d}, MEAN={:.4f}, STDEV: {:.4f}", cluster_distance_buffer.size(), mean, stdev);
        for (auto &batch_item : batch) {
            batch_item.sample_mean = mean;
            batch_item.sample_stdev = stdev;
        }
    }

    void _process_data_vqvae(std::vector<hphs_mix::SearchOutput> &search_outputs) {
        learning_inputs_vqvae.clear();
        for (auto &result : search_outputs) {
            for (const auto &[obs_input, obs_target] :
                 std::views::zip(result.vqvae_input_observations, result.vqvae_target_observations))
            {
                learning_inputs_vqvae.emplace_back(std::move(obs_input), std::move(obs_target));
            }
        }
    }

    void _process_data_subgoal(std::vector<hphs_mix::SearchOutput> &search_outputs) {
        learning_inputs_subgoal.clear();
        for (auto &result : search_outputs) {
            for (const auto &[obs_input, obs_target] :
                 std::views::zip(result.subgoal_input_observations, result.subgoal_target_observations))
            {
                learning_inputs_subgoal.emplace_back(std::move(obs_input), std::move(obs_target), result.num_expanded);
            }
        }
    }

    void _process_data_conditional_low(std::vector<hphs_mix::SearchOutput> &search_outputs) {
        learning_inputs_conditional_low.clear();
        for (auto &result : search_outputs) {
            for (const auto &[obs_input, obs_target, action] : std::views::zip(
                     result.low_partial_input_observations,
                     result.low_partial_target_observations,
                     result.low_partial_actions
                 ))
            {
                learning_inputs_conditional_low
                    .emplace_back(std::move(obs_input), std::move(obs_target), action, result.num_expanded);
            }
        }
    }

    void _process_data_low(std::vector<hphs_mix::SearchOutput> &search_outputs) {
        learning_inputs_low.clear();
        for (auto &result : search_outputs) {
            for (const auto &[obs, action, cost] : std::views::zip(
                     result.solution_path_observations,
                     result.solution_path_actions,
                     result.solution_path_costs
                 ))
            {
                learning_inputs_low.emplace_back(std::move(obs), action, cost, result.num_expanded);
            }
        }
    }

    void process_data(std::vector<hphs_mix::SearchOutput> &&search_outputs) {
        // Record cluster sample distances
        for (const auto &search_output : search_outputs) {
            for (const auto &distance : search_output.cluster_distances) {
                cluster_distance_buffer.add(distance);
            }
        }
        _process_data_vqvae(search_outputs);
        _process_data_subgoal(search_outputs);
        _process_data_conditional_low(search_outputs);
        _process_data_low(search_outputs);
    }

    template <typename ModelT, typename LearningInputT>
    void _learning_step(
        ModelT &model,
        std::vector<LearningInputT> &learning_inputs,
        int steps,
        std::mt19937 &rng,
        const std::string &loss_type
    ) {
        if (!learning_inputs.empty()) {
            for (int i = 0; i < steps; ++i) {
                std::shuffle(learning_inputs.begin(), learning_inputs.end(), rng);
                auto batched_input = split_to_batch(learning_inputs, learning_batch_size);
                double loss = 0;
                for (auto &batch_item : batched_input) {
                    loss += model->Learn(batch_item);
                }
                SPDLOG_INFO("{:s} Loss: {:.4f}", loss_type, loss / batched_input.size());
            }
        }
    }

    void _learning_step_vqvae(int steps, std::mt19937 &rng) {
        if (!learning_inputs_vqvae.empty()) {
            for (int i = 0; i < steps; ++i) {
                std::shuffle(learning_inputs_vqvae.begin(), learning_inputs_vqvae.end(), rng);
                auto batched_input = split_to_batch(learning_inputs_vqvae, learning_batch_size / 4);
                double loss = 0;
                for (auto &batch_item : batched_input) {
                    loss += model_vsc->LearnVQVAE(batch_item);
                }
                SPDLOG_INFO("VQVAE Loss: {:.4f}", loss / batched_input.size());
            }
        }
    }

    void _learning_step_subgoal(int steps, std::mt19937 &rng) {
        if (!learning_inputs_subgoal.empty()) {
            for (int i = 0; i < steps; ++i) {
                std::shuffle(learning_inputs_subgoal.begin(), learning_inputs_subgoal.end(), rng);
                auto batched_input = split_to_batch(learning_inputs_subgoal, learning_batch_size);
                double loss = 0;
                int counter = 0;
                for (auto &batch_item : batched_input) {
                    loss += model_vsc->LearnSubgoal(batch_item);
                    if (++counter == 5) {
                        break;
                    }
                }
                SPDLOG_INFO("Subgoal Loss: {:.4f}", loss / batched_input.size());
            }
        }
    }

    void _learning_step_conditional_low(int steps, std::mt19937 &rng) {
        if (!learning_inputs_conditional_low.empty()) {
            for (int i = 0; i < steps; ++i) {
                std::shuffle(learning_inputs_conditional_low.begin(), learning_inputs_conditional_low.end(), rng);
                auto batched_input = split_to_batch(learning_inputs_conditional_low, learning_batch_size);
                double loss = 0;
                int counter = 0;
                for (auto &batch_item : batched_input) {
                    loss += model_vsc->LearnConditionalLow(batch_item);
                    if (++counter == 5) {
                        break;
                    }
                }
                SPDLOG_INFO("Conditional Loss: {:.4f}", loss / batched_input.size());
            }
        }
    }

    void learning_step(std::mt19937 &rng) {
        _learning_step_vqvae(grad_steps, rng);
        _learning_step_subgoal(grad_steps / 2, rng);
        _learning_step_conditional_low(grad_steps, rng);
        _learning_step(model_low, learning_inputs_low, grad_steps, rng, "Low");
    }

private:
    std::shared_ptr<VSCWrapperT> model_vsc;
    std::shared_ptr<TwoHeadedConvNetWrapper> model_low;
    int learning_batch_size;
    int grad_steps;
    std::vector<typename VSCWrapperT::VQVAELearningInput> learning_inputs_vqvae;
    std::vector<typename VSCWrapperT::SubgoalLearningInput> learning_inputs_subgoal;
    std::vector<typename VSCWrapperT::ConditionalLowLearningInput> learning_inputs_conditional_low;
    std::vector<typename TwoHeadedConvNetWrapper::LearningInput> learning_inputs_low;
    CircularBuffer<int> cluster_distance_buffer;
    ReplayBuffer<typename VSCWrapperT::VQVAELearningInput> replay_buffer;
};

auto init_model_twoheaded_convnet(
    const ObservationShape &obs_shape,
    int num_actions,
    const json &model_config_json,
    const std::string &base_name = ""
) -> std::shared_ptr<TwoHeadedConvNetWrapper> {
    TwoHeadedConvNetConfig model_config{
        obs_shape,
        num_actions,
        model_config_json["resnet_channels"].template get<int>(),
        model_config_json["resnet_blocks"].template get<int>(),
        model_config_json["policy_channels"].template get<int>(),
        model_config_json["heuristic_channels"].template get<int>(),
        model_config_json["policy_mlp_layers"].template get<std::vector<int>>(),
        model_config_json["heuristic_mlp_layers"].template get<std::vector<int>>(),
        model_config_json["use_batchnorm"].template get<bool>(),
    };
    return std::make_shared<TwoHeadedConvNetWrapper>(
        model_config,
        model_config_json["learning_rate"].template get<double>(),
        model_config_json["l2_weight_decay"].template get<double>(),
        absl::StrFormat("cuda:%d", absl::GetFlag(FLAGS_device_num)),
        absl::GetFlag(FLAGS_output_dir),
        base_name
    );
}

template <hphs_mix::VSCWrapper VSCWrapperT>
auto init_model_vsc(
    const ObservationShape &obs_shape,
    int num_actions,
    const json &model_config_json,
    const std::string &base_name = ""
) -> std::shared_ptr<VSCWrapperT> {
    std::unordered_map<int, float> recon_weights;
    for (const auto &[k, v] : model_config_json["vqvae"]["recon_weights"].items()) {
        recon_weights[std::stoi(k)] = v;
    }
    VSCConfig model_config{
        obs_shape,
        num_actions,
        {// Subgoal
         .resnet_channels = model_config_json["subgoal"]["resnet_channels"].template get<int>(),
         .resnet_blocks = model_config_json["subgoal"]["resnet_blocks"].template get<int>(),
         .reduce_channels = model_config_json["subgoal"]["reduce_channels"].template get<int>(),
         .mlp_layers = model_config_json["subgoal"]["mlp_layers"].template get<std::vector<int>>(),
         .use_batchnorm = model_config_json["subgoal"]["use_batchnorm"].template get<bool>()
        },
        {// Conditional low
         .resnet_channels = model_config_json["conditional_low"]["resnet_channels"].template get<int>(),
         .resnet_blocks = model_config_json["conditional_low"]["resnet_blocks"].template get<int>(),
         .reduce_channels = model_config_json["conditional_low"]["reduce_channels"].template get<int>(),
         .mlp_layers = model_config_json["conditional_low"]["mlp_layers"].template get<std::vector<int>>(),
         .use_batchnorm = model_config_json["conditional_low"]["use_batchnorm"].template get<bool>()
        },
        {// VQVAE
         .resnet_channels = model_config_json["vqvae"]["resnet_channels"].template get<int>(),
         .embedding_dim = model_config_json["vqvae"]["embedding_dim"].template get<int>(),
         .num_embeddings = model_config_json["vqvae"]["num_embeddings"].template get<int>(),
         .recon_weights = recon_weights,
         .use_ema = model_config_json["vqvae"]["use_ema"].template get<bool>(),
         .decay = model_config_json["vqvae"]["decay"].template get<double>(),
         .epsilon = model_config_json["vqvae"]["epsilon"].template get<double>(),
         .beta = model_config_json["vqvae"]["beta"].template get<double>()
        }
    };
    return std::make_shared<VSCWrapperT>(
        model_config,
        model_config_json["learning_rate"].template get<double>(),
        model_config_json["l2_weight_decay"].template get<double>(),
        absl::StrFormat("cuda:%d", absl::GetFlag(FLAGS_device_num)),
        absl::GetFlag(FLAGS_output_dir),
        base_name
    );
}

using VSCVariant = std::variant<std::shared_ptr<VSCWrapper>, std::shared_ptr<VSCFlatWrapper>>;

auto init_model_vsc(const ObservationShape &obs_shape, int num_actions) -> VSCVariant {
    std::ifstream f(absl::GetFlag(FLAGS_model_vsc_path));
    json model_config_json = json::parse(f);

    // Create model
    if (model_config_json["model_type"].template get<std::string>() == VSCWrapper::name) {
        return init_model_vsc<VSCWrapper>(obs_shape, num_actions, model_config_json, "vsc");
    } else if (model_config_json["model_type"].template get<std::string>() == VSCFlatWrapper::name) {
        return init_model_vsc<VSCFlatWrapper>(obs_shape, num_actions, model_config_json, "vsc");
    } else {
        SPDLOG_ERROR("Unknown vqvae model type");
        std::exit(1);
    }
}

auto init_model_low(const ObservationShape &obs_shape, int num_actions) {
    std::ifstream f(absl::GetFlag(FLAGS_model_low_path));
    json model_config_json = json::parse(f);

    // Create model
    if (model_config_json["model_type"].template get<std::string>() == TwoHeadedConvNetWrapper::name) {
        return init_model_twoheaded_convnet(obs_shape, num_actions, model_config_json);
    } else {
        SPDLOG_ERROR("Unknown low model type");
        std::exit(1);
    }
}

template <typename EnvT, algorithm::hphs_mix::VSCWrapper VSCWrapperT>
void templated_main(
    std::vector<EnvT> &problems,
    std::shared_ptr<VSCWrapperT> &model_vsc,
    std::shared_ptr<TwoHeadedConvNetWrapper> &model_low
) {
    using SearchInputT = hphs_mix::SearchInput<EnvT, VSCWrapperT>;
    using SearchOutputT = hphs_mix::SearchOutput;
    using LearningInputT = TwoHeadedConvNetWrapper::LearningInput;

    if (absl::GetFlag(FLAGS_resume)) {
        model_vsc->LoadCheckpointWithoutOptimizer(-1);
        model_low->LoadCheckpointWithoutOptimizer(-1);
    }
    model_vsc->print();
    model_low->print();
    std::shared_ptr<StopToken> stop_token = signal_installer();

    // Learner
    HPHSLearner<EnvT, VSCWrapperT>
        hphs_learner(model_vsc, model_low, absl::GetFlag(FLAGS_learning_batch_size), absl::GetFlag(FLAGS_grad_steps));

    // Create search inputs
    auto [problems_train, problems_validate] = create_search_inputs(problems, stop_token, model_vsc, model_low);

    for (auto &p : problems_validate) {
        p.create_cluster_graph = false;
    }

    std::mt19937 rng(absl::GetFlag(FLAGS_seed));
    train_bootstrap<SearchInputT, SearchOutputT>(
        problems_train,
        problems_validate,
        hphs_mix::search<EnvT, VSCWrapperT>,
        hphs_learner,
        absl::GetFlag(FLAGS_output_dir),
        rng,
        stop_token,
        absl::GetFlag(FLAGS_search_budget),
        absl::GetFlag(FLAGS_validation_solved_ratio),
        absl::GetFlag(FLAGS_num_threads),
        absl::GetFlag(FLAGS_num_problems_per_batch),
        absl::GetFlag(FLAGS_max_iterations),
        absl::GetFlag(FLAGS_max_budget),
        absl::GetFlag(FLAGS_time_budget),
        static_cast<BootstrapPolicy>(absl::GetFlag(FLAGS_bootstrap_policy)),
        absl::GetFlag(FLAGS_bootstrap_factor),
        absl::GetFlag(FLAGS_extra_iterations),
        absl::GetFlag(FLAGS_resume)
    );
}

// helper type for the visitor
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <typename EnvT>
void templated_main() {
    // Load problems
    auto [problems, _] = load_problems<EnvT>(absl::GetFlag(FLAGS_problems_path));

    auto model_vsc = init_model_vsc(problems[0].observation_shape(), problems[0].num_actions);
    auto model_low = init_model_low(problems[0].observation_shape(), problems[0].num_actions);

    std::visit(
        overloaded{
        [&](std::shared_ptr<VSCWrapper> &_model_vsc) {
            templated_main<EnvT, VSCWrapper>(problems, _model_vsc, model_low);
        },
        [&](std::shared_ptr<VSCFlatWrapper> &_model_vsc) {
            templated_main<EnvT, VSCFlatWrapper>(problems, _model_vsc, model_low);
        }
        },
        model_vsc
    );
}

}    // namespace

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(absl::GetFlag(FLAGS_output_dir));

    // Initialize torch and loggers (console + file)
    // hpts::init_torch(0);
    hpts::init_torch(absl::GetFlag(FLAGS_seed));
    hpts::init_loggers(false, absl::GetFlag(FLAGS_output_dir), "_train");

    // Dump invocation of program
    hpts::log_flags(argc, argv);

    if (absl::GetFlag(FLAGS_environment) == bd::BoulderDashTreeState::name) {
        templated_main<bd::BoulderDashTreeState>();
    } else if (absl::GetFlag(FLAGS_environment) == cw::CraftWorldTreeState::name) {
        templated_main<cw::CraftWorldTreeState>();
    } else if (absl::GetFlag(FLAGS_environment) == bw::BoxWorldTreeState::name) {
        templated_main<bw::BoxWorldTreeState>();
    } else if (absl::GetFlag(FLAGS_environment) == env::sokoban::SokobanTreeState::name) {
        templated_main<env::sokoban::SokobanTreeState>();
    } else if (absl::GetFlag(FLAGS_environment) == env::tsp::TSPTreeState::name) {
        templated_main<env::tsp::TSPTreeState>();
    } else {
        SPDLOG_ERROR("Unknown environment type: {:s}.", absl::GetFlag(FLAGS_environment));
        std::exit(1);
    }

    hpts::close_loggers();
}
