#include "algorithm/sgps.h"
#include "algorithm/test_runner.h"
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
#include "util/stop_token.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>

#include <filesystem>
#include <memory>
#include <string>

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
ABSL_FLAG(double, mix_low_alpha, 0.5, "Mixing alpha for low level policy with conditional mixture");
ABSL_FLAG(double, rho, 1.0, "Rho parameter for Louvain algorithm which controls important of in edges");
ABSL_FLAG(int, num_cluster_samples, 1, "Number of samples to create training input on the Louvain clustering");
ABSL_FLAG(int, max_iterations, INF_I, "Maximum number of iterations of running the bootstrap process");
ABSL_FLAG(double, time_budget, INF_D, "Budget in seconds before terminating");
ABSL_FLAG(int, num_threads, 1, "Number of threads to run in the search thread pool");
ABSL_FLAG(std::string, export_suffix, "", "Export suffix to place on output logs/files");
ABSL_FLAG(bool, use_heuristic, true, "True to use a heuristic for PHS, false for LevinTS");
// NOLINTEND

namespace {

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
            -1,
            1,
            0,
            static_cast<unsigned long long int>(i) * search_inputs.size(),
            false,
            stop_token,
            model_vsc,
            model_low
        );
    }
    return search_inputs;
}

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
        "cuda:0",
        absl::GetFlag(FLAGS_output_dir),
        base_name
    );
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
        "cuda:0",
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

template <typename EnvT, algorithm::hphs_mix::VSCWrapper VSCWrapperT>
void templated_main(
    std::vector<EnvT> &problems,
    std::shared_ptr<VSCWrapperT> &model_vsc,
    std::shared_ptr<TwoHeadedConvNetWrapper> &model_low
) {
    using SearchInputT = hphs_mix::SearchInput<EnvT, VSCWrapperT>;
    using SearchOutputT = hphs_mix::SearchOutput;
    using LearningInputT = TwoHeadedConvNetWrapper::LearningInput;

    model_vsc->LoadCheckpointWithoutOptimizer(-1);
    model_low->LoadCheckpointWithoutOptimizer(-1);

    model_vsc->print();
    model_low->print();
    std::shared_ptr<StopToken> stop_token = signal_installer();

    // Create search inputs
    auto search_inputs = create_search_inputs(problems, stop_token, model_vsc, model_low);

    test_runner<SearchInputT, SearchOutputT>(
        search_inputs,
        hphs_mix::search<EnvT, VSCWrapperT>,
        absl::GetFlag(FLAGS_output_dir),
        stop_token,
        absl::GetFlag(FLAGS_search_budget),
        absl::GetFlag(FLAGS_num_threads),
        absl::GetFlag(FLAGS_max_iterations),
        absl::GetFlag(FLAGS_time_budget),
        absl::GetFlag(FLAGS_export_suffix)
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
    hpts::init_torch(0);
    std::string export_suffix = absl::GetFlag(FLAGS_export_suffix);
    if (export_suffix != "") {
        export_suffix = absl::StrCat("_", export_suffix);
    }
    hpts::init_loggers(false, absl::GetFlag(FLAGS_output_dir), absl::StrCat("_test", export_suffix));

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
