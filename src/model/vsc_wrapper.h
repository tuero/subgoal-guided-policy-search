// File: vsc_wrapper.h
// Description: A wrapper over a VQVAE, Subgoal, and conditional low model

#ifndef HPTS_WRAPPER_VSC_H_
#define HPTS_WRAPPER_VSC_H_

#include "common/observation.h"
#include "model/base_model_wrapper.h"
#include "model/vsc.h"

namespace hpts::model::wrapper {

using VSCConfig = network::VSCConfig;

class VSCFlatWrapper : public BaseModelWrapper {
public:
    inline static const std::string name = "vsc_flat";

    struct InferenceInput {
        Observation obs;
    };
    struct InferenceRefInput {
        Observation &obs;    // NOLINT
    };

    struct InferenceOutput {
        std::vector<double> subgoal_policy;
        std::vector<std::vector<double>> conditional_low_policies;
    };

    struct VQVAEQuantizeInput {
        Observation obs_input;
        Observation obs_target;
    };
    struct VQVAEQuantizeRefInput {
        Observation &obs_input;     // NOLINT
        Observation &obs_target;    // NOLINT
    };

    struct VQVAEQuantizeOutput {
        Observation quantized_obs;
    };

    struct VQVAELearningInput {
        Observation obs_input;
        Observation obs_target;
    };
    struct SubgoalLearningInput {
        Observation obs_input;
        Observation obs_target;
        int solution_expanded;
    };
    struct ConditionalLowLearningInput {
        Observation obs_input;
        Observation obs_target;
        int target_action;
        int solution_expanded;
    };

    VSCFlatWrapper(
        VSCConfig config,
        double learning_rate,
        double l2_weight_decay,
        const std::string &device,
        const std::string &output_path,
        const std::string &checkpoint_base_name = ""
    );

    void print() const override;

    auto SaveCheckpoint(long long int step = -1) -> std::string override;
    auto SaveCheckpointWithoutOptimizer(long long int step = -1) -> std::string override;

    using BaseModelWrapper::LoadCheckpoint;
    using BaseModelWrapper::LoadCheckpointWithoutOptimizer;
    void LoadCheckpoint(const std::string &path) override;
    void LoadCheckpointWithoutOptimizer(const std::string &path) override;

    /**
     * Perform inference
     * @param inputs Batched observations (implementation defined)
     * @returns Implementation defined output
     */
    [[nodiscard]] auto Inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput>;
    [[nodiscard]] auto Inference(std::vector<InferenceRefInput> &batch) -> std::vector<InferenceOutput>;
    [[nodiscard]] auto Quantize(std::vector<VQVAEQuantizeInput> &batch) -> std::vector<VQVAEQuantizeOutput>;
    [[nodiscard]] auto Quantize(std::vector<VQVAEQuantizeRefInput> &batch) -> std::vector<VQVAEQuantizeOutput>;

    /**
     * Perform a model update learning step
     * @param batch Batched learning input
     * @returns Loss for current batch
     */
    auto LearnVQVAE(std::vector<VQVAELearningInput> &batch) -> double;
    auto LearnSubgoal(std::vector<SubgoalLearningInput> &batch) -> double;
    auto LearnConditionalLow(std::vector<ConditionalLowLearningInput> &batch) -> double;

protected:
    template <typename InferenceInputT>
        requires IsAny<InferenceInputT, InferenceInput, InferenceRefInput>
    [[nodiscard]] auto inference(std::vector<InferenceInputT> &batch) -> std::vector<InferenceOutput>;

    template <typename QuantizeInputT>
        requires IsAny<QuantizeInputT, VQVAEQuantizeInput, VQVAEQuantizeRefInput>
    [[nodiscard]] auto quantize(std::vector<QuantizeInputT> &batch) -> std::vector<VQVAEQuantizeOutput>;

    VSCConfig config;
    network::VSCFlat model_;
    torch::optim::Adam vqvae_optimizer_;
    torch::optim::Adam subgoal_optimizer_;
    torch::optim::Adam conditional_low_optimizer_;
    double beta;
    bool use_ema;
};

// --------------------------------------------

class VSCWrapper : public BaseModelWrapper {
public:
    inline static const std::string name = "vsc";

    struct InferenceInput {
        Observation obs;
    };
    struct InferenceRefInput {
        Observation &obs;    // NOLINT
    };

    struct InferenceOutput {
        std::vector<double> subgoal_policy;
        std::vector<std::vector<double>> conditional_low_policies;
    };

    struct VQVAEQuantizeInput {
        Observation obs_input;
        Observation obs_target;
    };
    struct VQVAEQuantizeRefInput {
        Observation &obs_input;     // NOLINT
        Observation &obs_target;    // NOLINT
    };

    struct VQVAEQuantizeOutput {
        Observation quantized_obs;
    };

    struct VQVAELearningInput {
        Observation obs_input;
        Observation obs_target;
    };
    struct SubgoalLearningInput {
        Observation obs_input;
        Observation obs_target;
        int solution_expanded;
    };
    struct ConditionalLowLearningInput {
        Observation obs_input;
        Observation obs_target;
        int target_action;
        int solution_expanded;
    };

    VSCWrapper(
        VSCConfig config,
        double learning_rate,
        double l2_weight_decay,
        const std::string &device,
        const std::string &output_path,
        const std::string &checkpoint_base_name = ""
    );

    void print() const override;

    auto SaveCheckpoint(long long int step = -1) -> std::string override;
    auto SaveCheckpointWithoutOptimizer(long long int step = -1) -> std::string override;

    using BaseModelWrapper::LoadCheckpoint;
    using BaseModelWrapper::LoadCheckpointWithoutOptimizer;
    void LoadCheckpoint(const std::string &path) override;
    void LoadCheckpointWithoutOptimizer(const std::string &path) override;

    /**
     * Perform inference
     * @param inputs Batched observations (implementation defined)
     * @returns Implementation defined output
     */
    [[nodiscard]] auto Inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput>;
    [[nodiscard]] auto Inference(std::vector<InferenceRefInput> &batch) -> std::vector<InferenceOutput>;

    /**
     * Perform a model update learning step
     * @param batch Batched learning input
     * @returns Loss for current batch
     */
    auto LearnVQVAE(std::vector<VQVAELearningInput> &batch) -> double;
    auto LearnSubgoal(std::vector<SubgoalLearningInput> &batch) -> double;
    auto LearnConditionalLow(std::vector<ConditionalLowLearningInput> &batch) -> double;

protected:
    template <typename InferenceInputT>
        requires IsAny<InferenceInputT, InferenceInput, InferenceRefInput>
    [[nodiscard]] auto inference(std::vector<InferenceInputT> &batch) -> std::vector<InferenceOutput>;

    VSCConfig config;
    network::VSC model_;
    torch::optim::Adam vqvae_optimizer_;
    torch::optim::Adam subgoal_optimizer_;
    torch::optim::Adam conditional_low_optimizer_;
    double beta;
    bool use_ema;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_VSC_H_
