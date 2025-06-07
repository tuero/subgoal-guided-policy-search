// File: policy_convnet_wrapper.h
// Description: Convnet wrapper for policy

#ifndef HPTS_WRAPPER_POLICY_CONVNET_H_
#define HPTS_WRAPPER_POLICY_CONVNET_H_

#include "common/observation.h"
#include "model/base_model_wrapper.h"
#include "model/policy_convnet.h"

namespace hpts::model::wrapper {

struct PolicyConvNetConfig {
    ObservationShape observation_shape;
    int num_actions;
    int resnet_channels;
    int resnet_blocks;
    int policy_channels;
    std::vector<int> policy_mlp_layers;
    bool use_batchnorm;
};

class PolicyConvNetWrapper : public BaseModelWrapper {
public:
    inline static const std::string name = "policy_convnet";

    struct InferenceInput {
        Observation observation;
    };
    struct InferenceRefInput {
        Observation &observation;    // NOLINT
    };

    struct InferenceOutput {
        std::vector<double> logits;
        std::vector<double> policy;
        std::vector<double> log_policy;
    };

    struct LearningInput {
        Observation observation;
        int target_action;
        int solution_expanded;
    };

    PolicyConvNetWrapper(
        PolicyConvNetConfig config,
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
    auto Learn(std::vector<LearningInput> &batch) -> double;

protected:
    template <typename InferenceInputT>
        requires IsAny<InferenceInputT, InferenceInput, InferenceRefInput>
    [[nodiscard]] auto inference(std::vector<InferenceInputT> &batch) -> std::vector<InferenceOutput>;

    PolicyConvNetConfig config;
    network::PolicyConvNet model_;
    torch::optim::Adam model_optimizer_;
    int input_flat_size;
    int num_actions;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_POLICY_CONVNET_H_
