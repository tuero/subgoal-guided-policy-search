// File: heuristic_convent.h
// Description: Convnet for heuristic predictions

#ifndef HPTS_MODEL_HEURISTIC_CONVNET_H_
#define HPTS_MODEL_HEURISTIC_CONVNET_H_

#include "common/observation.h"
#include "model/layers.h"

#include <vector>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts::model::network {

struct HeuristicConvNetOutput {
    torch::Tensor heuristic;
};

class HeuristicConvNetImpl : public torch::nn::Module {
public:
    /**
     * ResNet style heuristic convnet
     * @param observation_shape Input observation shape to the network
     * @param resnet_channels Number of channels for each resenet block
     * @param resnet_blocks Number of resnet blocks
     * @param reduce_channels Number of channels in the heuristic reduce head
     * @param mlp_layers Hidden layer sizes for the heuristic head MLP
     * @param use_batchnorm Flag to use batchnorm in the resnet layers
     */
    HeuristicConvNetImpl(
        const ObservationShape &observation_shape,
        int resnet_channels,
        int resnet_blocks,
        int reduce_channels,
        const std::vector<int> &mlp_layers,
        bool use_batchnorm
    );
    [[nodiscard]] auto forward(torch::Tensor x) -> HeuristicConvNetOutput;

private:
    int input_channels_;
    int input_height_;
    int input_width_;
    int resnet_channels_;
    int reduce_channels_;
    int mlp_input_size_;
    ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_;    // Conv pass before passing to heuristic mlp
    MLP mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(HeuristicConvNet);

}    // namespace hpts::model::network

#endif    // HPTS_MODEL_HEURISTIC_CONVNET_H_
