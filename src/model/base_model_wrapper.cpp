// File: base_model_wrapper.cpp
// Description: Holds model + optimizer to directly interface with nn::Module for inference + learning

#include "model/base_model_wrapper.h"

// NOLINTBEGIN
#include <absl/strings/str_cat.h>
// NOLINTEND

namespace hpts::model {

BaseModelWrapper::BaseModelWrapper(const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name)
    : device_(device),
      path_(absl::StrCat(output_path, "/checkpoints/")),
      checkpoint_base_name_(
          checkpoint_base_name.empty() ? checkpoint_base_name : absl::StrCat(checkpoint_base_name, "-")),
      torch_device_(device) {}

void BaseModelWrapper::LoadCheckpoint(long long int step) {
    LoadCheckpoint(absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step));
}

void BaseModelWrapper::LoadCheckpointWithoutOptimizer(long long int step) {
    LoadCheckpointWithoutOptimizer(absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step));
}

}    // namespace hpts::model
