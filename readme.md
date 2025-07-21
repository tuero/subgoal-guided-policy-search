
## Environments
The following environments and problem instances were used, which have both C++/Python bindings.
These are installed automatically when configuring the installation with cmake.
- [Boulderdash](https://github.com/tuero/boulderdash_cpp)
- [Boxworld](https://github.com/tuero/boxworld_cpp)
- [Craftworld](https://github.com/tuero/craftworld_cpp_v2)
- [Sokoban](https://github.com/tuero/sokoban_cpp)
- [TSP](https://github.com/tuero/tsp_cpp/)


## Installing

This library relies on C++23 features, and was tested with `g++14`

```shell
# Get libtorch through pip distribution
conda create -n sgps python=3.11
conda activate sgps
pip3 install torch torchvision torchaudio

git clone https://github.com/tuero/subgoal-guided-policy-search.git
cd subgoal-guided-policy-search
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j$(nproc)
```

## Example Training Run
```shell
# cd into the build directory
CUBLAS_WORKSPACE_CONFIG=:16:8 ./src/apps/sgps/sgps_train --environment=tsp --problems_path=<PATH/TO/PROBLEMS>/tsp/train.txt --output_dir=<PATH/TO/EXPERIMENTS>/sgps_tsp_hard_s0 --model_vsc_path=<PATH/TO/MODELS>/vsc_flat.json --model_low_path=<PATH/TO/MODELS>/twoheaded_convnet_low.json --search_budget=4000 --inference_batch_size=32 --mix_epsilon=0.01 --seed=0 --num_train=10000 --num_validate=1000 --max_iterations=30 --grad_steps=10 --learning_batch_size=256 --num_problems_per_batch=32 --mix_low_alpha=0.25 --rho=1 --num_cluster_samples=5 --num_threads=8 --cluster_level=5 --bootstrap_policy=1 --validation_solved_ratio=0.9 --device_num=0
```

## Example Test Run
```shell
# cd into the build directory
CUBLAS_WORKSPACE_CONFIG=:16:8 ./src/apps/sgps/sgps_test --environment=tsp --problems_path=<PATH/TO/PROBLEMS>/tsp/test_100.txt --output_dir=<PATH/TO/EXPERIMENTS>/sgps_tsp_hard_s0 --model_vsc_path=<PATH/TO/MODELS>/vsc_flat.json --model_low_path=<PATH/TO/MODELS>/twoheaded_convnet_low.json --search_budget=4000 --inference_batch_size=1 --mix_epsilon=0.01 --max_iterations=11 --num_threads=8 --export_suffix=b1 --mix_low_alpha=0.5
```
