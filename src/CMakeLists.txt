include_directories(.)

# Link to all children
link_libraries(boulderdash)
link_libraries(sokoban)
link_libraries(craftworld)
link_libraries(boxworld)
link_libraries(tsp)

link_libraries(${TORCH_LIBRARIES})
link_libraries(spdlog::spdlog_header_only)
link_libraries( 
    absl::base
    absl::flags
    absl::flags_parse
    absl::hash
    absl::strings
    absl::synchronization
    absl::optional
    absl::time
)
link_libraries(nlohmann_json::nlohmann_json)

add_library(hpts_core STATIC
    common/init.cpp
    common/metrics_tracker.cpp
    common/timer.cpp
    common/signaller.cpp
    env/boxworld.cpp
    env/craftworld.cpp
    env/boulderdash.cpp
    env/sokoban.cpp
    env/tsp.cpp
    model/base_model_wrapper.cpp
    model/heuristic_convnet.cpp
    model/heuristic_convnet_wrapper.cpp
    model/layers.cpp
    model/policy_convnet.cpp
    model/policy_convnet_wrapper.cpp
    model/twoheaded_convnet.cpp
    model/twoheaded_convnet_wrapper.cpp
    model/vsc.cpp 
    model/vsc_wrapper.cpp
    model/vqvae.cpp
    util/utility.cpp
)
target_compile_features(hpts_core PUBLIC cxx_std_23)

add_subdirectory(apps)
