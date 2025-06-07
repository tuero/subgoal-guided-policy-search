cmake_minimum_required (VERSION 3.25)

project(sugoal_guided_policy_search)

include(FetchContent)

# Required for python modules
set(ABSL_PROPAGATE_CXX_STD ON)

# Set global and local warnings
add_library(common_interface INTERFACE)
if(MSVC)
    message("Windows support is not tested")
    target_compile_options(common_interface INTERFACE 
        /W4 /permissive
    )
else()
    target_compile_options(common_interface INTERFACE 
        -Wall -Wextra -pedantic 
        -Wsign-conversion -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wunused
    )
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -O0 -fno-omit-frame-pointer")
elseif(CMAKE_BUILD_TYPE STREQUAL "Testing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fno-omit-frame-pointer")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O2 -g -fno-omit-frame-pointer")
endif()

# Environment dependencies
# Boxworld
message("Configuring BoxWorld")
FetchContent_Declare(boxworld
    GIT_REPOSITORY https://github.com/tuero/boxworld_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(boxworld)

# Craftworld
message("Configuring CraftWorld")
FetchContent_Declare(craftworld
    GIT_REPOSITORY https://github.com/tuero/craftworld_cpp_v2.git
    GIT_TAG master
)
FetchContent_MakeAvailable(craftworld)

# Sokoban
message("Configuring Sokoban")
FetchContent_Declare(sokoban
    GIT_REPOSITORY https://github.com/tuero/sokoban_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(sokoban)

# StonesNGems
message("Configuring Boulderdash")
FetchContent_Declare(boulderdash
    GIT_REPOSITORY https://github.com/tuero/boulderdash_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(boulderdash)

# TSP
message("Configuring TSP")
FetchContent_Declare(tsp
    GIT_REPOSITORY https://github.com/tuero/tsp_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(tsp)

# Abseil
message("Configuring absl")
FetchContent_Declare(absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG master
    SYSTEM
)
FetchContent_MakeAvailable(absl)

# spdlog
message("Configuring spdlog")
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
    SYSTEM
)
FetchContent_GetProperties(spdlog)
if (NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
    add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif()

# json
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)


# libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Project directories
add_subdirectory(src)
