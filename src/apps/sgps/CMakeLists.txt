# SGPS Mix Clustering
add_executable(sgps_train sgps_train.cpp)
target_link_libraries(sgps_train hpts_core)
target_compile_features(sgps_train PUBLIC cxx_std_23)

add_executable(sgps_test sgps_test.cpp)
target_link_libraries(sgps_test hpts_core)
target_compile_features(sgps_test PUBLIC cxx_std_23)
