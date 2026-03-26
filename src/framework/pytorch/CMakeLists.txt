find_package(Torch REQUIRED)

add_library(recstore_torch_ops SHARED
    op_torch.cc
    ../op.cc
)

target_compile_definitions(recstore_torch_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

if(USE_FAKE_KVCLIENT)
    target_compile_definitions(recstore_torch_ops PRIVATE USE_FAKE_KVCLIENT)
endif()

target_include_directories(recstore_torch_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/ps/grpc
)

option(USE_PERF_REPORT "Enable performance reporting in Debug mode" OFF)

if(CMAKE_BUILD_TYPE MATCHES "Debug" AND USE_PERF_REPORT)
    target_compile_definitions(recstore_torch_ops PRIVATE ENABLE_PERF_REPORT)
    target_link_libraries(recstore_torch_ops PRIVATE report_client)
endif()

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
target_link_libraries(recstore_torch_ops PRIVATE
    grpc_ps_client
    brpc_ps_client
    protobuf::libprotobuf
    gRPC::grpc++
    glog::glog
    ${TORCH_LIBRARIES}
)

target_link_options(recstore_torch_ops PRIVATE -Wl,--no-as-needed)

# Set the required C++ standard.
set_target_properties(recstore_torch_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Set the output name of the library.
set_target_properties(recstore_torch_ops PROPERTIES
    OUTPUT_NAME "_recstore_ops"
)

