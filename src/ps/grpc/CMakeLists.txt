find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

set(BUILD_SHARED_LIBS ON)

# gRPC PS Client Library
add_library(grpc_ps_client SHARED
    grpc_ps_client.cpp grpc_ps_client.h
)


target_link_libraries(
    grpc_ps_client PUBLIC
    base
    ps_base
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    report_client
)
target_compile_definitions(grpc_ps_client PUBLIC ENABLE_PERF_REPORT)

# gRPC PS Client Test
add_executable(grpc_ps_client_test
    grpc_ps_client_test.cpp
)
target_link_libraries(grpc_ps_client_test PUBLIC
    grpc_ps_client
)

# gRPC PS Server
add_executable(grpc_ps_server
    grpc_ps_server.cpp
    grpc_ps_server.h
)

target_link_libraries(grpc_ps_server PUBLIC
    base
    ps_base
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    stdc++fs
    kv_engine
    sparse_tensor
    gflags
    glog
    TBB::tbb
    report_client
)
target_compile_definitions(grpc_ps_server PRIVATE ENABLE_PERF_REPORT)

# Distributed gRPC PS Client
add_library(dist_grpc_ps_client
    dist_grpc_ps_client.cpp dist_grpc_ps_client.h
)
target_link_libraries(dist_grpc_ps_client PUBLIC
    base
    ps_base
    grpc_ps_client
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    report_client
)
target_compile_definitions(dist_grpc_ps_client PUBLIC ENABLE_PERF_REPORT)

# Distributed gRPC PS Client Test
add_executable(dist_grpc_ps_client_test
    dist_grpc_ps_client_test.cpp
)
target_link_libraries(dist_grpc_ps_client_test PUBLIC
    dist_grpc_ps_client
)

# gRPC PS Client Python Binding
add_library(grpc_ps_client_python MODULE
    grpc_ps_client_pytorch.cpp
)

target_include_directories(grpc_ps_client_python PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(grpc_ps_client_python PUBLIC
    grpc_ps_client
    ${TORCH_LIBRARIES}
    Python3::Python
)
