find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

set(REPORT_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libreport.so")
if(EXISTS "${REPORT_LIB_PATH}")
    set(REPORT_LIB "${REPORT_LIB_PATH}")
    message(STATUS "Found libreport.so at ${REPORT_LIB}")
else()
    message(WARNING "libreport.so not found at ${REPORT_LIB_PATH}, performance reporting will not work")
    set(REPORT_LIB "")
endif()

add_subdirectory(base)
add_subdirectory(proto)
add_subdirectory(grpc)
add_subdirectory(brpc)
add_subdirectory(rdma)

if(brpc_FOUND)
    add_executable(ps_server
        ps_server.cpp
    )
    target_compile_definitions(ps_server PRIVATE RECSTORE_NO_SERVER_MAIN=1)
    target_include_directories(ps_server PUBLIC ${BRPC_INCLUDE_PATH})
    
    target_link_libraries(ps_server PUBLIC
        base
        ps_base
        ps_proto
        # gRPC dependencies
        gRPC::grpc++_reflection
        gRPC::grpc++
        protobuf::libprotobuf
        # bRPC dependencies
        ${BRPC_LIB}
        ${CUSTOM_PROTOBUF_LIB}
        ${SYSTEM_OPENSSL_SSL_LIB}
        ${SYSTEM_OPENSSL_CRYPTO_LIB}
        ZLIB::ZLIB
        Threads::Threads
        dl
        # Common dependencies
        stdc++fs
        kv_engine
        sparse_tensor
        gflags
        glog
        leveldb
        TBB::tbb
        ${REPORT_LIB}
    )
    
    target_sources(ps_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc/grpc_ps_server.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/brpc/brpc_ps_server.cpp
    )
endif()


