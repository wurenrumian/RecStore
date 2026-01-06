find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

add_subdirectory(proto)


set(BUILD_SHARED_LIBS ON)

add_library(grpc_ps_client SHARED
    grpc_ps_client.cpp grpc_ps_client.h
)

find_library(REPORT_LIB report PATHS ${CMAKE_CURRENT_SOURCE_DIR} NO_DEFAULT_PATH)
if(NOT REPORT_LIB)
    message(WARNING "libreport.so not found in ${CMAKE_CURRENT_SOURCE_DIR}, performance reporting will not work")
    set(REPORT_LIB "")
endif()

target_link_libraries(
    grpc_ps_client PUBLIC
    base
    base_ps
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    ${REPORT_LIB}
)

add_executable(grpc_ps_client_test
    grpc_ps_client_test.cpp
)
target_link_libraries(grpc_ps_client_test PUBLIC
    grpc_ps_client
)


add_executable(grpc_ps_server
    grpc_ps_server.cpp
    grpc_ps_server.h
)

target_link_libraries(grpc_ps_server PUBLIC
    base
    base_ps
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
    ${REPORT_LIB}
)

add_library(dist_grpc_ps_client
    dist_grpc_ps_client.cpp dist_grpc_ps_client.h
)
target_link_libraries(dist_grpc_ps_client PUBLIC
    base
    base_ps
    grpc_ps_client
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    ${REPORT_LIB}
)

add_executable(dist_grpc_ps_client_test
    dist_grpc_ps_client_test.cpp
)
target_link_libraries(dist_grpc_ps_client_test PUBLIC
    dist_grpc_ps_client
)

set(BRPC_ROOT "${PROJECT_SOURCE_DIR}/third_party/brpc-install")
set(PROTOBUF_ROOT "${PROJECT_SOURCE_DIR}/third_party/protobuf-install")



# [A] 查找 bRPC (禁止去系统找)
find_path(BRPC_INCLUDE_PATH
    NAMES brpc/server.h
    PATHS ${BRPC_ROOT}/include
    NO_DEFAULT_PATH
)

find_library(BRPC_LIB
    NAMES brpc libbrpc
    PATHS ${BRPC_ROOT}/lib ${BRPC_ROOT}/lib64
    NO_DEFAULT_PATH
)

# [B] 查找 Protobuf (禁止去系统找)
find_path(CUSTOM_PROTOBUF_INCLUDE
    NAMES google/protobuf/message.h
    PATHS ${PROTOBUF_ROOT}/include
    NO_DEFAULT_PATH
)
find_library(CUSTOM_PROTOBUF_LIB
    NAMES protobuf libprotobuf
    PATHS ${PROTOBUF_ROOT}/lib ${PROTOBUF_ROOT}/lib64
    NO_DEFAULT_PATH
)

# [C] 查找系统库 (OpenSSL, ZLIB)
# 直接限定在系统目录，避免链接到 third_party/grpc-install
# 必须使用系统的 否则在链接阶段会报错

find_path(SYSTEM_OPENSSL_INCLUDE
    NAMES openssl/ssl.h
    PATHS /usr/include
    NO_DEFAULT_PATH
)
find_library(SYSTEM_OPENSSL_SSL_LIB
    NAMES ssl
    PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)
find_library(SYSTEM_OPENSSL_CRYPTO_LIB
    NAMES crypto
    PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED) # for pthread

if(NOT SYSTEM_OPENSSL_INCLUDE OR NOT SYSTEM_OPENSSL_SSL_LIB OR NOT SYSTEM_OPENSSL_CRYPTO_LIB)
    message(FATAL_ERROR "Could NOT find system OpenSSL libraries under /usr")
endif()


if(BRPC_INCLUDE_PATH AND BRPC_LIB AND CUSTOM_PROTOBUF_INCLUDE AND CUSTOM_PROTOBUF_LIB)
    set(brpc_FOUND TRUE)
    message(STATUS "Found Custom bRPC: ${BRPC_LIB}")
    message(STATUS "Found Custom Protobuf: ${CUSTOM_PROTOBUF_LIB}")
    message(STATUS "Found System OpenSSL: ${SYSTEM_OPENSSL_SSL_LIB}")
else()
    message(FATAL_ERROR "Could NOT find custom bRPC or Protobuf! \n bRPC: ${BRPC_LIB} \n Proto: ${CUSTOM_PROTOBUF_LIB}")
endif()

if(brpc_FOUND)
    message(STATUS "Building bRPC clients and servers")
    
    add_library(brpc_ps_client SHARED
        brpc_ps_client.cpp brpc_ps_client.h
    )
    target_include_directories(brpc_ps_client PUBLIC ${BRPC_INCLUDE_PATH})
    
    target_link_libraries(brpc_ps_client PUBLIC
        base
        base_ps
        ps_proto
       
        ${BRPC_LIB}             
        ${CUSTOM_PROTOBUF_LIB}   
        ${SYSTEM_OPENSSL_SSL_LIB}     
        ${SYSTEM_OPENSSL_CRYPTO_LIB} 
        ZLIB::ZLIB              
        Threads::Threads         
        dl                       
        gflags
        leveldb
        ${REPORT_LIB}
    )


    add_executable(brpc_ps_client_test
        brpc_ps_client_test.cpp
    )
    target_link_libraries(brpc_ps_client_test PUBLIC
        brpc_ps_client
    )


    add_executable(brpc_ps_server
        brpc_ps_server.cpp
        brpc_ps_server.h
    )
    target_include_directories(brpc_ps_server PUBLIC ${BRPC_INCLUDE_PATH})
    target_link_libraries(brpc_ps_server PUBLIC
        base
        base_ps
        ps_proto
        ${BRPC_LIB}
        ${CUSTOM_PROTOBUF_LIB}
        ${SYSTEM_OPENSSL_SSL_LIB}
        ${SYSTEM_OPENSSL_CRYPTO_LIB}
        ZLIB::ZLIB
        Threads::Threads
        dl
        stdc++fs
        kv_engine
        gflags
        # glog
        leveldb
        TBB::tbb
        ${REPORT_LIB}
    )

    add_library(dist_brpc_ps_client
        dist_brpc_ps_client.cpp dist_brpc_ps_client.h
    )
    target_include_directories(dist_brpc_ps_client PUBLIC ${BRPC_INCLUDE_PATH})
    target_link_libraries(dist_brpc_ps_client PUBLIC
        base
        base_ps
        brpc_ps_client
        ps_proto
        ${BRPC_LIB}
        ${CUSTOM_PROTOBUF_LIB}
        ${SYSTEM_OPENSSL_SSL_LIB}
        ${SYSTEM_OPENSSL_CRYPTO_LIB}
        ZLIB::ZLIB
        leveldb
        dl
        ${REPORT_LIB}
    )

    add_executable(dist_brpc_ps_client_test
        dist_brpc_ps_client_test.cpp
    )
    target_link_libraries(dist_brpc_ps_client_test PUBLIC
        dist_brpc_ps_client
    )

    add_executable(ps_server
        ps_server.cpp
        grpc_ps_server.cpp
        brpc_ps_server.cpp
    )
    target_compile_definitions(ps_server PRIVATE RECSTORE_NO_SERVER_MAIN=1)
    target_include_directories(ps_server PUBLIC ${BRPC_INCLUDE_PATH})
    target_link_libraries(ps_server PUBLIC
        base
        base_ps
        ps_proto
        # gRPC side
        gRPC::grpc++_reflection
        gRPC::grpc++
        protobuf::libprotobuf
        # bRPC side
        ${BRPC_LIB}
        ${CUSTOM_PROTOBUF_LIB}
        ${SYSTEM_OPENSSL_SSL_LIB}
        ${SYSTEM_OPENSSL_CRYPTO_LIB}
        ZLIB::ZLIB
        Threads::Threads
        dl
        # common deps
        stdc++fs
        kv_engine
        sparse_tensor
        gflags
        glog
        leveldb
        TBB::tbb
        ${REPORT_LIB}
    )
endif()

# add_executable(grpc_ps_server_async
#     grpc_ps_server_async.cpp 
#     grpc_ps_server_async.h 
#     cache_ps_impl.h
#     ${common}
# )

# target_link_libraries(grpc_ps_server_async PUBLIC
#     base
#     ps_proto
#     gRPC::grpc++_reflection
#     gRPC::grpc++
#     protobuf::libprotobuf
#     stdc++fs
#     kv_engine
# )


add_library(grpc_ps_client_python MODULE
    grpc_ps_client_pytorch.cpp
)

target_include_directories(grpc_ps_client_python PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(grpc_ps_client_python PUBLIC
    grpc_ps_client
    ${TORCH_LIBRARIES}
    Python3::Python
)
