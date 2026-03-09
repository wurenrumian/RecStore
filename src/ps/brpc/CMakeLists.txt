set(BRPC_ROOT "${PROJECT_SOURCE_DIR}/third_party/brpc-install")
set(PROTOBUF_ROOT "${PROJECT_SOURCE_DIR}/third_party/protobuf-install")

# Find bRPC (禁止去系统找)
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

# Find Protobuf (禁止去系统找)
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

# Find system libraries (OpenSSL, ZLIB)
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
find_package(Threads REQUIRED)

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
    
    # bRPC PS Client Library
    add_library(brpc_ps_client SHARED
        brpc_ps_client.cpp brpc_ps_client.h
    )
    target_include_directories(brpc_ps_client PUBLIC ${BRPC_INCLUDE_PATH})
    
    target_link_libraries(brpc_ps_client PUBLIC
        base
        ps_base
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
    )

    # bRPC PS Client Test
    add_executable(brpc_ps_client_test
        brpc_ps_client_test.cpp
    )
    target_link_libraries(brpc_ps_client_test PUBLIC
        brpc_ps_client
    )

    # bRPC PS Server
    add_executable(brpc_ps_server
        brpc_ps_server.cpp
        brpc_ps_server.h
    )
    target_include_directories(brpc_ps_server PUBLIC ${BRPC_INCLUDE_PATH})
    target_link_libraries(brpc_ps_server PUBLIC
        base
        ps_base
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
        sparse_tensor
        gflags
        leveldb
        TBB::tbb
    )

    # Distributed bRPC PS Client
    add_library(dist_brpc_ps_client
        dist_brpc_ps_client.cpp dist_brpc_ps_client.h
    )
    target_include_directories(dist_brpc_ps_client PUBLIC ${BRPC_INCLUDE_PATH})
    target_link_libraries(dist_brpc_ps_client PUBLIC
        base
        ps_base
        brpc_ps_client
        ps_proto
        ${BRPC_LIB}
        ${CUSTOM_PROTOBUF_LIB}
        ${SYSTEM_OPENSSL_SSL_LIB}
        ${SYSTEM_OPENSSL_CRYPTO_LIB}
        ZLIB::ZLIB
        leveldb
        dl

    )

    # Distributed bRPC PS Client Test
    add_executable(dist_brpc_ps_client_test
        dist_brpc_ps_client_test.cpp
    )
    target_link_libraries(dist_brpc_ps_client_test PUBLIC
        dist_brpc_ps_client
    )
    
    # Export brpc_FOUND to parent scope for unified ps_server
    set(brpc_FOUND TRUE PARENT_SCOPE)
    set(BRPC_LIB ${BRPC_LIB} PARENT_SCOPE)
    set(CUSTOM_PROTOBUF_LIB ${CUSTOM_PROTOBUF_LIB} PARENT_SCOPE)
    set(SYSTEM_OPENSSL_SSL_LIB ${SYSTEM_OPENSSL_SSL_LIB} PARENT_SCOPE)
    set(SYSTEM_OPENSSL_CRYPTO_LIB ${SYSTEM_OPENSSL_CRYPTO_LIB} PARENT_SCOPE)
    set(BRPC_INCLUDE_PATH ${BRPC_INCLUDE_PATH} PARENT_SCOPE)
endif()
