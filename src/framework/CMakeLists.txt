option(USE_FAKE_KVCLIENT "Use FakeKVClientOp for testing" OFF)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)

if(USE_FAKE_KVCLIENT)
    message(STATUS "Using FakeKVClientOp for testing!")
    add_definitions(-DUSE_FAKE_KVCLIENT)
endif()

add_test(
    NAME pytorch_client_test
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/python_client/client_test.py
            "$<TARGET_FILE:recstore_torch_ops>"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/python_client
)

add_test(
    NAME dist_emb_unittest
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb
)

set_tests_properties(dist_emb_unittest PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)

add_test(
    NAME ebc_precision_test
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_ebc_precision_wrapper
)

set_tests_properties(ebc_precision_test PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)

add_test(
    NAME ebc_multiprocess_precision_test
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_ebc_precision_wrapper.TestEBCPrecision.test_multiprocess_precision
)

set_tests_properties(ebc_multiprocess_precision_test PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)
