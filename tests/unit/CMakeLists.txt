include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        b514bdc898e2951020cbdca1304b75f5950d1f59 # v1.15.2
  GIT_SHALLOW    FALSE
  SYSTEM)
FetchContent_MakeAvailable(googletest)

file(GLOB DMS_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
add_executable(dms_tests ${DMS_TEST_SOURCES})
target_link_libraries(dms_tests PRIVATE dms_engine_internal dms_plugin_core GTest::gtest_main dms_stb)
target_compile_definitions(dms_tests PRIVATE
  DMS_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
  DMS_ORT_RUNTIME_DIR="${DMS_ORT_RUNTIME_DIR}"
  DMS_TEST_TMP_DIR="${CMAKE_BINARY_DIR}/test_tmp")
dms_apply_warnings(dms_tests)
if(MSVC)
  target_compile_options(dms_tests PRIVATE /wd4996)
endif()
set_target_properties(dms_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(GoogleTest)
gtest_discover_tests(dms_tests
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  PROPERTIES ENVIRONMENT "DMS_LOG_DIR=${CMAKE_BINARY_DIR}/test_tmp/logs"
  DISCOVERY_TIMEOUT 60)

if(DMS_BUILD_CLI)
  add_test(NAME cli_help COMMAND depthcli --help)
endif()
