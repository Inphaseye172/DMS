# After Effects plugin (.aex). Windows + MSVC only; requires the AE SDK (AE_SDK_ROOT).
include(AESDK)

ae_compile_pipl(DMS_PIPL_RC "${CMAKE_CURRENT_SOURCE_DIR}/ae/DMS_Depth_PiPL.r")

add_library(DMS_Depth MODULE
  ae/DMS_Depth.cpp
  ae/DMS_Depth_Params.cpp
  ae/DMS_Depth_Render.cpp
  ae/DMS_Depth_Dialog.cpp
  ae/DMS_Depth_extra.rc
  "${DMS_PIPL_RC}"
  ${AE_SDK_UTIL_SOURCES})
target_include_directories(DMS_Depth PRIVATE ae)
target_include_directories(DMS_Depth SYSTEM PRIVATE ${AE_SDK_INCLUDE_DIRS})
target_link_libraries(DMS_Depth PRIVATE dms_plugin_core dms_engine comctl32 shell32 user32)
target_compile_definitions(DMS_Depth PRIVATE MSWindows WIN32 _WINDOWS)
set_target_properties(DMS_Depth PROPERTIES
  SUFFIX ".aex"
  PREFIX ""
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build/debug"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build/debug")
# We embed our own comctl32 v6 manifest (resource id 2) for the dialog activation context.
target_link_options(DMS_Depth PRIVATE /MANIFEST:NO)
dms_apply_warnings(DMS_Depth)
dms_apply_analyze(DMS_Depth)
# The AE SDK util .c file is not ours: compile it without /WX.
set_source_files_properties(${AE_SDK_UTIL_SOURCES} PROPERTIES COMPILE_OPTIONS "/W0;/WX-")
# Never instrument the .aex with ASan (host process we do not control); windows-debug therefore
# builds the engine/CLI/tests with ASan and skips the plugin (see CMakePresets.json).
if(DMS_ENABLE_ASAN)
  message(FATAL_ERROR "DMS_ENABLE_ASAN=ON is not supported together with DMS_BUILD_PLUGIN=ON")
endif()
