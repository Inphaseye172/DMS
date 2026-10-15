# Host-independent plugin core: built on every platform and unit-tested on Linux.
add_library(dms_plugin_core STATIC params_model.cpp sequence_data.cpp pixels.cpp render_core.cpp)
target_include_directories(dms_plugin_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(dms_plugin_core PUBLIC dms_engine)
dms_apply_warnings(dms_plugin_core)
dms_apply_analyze(dms_plugin_core)
