add_library(dms_engine STATIC
  src/engine.cpp
  src/engine_compute.cpp
  src/logging.cpp
  src/platform.cpp
  src/sha256.cpp
  src/image_ops.cpp
  src/engine_stubs.cpp
)
target_include_directories(dms_engine PUBLIC include PRIVATE src)
target_include_directories(dms_engine SYSTEM PRIVATE "${DMS_ORT_INCLUDE_DIR}")
target_link_libraries(dms_engine PRIVATE dms_third_party)
target_compile_definitions(dms_engine PRIVATE DMS_ENGINE_VERSION_STRING="${PROJECT_VERSION}" ORT_API_MANUAL_INIT)
if(DMS_ENABLE_TENSORRT)
  target_compile_definitions(dms_engine PRIVATE DMS_ENABLE_TENSORRT=1)
endif()
if(WIN32)
  target_link_libraries(dms_engine PUBLIC winhttp shell32 ole32 psapi)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(dms_engine PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
dms_apply_warnings(dms_engine)
dms_apply_analyze(dms_engine)

# Internal headers are exposed to unit tests only.
add_library(dms_engine_internal INTERFACE)
target_include_directories(dms_engine_internal INTERFACE src)
target_include_directories(dms_engine_internal SYSTEM INTERFACE "${DMS_ORT_INCLUDE_DIR}")
target_link_libraries(dms_engine_internal INTERFACE dms_engine dms_third_party)
