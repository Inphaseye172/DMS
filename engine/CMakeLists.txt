add_library(dms_engine STATIC
  src/engine.cpp
  src/engine_compute.cpp
  src/logging.cpp
  src/platform.cpp
  src/sha256.cpp
  src/image_ops.cpp
  src/engine_stubs.cpp
  src/manifest.cpp
  src/transport.cpp
  src/downloader.cpp
  src/ort_runtime.cpp
  src/inference_service.cpp
  src/pipeline.cpp
  src/gpu_probe.cpp
  src/backend_plan.cpp
  src/frame_cache.cpp
  src/instance_registry.cpp
  src/postprocess.cpp
  src/temporal.cpp
)

# Built-in model manifest compiled into the engine.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/models_manifest.json" DMS_MANIFEST_JSON)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/models_manifest.inc.in" "${CMAKE_CURRENT_BINARY_DIR}/generated/models_manifest.inc" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/models_manifest.json")
target_include_directories(dms_engine PUBLIC include PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(dms_engine SYSTEM PRIVATE "${DMS_ORT_INCLUDE_DIR}")
target_link_libraries(dms_engine PRIVATE dms_third_party)
target_compile_definitions(dms_engine PRIVATE DMS_ENGINE_VERSION_STRING="${PROJECT_VERSION}" ORT_API_MANUAL_INIT)
if(DMS_ENABLE_TENSORRT)
  target_compile_definitions(dms_engine PRIVATE DMS_ENABLE_TENSORRT=1)
endif()
if(WIN32)
  target_link_libraries(dms_engine PUBLIC winhttp shell32 ole32 psapi dxguid)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(dms_engine PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
dms_apply_warnings(dms_engine)
dms_apply_analyze(dms_engine)

# Developer/test convenience: place the ONNX Runtime shared library where the engine looks for it by
# default (<exe dir>/DMS_deps/ort-cpu or ort-cuda). The plugin installer does the same for the .aex.
if(NOT WIN32)
  file(GLOB _ort_libs "${DMS_ORT_RUNTIME_DIR}/libonnxruntime.so*")
  file(COPY ${_ort_libs} DESTINATION "${CMAKE_BINARY_DIR}/bin/DMS_deps/ort-cpu" FOLLOW_SYMLINK_CHAIN)
else()
  file(GLOB _ort_libs "${DMS_ORT_RUNTIME_DIR}/*.dll")
  file(COPY ${_ort_libs} DESTINATION "${CMAKE_BINARY_DIR}/bin/DMS_deps/ort-cuda")
endif()

# Internal headers are exposed to unit tests only.
add_library(dms_engine_internal INTERFACE)
target_include_directories(dms_engine_internal INTERFACE src "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(dms_engine_internal SYSTEM INTERFACE "${DMS_ORT_INCLUDE_DIR}")
target_link_libraries(dms_engine_internal INTERFACE dms_engine dms_third_party)
target_compile_definitions(dms_engine_internal INTERFACE ORT_API_MANUAL_INIT)
