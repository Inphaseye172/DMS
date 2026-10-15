add_executable(depthcli main.cpp image_io.cpp)
target_link_libraries(depthcli PRIVATE dms_engine dms_stb)
dms_apply_warnings(depthcli)
dms_apply_analyze(depthcli)
# Put the CLI next to the tests' working directory layout.
set_target_properties(depthcli PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
