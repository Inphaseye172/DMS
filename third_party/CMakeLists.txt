# Vendored header-only libraries (hashes in THIRD_PARTY.md).
add_library(dms_third_party INTERFACE)
target_include_directories(dms_third_party SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# stb implementation is compiled once here so that -Werror warnings in stb never affect DMS targets.
add_library(dms_stb STATIC stb_impl.c)
target_include_directories(dms_stb SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(MSVC)
  target_compile_definitions(dms_stb PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(dms_stb PRIVATE /w)
else()
  target_compile_options(dms_stb PRIVATE -w)
endif()
