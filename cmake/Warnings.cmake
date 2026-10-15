# Same folder for every configuration (multi-config generators would add bin/Release), so the
# executable always sits next to bin/DMS_deps where the engine looks for ONNX Runtime by default.
function(dms_set_bin_dir target)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  foreach(_cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${_cfg} _CFG)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
  endforeach()
endfunction()

# Warnings-as-errors for all DMS targets. Applied via dms_apply_warnings(<target>).
function(dms_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /EHsc
      /wd4324   # structure was padded due to alignment specifier (intentional in cache structs)
    )
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion
      -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual
      -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${target} PRIVATE -Wduplicated-cond -Wlogical-op)
    else()
      # GCC 13 reports false positives for -Wnull-dereference inside libstdc++ at -O3; Clang does not.
      target_compile_options(${target} PRIVATE -Wnull-dereference)
    endif()
  endif()
  if(DMS_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  elseif(DMS_ENABLE_ASAN AND MSVC)
    target_compile_options(${target} PRIVATE /fsanitize=address)
  endif()
endfunction()
