# Warnings-as-errors for all DMS targets. Applied via dms_apply_warnings(<target>).
function(dms_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /EHsc
      /wd4324   # structure was padded due to alignment specifier (intentional in cache structs)
    )
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion
      -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wnull-dereference
      -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${target} PRIVATE -Wduplicated-cond -Wlogical-op)
    endif()
  endif()
  if(DMS_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  elseif(DMS_ENABLE_ASAN AND MSVC)
    target_compile_options(${target} PRIVATE /fsanitize=address)
  endif()
endfunction()
