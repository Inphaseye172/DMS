# Locates the After Effects SDK. Accepts AE_SDK_ROOT pointing at the SDK root (containing Examples/)
# or at the Examples folder itself. Fails with a clear message when missing.
set(AE_SDK_ROOT "$ENV{AE_SDK_ROOT}" CACHE PATH "Path to the After Effects SDK (folder containing Examples/ or the Examples folder)")
if(AE_SDK_ROOT STREQUAL "")
  message(FATAL_ERROR
    "AE_SDK_ROOT is not set.\n"
    "Download the After Effects SDK from https://developer.adobe.com/after-effects/ and either\n"
    "  set the AE_SDK_ROOT environment variable, or\n"
    "  pass -DAE_SDK_ROOT=C:/path/to/AfterEffectsSDK to CMake.\n"
    "To build only the engine, CLI and tests, configure with -DDMS_BUILD_PLUGIN=OFF (preset windows-engine-only).")
endif()
file(TO_CMAKE_PATH "${AE_SDK_ROOT}" _ae_root)
set(AE_SDK_EXAMPLES "")
foreach(_cand "${_ae_root}/Examples" "${_ae_root}")
  if(EXISTS "${_cand}/Headers/AE_Effect.h")
    set(AE_SDK_EXAMPLES "${_cand}")
    break()
  endif()
endforeach()
if(AE_SDK_EXAMPLES STREQUAL "")
  message(FATAL_ERROR "AE_SDK_ROOT=${AE_SDK_ROOT} does not contain Headers/AE_Effect.h (looked in <root>/Examples and <root>).")
endif()
find_program(AE_PIPL_TOOL NAMES PiPLtool PiPLTool PATHS "${AE_SDK_EXAMPLES}/Resources" "${_ae_root}/Resources" NO_DEFAULT_PATH)
if(NOT AE_PIPL_TOOL)
  message(FATAL_ERROR "PiPLtool.exe not found under ${AE_SDK_EXAMPLES}/Resources")
endif()
set(AE_SDK_INCLUDE_DIRS
  "${AE_SDK_EXAMPLES}/Headers"
  "${AE_SDK_EXAMPLES}/Headers/SP"
  "${AE_SDK_EXAMPLES}/Headers/adobesdk"
  "${AE_SDK_EXAMPLES}/Headers/adobesdk/config"
  "${AE_SDK_EXAMPLES}/Headers/adobesdk/drawbotsuite"
  "${AE_SDK_EXAMPLES}/Util"
  "${AE_SDK_EXAMPLES}/Resources")
set(AE_SDK_UTIL_SOURCES "${AE_SDK_EXAMPLES}/Util/AEFX_SuiteHelper.c")
message(STATUS "After Effects SDK: ${AE_SDK_EXAMPLES}")
message(STATUS "PiPLtool: ${AE_PIPL_TOOL}")

# Compiles a PiPL .r file into a Windows .rc the same way Adobe's sample projects do:
#   cl /EP x.r > x.rr ; PiPLtool x.rr x.rrc ; cl /D MSWindows /EP x.rrc > x.rc
function(ae_compile_pipl out_rc_var pipl_r)
  get_filename_component(_name "${pipl_r}" NAME_WE)
  set(_rr  "${CMAKE_CURRENT_BINARY_DIR}/${_name}.rr")
  set(_rrc "${CMAKE_CURRENT_BINARY_DIR}/${_name}.rrc")
  set(_rc  "${CMAKE_CURRENT_BINARY_DIR}/${_name}.rc")
  set(_incs "")
  foreach(_d ${AE_SDK_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/ae")
    list(APPEND _incs "/I${_d}")
  endforeach()
  add_custom_command(
    OUTPUT "${_rc}"
    COMMAND "${CMAKE_CXX_COMPILER}" /nologo ${_incs} /EP "${pipl_r}" > "${_rr}"
    COMMAND "${AE_PIPL_TOOL}" "${_rr}" "${_rrc}"
    COMMAND "${CMAKE_CXX_COMPILER}" /nologo /D "MSWindows" /EP "${_rrc}" > "${_rc}"
    DEPENDS "${pipl_r}" "${CMAKE_CURRENT_SOURCE_DIR}/ae/dms_version.h"
    COMMENT "Compiling PiPL ${_name}"
    VERBATIM)
  set(${out_rc_var} "${_rc}" PARENT_SCOPE)
endfunction()
