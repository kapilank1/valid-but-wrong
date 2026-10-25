# vbw_edge: C++ edge monitor for Valid-But-Wrong. See edge/SPEC.md and edge/README.md.
#
#   Windows:  powershell -ExecutionPolicy Bypass -File edge\build.ps1 -Test
#   Linux:    cmake -S edge -B edge/build -G Ninja -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.31.0
#             cmake --build edge/build && ctest --test-dir edge/build --output-on-failure
cmake_minimum_required(VERSION 3.20)
project(vbw_edge LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Executables land directly in the build dir (edge/build/vbw_edge.exe), next to the ONNX Runtime DLLs.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

get_filename_component(VBW_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# ------------------------------------------------------------------------------------------- ONNX Runtime
if(WIN32)
  set(_ort_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.31.0")
elseif(APPLE)
  set(_ort_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-osx-universal2-1.31.0")
else()
  set(_ort_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.31.0")
endif()
set(ONNXRUNTIME_ROOT "${_ort_default}" CACHE PATH "ONNX Runtime prebuilt package root (contains include/ and lib/)")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. Download the 1.31.0 package into "
                      "edge/third_party (see edge/README.md) or pass -DONNXRUNTIME_ROOT=<dir>.")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
  set(VBW_ORT_RUNTIME_FILES
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
elseif(APPLE)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
  set(VBW_ORT_RUNTIME_FILES "")
else()
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  set(VBW_ORT_RUNTIME_FILES "")
  set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
  set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()

# ------------------------------------------------------------------------------------------- warnings
if(MSVC)
  set(VBW_WARNINGS /W4 /permissive- /EHsc /utf-8 /Zc:__cplusplus /external:W0)
else()
  set(VBW_WARNINGS -Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------------------------------- vbw_core
# vbw_core = every src/*.cpp except main.cpp. CONFIGURE_DEPENDS re-runs the glob at build time, so files other
# agents add (l1.cpp, l2.cpp, alerting.cpp, detector.cpp, ...) are picked up without editing this file.
# The platform layer (config, frame, udp) is its own library inside it, so test_frame / test_config build and run
# even while the detector sources are mid-change; anything linking vbw_core gets both.
set(VBW_PLATFORM_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/config.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/frame.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/udp.cpp")
file(GLOB VBW_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB VBW_CORE_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
list(REMOVE_ITEM VBW_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp" ${VBW_PLATFORM_SOURCES})

add_library(vbw_platform STATIC ${VBW_PLATFORM_SOURCES})
target_include_directories(vbw_platform PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(vbw_platform PRIVATE ${VBW_WARNINGS})
if(WIN32)
  target_link_libraries(vbw_platform PUBLIC ws2_32)
  target_compile_definitions(vbw_platform PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(VBW_CORE_SOURCES)
  add_library(vbw_core STATIC ${VBW_CORE_SOURCES} ${VBW_CORE_HEADERS})
  target_compile_options(vbw_core PRIVATE ${VBW_WARNINGS})
  set(_vbw_core_scope PUBLIC)
else()
  add_library(vbw_core INTERFACE)
  set(_vbw_core_scope INTERFACE)
endif()
target_link_libraries(vbw_core ${_vbw_core_scope} vbw_platform onnxruntime)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mavlink")
  target_include_directories(vbw_core SYSTEM ${_vbw_core_scope} "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mavlink")
endif()

# Copies the ONNX Runtime shared libraries next to an executable (Windows needs them beside the .exe).
function(vbw_copy_runtime target)
  foreach(f IN LISTS VBW_ORT_RUNTIME_FILES)
    if(EXISTS "${f}")
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${f}" "$<TARGET_FILE_DIR:${target}>"
        VERBATIM)
    endif()
  endforeach()
endfunction()

# ------------------------------------------------------------------------------------------- vbw_edge
# main.cpp needs the detector (src/detector.hpp); while it is missing only the library and tests are built.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/detector.hpp" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
  add_executable(vbw_edge "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
  target_compile_options(vbw_edge PRIVATE ${VBW_WARNINGS})
  target_link_libraries(vbw_edge PRIVATE vbw_core)
  vbw_copy_runtime(vbw_edge)
else()
  message(WARNING "src/detector.hpp or src/main.cpp not found: skipping the vbw_edge executable")
endif()

# ------------------------------------------------------------------------------------------- tests
# Tests are plain executables that return non-zero on failure. They run from the repo root so they can open
# edge/config/... and edge/testdata/...; VBW_REPO_ROOT is also compiled in and exported as a fallback.
enable_testing()
# Not used directly: makes the build re-run configure when a test source appears (the EXISTS checks below).
file(GLOB VBW_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

function(vbw_add_exe name lib)
  add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/tests/${name}.cpp")
  target_compile_options(${name} PRIVATE ${VBW_WARNINGS})
  target_compile_definitions(${name} PRIVATE VBW_REPO_ROOT="${VBW_REPO_ROOT}")
  target_link_libraries(${name} PRIVATE ${lib})
  vbw_copy_runtime(${name})
endfunction()

function(vbw_add_test name lib)
  vbw_add_exe(${name} ${lib})
  add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY "${VBW_REPO_ROOT}")
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "VBW_REPO_ROOT=${VBW_REPO_ROOT}" TIMEOUT 600)
endfunction()

vbw_add_test(test_frame vbw_platform)
vbw_add_test(test_config vbw_platform)
foreach(t IN ITEMS test_l1 test_detector)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cpp")
    vbw_add_test(${t} vbw_core)
  endif()
endforeach()
# The benchmark is built but not registered with CTest (run it by hand: edge\build\bench_detector.exe).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_detector.cpp")
  vbw_add_exe(bench_detector vbw_core)
endif()
