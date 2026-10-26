cmake_minimum_required(VERSION 3.18)
project(pnp_cxx_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)  # default; demo glue at -O0 otherwise
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

# Link the Rust core's cdylib (self-contained Rust deps; MOSEK dynamic).
# Build it first, from the repo root:
#   cargo build -p pnp-cxx --release
set(RUST_DYLIB "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release/${CMAKE_SHARED_LIBRARY_PREFIX}pnp_cxx${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(PNP_CXX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../pnp-cxx/include")

foreach(demo demo_pinhole demo_spherical)
    add_executable(${demo} ${demo}.cpp)
    target_include_directories(${demo} PRIVATE ${PNP_CXX_INCLUDE})
    target_link_libraries(${demo} PRIVATE Eigen3::Eigen "${RUST_DYLIB}")
endforeach()
