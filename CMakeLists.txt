cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(ising_config_generator NONE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/tool/standards/config)
include(core)

# Eigen3
find_package(Eigen3 REQUIRED)
include(${EIGEN3_USE_FILE})

include_directories(${PROJECT_SOURCE_DIR}/tool/standards)
include_directories(${PROJECT_SOURCE_DIR}/tool/lattice)
include_directories(${PROJECT_SOURCE_DIR}/tool/cluster-mc)
add_definitions(-DALPS_INDEP_SOURCE)

set(PROGS ising-mrg-201911)
foreach(name ${PROGS})
  add_executable(${name} ${name}.cpp)
endforeach(name)
