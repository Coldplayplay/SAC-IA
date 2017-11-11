cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcl-sac)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable (main main.cpp)
target_link_libraries (main ${PCL_LIBRARIES} )
