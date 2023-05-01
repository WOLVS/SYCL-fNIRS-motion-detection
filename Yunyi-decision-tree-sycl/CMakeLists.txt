cmake_minimum_required(VERSION 3.20)

set(ProjectName DecisionTree)

project(${ProjectName} CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ComputeCpp REQUIRED)

add_compile_options(-O3 -Wall)

enable_testing()

set(Headers
    Data.h
    DecisionNode.h
)

set(Source
    DecisionNode.cpp
    main.cpp
)


add_executable(${ProjectName} ${Source} ${Headers})
target_include_directories(${ProjectName} PUBLIC ${ComputeCpp_INCLUDE_DIRS})
target_link_libraries(${ProjectName} PUBLIC ComputeCpp::ComputeCpp)

# Apply ComputeCpp device compiler flags only to the device code
set_source_files_properties(DecisionNode.cpp PROPERTIES COMPUTECPP_SOURCE_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")
set_source_files_properties(main.cpp PROPERTIES COMPUTECPP_SOURCE_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

# Enable automatic generation of the integration header
add_sycl_to_target(TARGET ${ProjectName} SOURCES DecisionNode.cpp main.cpp)
