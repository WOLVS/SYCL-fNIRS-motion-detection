cmake_minimum_required(VERSION 3.10)

project(DecisionTreeExample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(DecisionTreeExample
    main.cpp
    DecisionNode.cpp
    Data.h
    DecisionNode.h
)
