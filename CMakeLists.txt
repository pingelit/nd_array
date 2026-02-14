cmake_minimum_required(VERSION 3.10)

project(nd_array)


add_executable(nd_array main.cpp)

target_compile_features(nd_array PRIVATE cxx_std_20)
