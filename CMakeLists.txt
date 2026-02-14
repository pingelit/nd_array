cmake_minimum_required(VERSION 3.14)

project(nd_array VERSION 1.0.0 LANGUAGES CXX)

# CPM Package Manager
include(cmake/CPM.cmake)

# Options
option(BUILD_TESTS "Build unit tests" ON)

# Main executable
add_executable(nd_array_demo main.cpp)
target_compile_features(nd_array_demo PRIVATE cxx_std_20)

# Header-only library
add_library(nd_array_lib INTERFACE)
target_include_directories(nd_array_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(nd_array_lib INTERFACE cxx_std_20)

# Build tests
if(BUILD_TESTS)
	enable_testing()
	
	# Fetch Catch2
	CPMAddPackage(
		NAME Catch2
		GITHUB_REPOSITORY catchorg/Catch2
		VERSION 3.5.0
	)
	
	# Test executable
	add_executable(nd_array_tests
		tests/test_nd_array.cpp
		tests/test_nd_span.cpp
	)
	
	target_link_libraries(nd_array_tests PRIVATE nd_array_lib Catch2::Catch2WithMain)
	
	# Enable all warnings and treat warnings as errors
	if(MSVC)
		target_compile_options(nd_array_tests PRIVATE /W4 /WX)
	else()
		target_compile_options(nd_array_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
	endif()
	
	# Discover tests
	include(CTest)
	include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
	catch_discover_tests(nd_array_tests)
endif()
