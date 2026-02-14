cmake_minimum_required(VERSION 3.14)

project(nd_array VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# Header-only library
add_library(nd_array_lib INTERFACE)
target_include_directories(nd_array_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(nd_array_lib INTERFACE cxx_std_17)

# Build examples
if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# Build tests
if(BUILD_TESTS)
	# CPM Package Manager
	include(cmake/CPM.cmake)
	
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
