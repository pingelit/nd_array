cmake_minimum_required(VERSION 3.14)

project(
	nd_array
	VERSION 0.0.1
	LANGUAGES CXX
)

set(ND_ARRAY_NOT_TOP_LEVEL OFF)
if(NOT ${PROJECT_IS_TOP_LEVEL})
	set(ND_ARRAY_NOT_TOP_LEVEL ON)
endif()

# Options
option(ND_ARRAY_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(ND_ARRAY_BUILD_EXAMPLES "Build example programs" ${PROJECT_IS_TOP_LEVEL})
option(ND_ARRAY_USE_SYSTEM_INCLUDE "Use system include for nd_array headers" ${ND_ARRAY_NOT_TOP_LEVEL})

if(ND_ARRAY_USE_SYSTEM_INCLUDE)
	set(ND_ARRAY_SYSTEM_INCLUDE "SYSTEM")
endif()

# Header-only library
add_library(nd_array_lib INTERFACE)
target_include_directories(
	nd_array_lib ${ND_ARRAY_SYSTEM_INCLUDE} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
													  $<INSTALL_INTERFACE:include>
)
target_compile_features(nd_array_lib INTERFACE cxx_std_17)

# Build examples
if(ND_ARRAY_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# Build tests
if(ND_ARRAY_BUILD_TESTS)
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
	add_executable(nd_array_tests tests/test_nd_array.cpp tests/test_nd_span.cpp)

	target_link_libraries(nd_array_tests PRIVATE nd_array_lib Catch2::Catch2WithMain)

	# Link catch2 as a system library to avoid warnings from catch2 headers
	get_target_property(lib_include_dirs Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
	target_include_directories(nd_array_tests SYSTEM PRIVATE ${lib_include_dirs})

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
