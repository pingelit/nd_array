# Unit Testing Setup

## Directory Structure

Create the following directories:
```
nd_array/
├── cmake/
│   └── CPM.cmake
└── tests/
    ├── test_nd_array.cpp
    └── test_nd_span.cpp
```

## Files to Create

### 1. cmake/CPM.cmake

```cmake
set(CPM_DOWNLOAD_VERSION 0.38.7)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})
```

### 2. tests/test_nd_array.cpp

See the comprehensive test file content below covering:
- Construction (default, variadic, initializer list, container)
- Element access (1D, 2D, 3D, out of bounds)
- Copy semantics (copy constructor, copy assignment, deep copy verification)
- Move semantics (move constructor, move assignment)
- Operations (fill, apply)
- Subspan (various dimensions, modifications, error handling)
- Slice (dimension reduction, modifications, error handling)
- Properties (rank, size, extents, data pointer)

### 3. tests/test_nd_span.cpp

See the comprehensive test file content below covering:
- Construction (variadic, initializer list, container, wrapping vectors)
- Element access (1D, 2D, 3D, modifications through span)
- Const access
- Subspan (various dimensions, modifications, error handling)
- Slice (dimension reduction, modifications, error handling)
- Properties (rank, extents, data pointer)
- C-array interop (wrapping C-arrays, std::array, std::vector)
- Integration with nd_array

## Building and Running Tests

### Configure
```bash
cmake --preset windows-debug
```

### Build
```bash
cmake --build --preset windows-debug
```

### Run Tests
```bash
cd build/windows-debug
ctest --output-on-failure
```

Or run the test executable directly:
```bash
./build/windows-debug/nd_array_tests
```

## Test Features

- **Catch2 v3**: Modern C++ testing framework
- **CPM**: Automatic dependency management
- **Warnings as Errors**: All tests compiled with `/W4 /WX` (MSVC) or `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang)
- **Comprehensive Coverage**: Over 60 test cases covering all functionality
- **CTest Integration**: Tests discoverable by CTest

## Test Categories

Tests are organized with tags:
- `[nd_array]` - nd_array tests
- `[nd_span]` - nd_span tests
- `[construction]` - Construction tests
- `[access]` - Element access tests
- `[copy]` - Copy semantics
- `[move]` - Move semantics
- `[operations]` - Operations like fill, apply
- `[subspan]` - Subspan tests
- `[slice]` - Slice tests
- `[properties]` - Property query tests
- `[const]` - Const correctness tests
- `[c-interop]` - C API interoperability
- `[integration]` - Integration tests

Run specific tests:
```bash
./nd_array_tests [nd_array]
./nd_array_tests [nd_span]
./nd_array_tests "[subspan]"
```
