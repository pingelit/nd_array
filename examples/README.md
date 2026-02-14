# Examples

This directory contains example programs demonstrating the usage of `nd_array` and `nd_span`.

## Building Examples

Examples are built by default. To disable:

```bash
cmake -B build -DBUILD_EXAMPLES=OFF
```

## Running Examples

### demo.cpp

Comprehensive demonstration of all features:

```bash
# Windows
.\build\windows-debug\examples\nd_array_demo.exe

# Linux/macOS
./build/linux-debug/examples/nd_array_demo
```

This example demonstrates:
- Creating nd_span from C-arrays
- Wrapping std::vector with nd_span
- Creating nd_array from containers
- 2D and 3D array operations
- Subspan and slice operations
- Fill and apply operations
- Copy semantics

## Creating Your Own Examples

To add a new example:

1. Create a new `.cpp` file in this directory
2. Add it to `examples/CMakeLists.txt`:
   ```cmake
   add_executable(my_example my_example.cpp)
   target_link_libraries(my_example PRIVATE nd_array_lib)
   ```
3. Include the header:
   ```cpp
   #include "../nd_array.hpp"
   ```

## Example Template

```cpp
#include "../nd_array.hpp"
#include <iostream>

int main() {
    // Create a 2D array
    nd_array<double> matrix(3, 4);
    
    // Fill with values
    for(size_t i = 0; i < 3; ++i) {
        for(size_t j = 0; j < 4; ++j) {
            matrix(i, j) = i * 4 + j;
        }
    }
    
    // Use it
    std::cout << "Matrix[1,2] = " << matrix(1, 2) << "\n";
    
    return 0;
}
```
