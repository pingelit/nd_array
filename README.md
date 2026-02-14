# nd_array - N-Dimensional Array Class

A high-performance C++ template class for n-dimensional arrays with minimal memory allocations and mdspan-like interface.

## Features

- **Single Memory Allocation**: Memory allocated only once during construction
- **Dynamic Rank**: Runtime-determined number of dimensions (with compile-time maximum)
- **Dynamic Extents**: Runtime-determined size for each dimension
- **mdspan-like Interface**: Familiar API similar to C++23 mdspan
- **Subviews/Subspans**: Efficient non-owning views into array data
- **Row-major Layout**: Contiguous storage with predictable memory layout
- **Zero-overhead Indexing**: Compile-time variadic template indexing

## Basic Usage

### Creating Arrays

```cpp
#include "nd_array.hpp"

// 2D array (3x4)
nd_array<double> matrix(3, 4);

// 3D array (2x3x4)
nd_array<int> tensor(2, 3, 4);

// Using initializer list
nd_array<float> arr({2, 3, 4});

// With custom max rank
nd_array<int, 16> high_rank_array(2, 3, 4, 5);
```

### Accessing Elements

```cpp
// Multi-dimensional indexing
matrix(1, 2) = 42.0;
double val = matrix(1, 2);

// 3D indexing
tensor(0, 1, 2) = 100;
```

### Array Properties

```cpp
size_t r = arr.rank();           // Number of dimensions
size_t total = arr.size();        // Total number of elements
size_t dim0 = arr.extent(0);      // Size of dimension 0
T* ptr = arr.data();              // Raw pointer to data
```

### Operations

```cpp
// Fill with value
arr.fill(0);

// Apply function to all elements
arr.apply([](int x) { return x * 2; });

// Copy (deep copy)
nd_array<int> copy = arr;
```

## Subviews

### Subspan - Extract a Range

```cpp
nd_array<int> arr(4, 5);

// Get rows 1-2 (exclusive end)
auto sub = arr.subspan(0, 1, 3);  // dimension 0, from 1 to 3

// Get columns 2-4
auto cols = arr.subspan(1, 2, 5);  // dimension 1, from 2 to 5
```

### Slice - Reduce Dimensionality

```cpp
nd_array<int> arr3d(2, 3, 4);

// Get a 2D slice (3x4) from the first layer
auto slice = arr3d.slice(0, 0);  // Fix dimension 0 at index 0

// Now slice has rank 2
size_t r = slice.rank();  // 2
```

### Multiple Range Subspan

```cpp
// Extract subregion using multiple ranges
auto sub = arr.subspan({{1, 3}, {2, 5}});  // rows 1-2, cols 2-4
```

## Implementation Details

### Memory Layout

- **Row-major order**: Last index varies fastest
- **Contiguous storage**: All elements in a single allocation
- **Stride-based indexing**: Efficient multi-dimensional access

### Performance Characteristics

- **Construction**: O(n) where n = total elements (one allocation + zero initialization)
- **Element access**: O(rank) computation, typically O(1) for small ranks
- **Subspan creation**: O(1) - no data copying, only metadata
- **Copy**: O(n) - deep copy of all elements

### Template Parameters

```cpp
template<typename T, size_t MaxRank = 8>
class nd_array;
```

- `T`: Element type
- `MaxRank`: Compile-time maximum rank (default: 8)

The actual rank is determined at runtime but cannot exceed `MaxRank`.

### Subspan/View Type

The `nd_span` nested class provides non-owning views:
- Does not allocate memory
- Shares data with parent array
- Can be used for read/write access
- Has the same interface as `nd_array` for element access

## Design Rationale

### Single Allocation Strategy

All memory is allocated in the constructor using `std::unique_ptr<T[]>`. This minimizes:
- Allocation overhead
- Memory fragmentation
- Cache misses due to pointer chasing

### Dynamic Rank with Static Maximum

Using `std::array<size_t, MaxRank>` for extents and strides provides:
- Stack allocation for metadata
- No heap allocation for small arrays
- Compile-time upper bound checking
- Efficient cache utilization

### Variadic Template Indexing

```cpp
template<typename... Indices>
T& operator()(Indices... indices);
```

Benefits:
- Type-safe indexing
- Compile-time arity checking
- No overhead compared to raw pointer arithmetic
- Natural syntax: `arr(i, j, k)`

## Limitations

- Maximum rank is fixed at compile time
- No automatic broadcasting or reshaping
- Slicing creates views, not copies (modify views to modify original)
- No built-in serialization

## Building

```bash
mkdir build
cd build
cmake ..
cmake --build .
./nd_array
```

## Requirements

- C++17 or later
- CMake 3.10 or later

## License

This is example code for educational purposes.
