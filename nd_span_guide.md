# nd_span - Standalone Non-Owning View

## Overview

The `nd_span` class has been extracted from `nd_array` and is now a standalone template class that provides a non-owning view over multidimensional data. This allows wrapping raw C-arrays or other contiguous memory buffers with a modern C++ interface.

## Key Features

- **Non-owning**: Does not manage memory, only provides a view
- **Independent**: Can be used without `nd_array`
- **C-interop**: Perfect for wrapping C-arrays from C APIs
- **Multiple constructors**: Flexible initialization options
- **Subviews**: Can create subspans and slices

## Usage Examples

### Wrapping a C-array

```cpp
// C API returns raw pointer
double* c_array = get_data_from_c_api();

// Wrap as 3x4 matrix
nd_span<double> span(c_array, 3, 4);

// Access with modern interface
span(1, 2) = 42.0;
double val = span(1, 2);
```

### Wrapping std::vector

```cpp
std::vector<int> data = {10, 20, 30, 40, 50, 60};
nd_span<int> span(data.data(), 2, 3);

// Use as 2x3 matrix
for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
        std::cout << span(i, j) << " ";
    }
}
```

### Using with nd_array

```cpp
nd_array<int> arr(4, 5);
// ... fill array ...

// Get a subview
auto sub = arr.subspan(0, 1, 3);  // rows 1-2
sub(0, 2) = 99;  // modifies original array
```

## Constructors

### Basic Constructor (variadic)

```cpp
nd_span(T* data, size_t extent1, size_t extent2, ...);
```

Create a span from a pointer and extents:
```cpp
int* data = new int[12];
nd_span<int> span(data, 3, 4);
```

### Initializer List Constructor

```cpp
nd_span(T* data, std::initializer_list<size_t> extents);
```

Create with runtime-determined rank:
```cpp
int* data = new int[24];
nd_span<int> span(data, {2, 3, 4});
```

### Container Constructor

```cpp
template<typename Container>
nd_span(T* data, const Container& extents);
```

Create from any container of extents:
```cpp
std::vector<size_t> extents = {2, 3, 4};
nd_span<int> span(data, extents);
```

### Internal Constructor

```cpp
nd_span(T* data, const std::array<size_t, MaxRank>& extents,
        const std::array<size_t, MaxRank>& strides, size_t rank);
```

Used internally for subspans and slices with custom strides.

## Operations

### Element Access

```cpp
template<typename... Indices>
T& operator()(Indices... indices);
```

Access elements with multidimensional indexing:
```cpp
span(i, j, k) = value;
```

### Subspan

```cpp
nd_span subspan(size_t dim, size_t start, size_t end);
```

Extract a range along one dimension:
```cpp
auto rows = span.subspan(0, 1, 3);  // rows 1-2
```

### Slice

```cpp
nd_span slice(size_t dim, size_t index);
```

Fix one dimension, reducing rank:
```cpp
auto slice = span3d.slice(0, 1);  // get layer 1 (now 2D)
```

### Queries

```cpp
size_t extent(size_t dim) const;  // Size of dimension
size_t rank() const;               // Number of dimensions
T* data();                         // Raw pointer to data
```

## Memory Layout

- **Row-major order**: Last index varies fastest
- **Stride-based**: Supports non-contiguous views (from subspans)
- **No allocation**: Zero heap allocations (except for the span object itself)

## C Interop Pattern

Common pattern for C API integration:

```cpp
// C API
extern "C" {
    double* get_matrix(size_t* rows, size_t* cols);
    void process_matrix(double* data, size_t rows, size_t cols);
}

// C++ wrapper
void modern_process() {
    size_t rows, cols;
    double* c_matrix = get_matrix(&rows, &cols);
    
    // Wrap in nd_span
    nd_span<double> matrix(c_matrix, rows, cols);
    
    // Use modern C++ interface
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) *= 2.0;
        }
    }
    
    // Pass back to C API (same memory)
    process_matrix(c_matrix, rows, cols);
}
```

## Safety Considerations

1. **Lifetime**: The span does not own the data. Ensure the underlying memory remains valid.
2. **Bounds checking**: All access is bounds-checked (throws `std::out_of_range`)
3. **Size matching**: Total elements must match the product of extents

## Comparison with nd_array

| Feature | nd_array | nd_span |
|---------|----------|---------|
| Memory ownership | Owns (unique_ptr) | Non-owning (raw pointer) |
| Allocation | Yes (construction) | No |
| Copy | Deep copy | Shallow (copies view) |
| Use case | Data storage | View/wrapper |
| C-interop | No | Yes |

## Implementation Details

The `nd_span` class:
- Uses the same `detail::offset_computer` helper as `nd_array`
- Computes strides in constructor (row-major layout)
- Stores extents and strides in `std::array<size_t, MaxRank>`
- All metadata on stack (no heap allocations)

## Example: Complete C API Integration

```cpp
#include "nd_array.hpp"

// External C library
extern "C" {
    double* allocate_image(size_t width, size_t height);
    void free_image(double* img);
}

void process_image() {
    // Get C-allocated memory
    double* img = allocate_image(640, 480);
    
    // Wrap as nd_span
    nd_span<double> image(img, 480, 640);
    
    // Process using modern C++ idioms
    for (size_t y = 0; y < image.extent(0); ++y) {
        for (size_t x = 0; x < image.extent(1); ++x) {
            image(y, x) = std::clamp(image(y, x), 0.0, 1.0);
        }
    }
    
    // Get a region of interest
    auto roi = image.subspan(0, 100, 200)  // rows 100-199
                    .subspan(1, 50, 150);   // cols 50-149
    
    // Free using C API
    free_image(img);
}
```
