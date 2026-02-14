# `nd_array` - Owning N-Dimensional Array

## Overview

The `nd_array` class is an owning, heap-allocated N-dimensional array with a
compact metadata layout and an mdspan-like interface. It stores elements in a
single contiguous allocation and provides efficient subviews via `nd_span`.

## Key Features

- **Owning**: Manages memory with a single allocation
- **Dynamic rank**: Runtime-determined number of dimensions
- **Dynamic extents**: Runtime-determined size for each dimension
- **mdspan-like API**: Familiar indexing and access patterns
- **Subviews**: Efficient `subspan` and `slice` views
- **Row-major layout**: Predictable contiguous storage

## Usage Examples

### Creating Arrays

```cpp
nd_array<double> matrix(3, 4);
nd_array<int> tensor(2, 3, 4);
nd_array<float> dynamic({2, 3, 4});
```

### Element Access

```cpp
matrix(1, 2) = 42.0;
double val = matrix(1, 2);
```

### Flat Iteration

```cpp
for (auto& value : matrix) {
	value += 1.0;
}
```

### Filling and Applying

```cpp
matrix.fill(0.0);
matrix.apply([](double x) { return x + 1.0; });
```

### Shape Operations

```cpp
auto reshaped = matrix.reshape(2, 6);
auto flat = matrix.flatten();
auto squeezed = matrix.squeeze();
auto transposed = matrix.T();
auto permuted = matrix.transpose({1, 0});
```

## Constructors

### Variadic Extents

```cpp
nd_array<double> arr(3, 4, 5);
```

### Initializer List

```cpp
nd_array<int> arr({2, 3, 4});
```

### Container of Extents

```cpp
std::vector<size_t> extents = {2, 3, 4};
nd_array<int> arr(extents);
```

## Operations

### Element Access

```cpp
template<typename... Indices>
T& operator()(Indices... indices);
```

### Fill

```cpp
void fill(const T& value);
```

### Apply

```cpp
template<typename Func>
void apply(Func func);
```

### Reshape

```cpp
nd_span<T> reshape(std::initializer_list<size_t> new_extents);
nd_span<T> reshape(size_t e0, size_t e1, ...);
```

### Transpose

```cpp
nd_span<T> transpose(std::initializer_list<size_t> axes);
nd_span<T> T();
```

### Flatten

```cpp
nd_span<T> flatten();
```

### Squeeze

```cpp
nd_span<T> squeeze();
```

## Subviews

### Subspan

```cpp
nd_span<T> subspan(size_t dim, size_t start, size_t end);
nd_span<const T> subspan(size_t dim, size_t start, size_t end) const;
```

Extract a range along one dimension:
```cpp
auto rows = arr.subspan(0, 1, 3);
```

### Slice

```cpp
nd_span<T> slice(size_t dim, size_t index);
nd_span<const T> slice(size_t dim, size_t index) const;
```

Fix one dimension, reducing rank:
```cpp
auto layer = tensor.slice(0, 1);
```

## Queries

```cpp
size_t rank() const;
size_t extent(size_t dim) const;
size_t stride(size_t dim) const;
auto extents() const; // view over active extents
size_t size() const;
T* data();
const T* data() const;
T* begin();
T* end();
const T* begin() const;
const T* end() const;
```

## Memory Layout

- **Row-major order**: Last index varies fastest
- **Contiguous storage**: Single heap allocation
- **Stride-based indexing**: Efficient multi-dimensional access

## Performance Characteristics

- **Construction**: O(n) for total elements (one allocation + zero init)
- **Element access**: O(rank) offset computation
- **Subview creation**: O(1), no data copies
- **Copy**: O(n) deep copy

## Safety Considerations

1. **Bounds checking**: Indexing throws `std::out_of_range` on invalid access
2. **Max rank**: Compile-time maximum via the `MaxRank` template parameter
3. **Subviews are views**: Slices and subspans alias the original data
4. **Const views**: Subviews from const arrays return `nd_span<const T>`
5. **Reshape/flatten**: These are views and do not copy data

## Comparison with nd_span

| Feature | nd_array | nd_span |
|---------|----------|---------|
| Memory ownership | Owns (unique_ptr) | Non-owning (raw pointer) |
| Allocation | Yes (construction) | No |
| Copy | Deep copy | Shallow (copies view) |
| Use case | Data storage | View/wrapper |
| C-interop | No | Yes |

