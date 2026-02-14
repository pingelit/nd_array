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
- **Modern C++ Features**: `[[nodiscard]]`, `constexpr`, `noexcept` for better safety and performance
- **Comprehensive Documentation**: Full Doxygen-style inline documentation with examples
- **Comprehensive Tests**: Full unit test suite with Catch2

## Quick Start

### Building

```bash
# Configure with tests and examples enabled (default)
cmake --preset windows-debug

# Build
cmake --build --preset windows-debug
```

## Reference

For design rationale and additional documentation see [nd_array guide](nd-array-guide.md) and [nd_span guide](nd-span-guide.md) as well as the [testing guide](testing.md) and [examples](examples.md).

## Limitations

- Maximum rank is fixed at compile time
- No automatic broadcasting or reshaping
- Slicing creates views, not copies (modify views to modify original)
- No built-in serialization
