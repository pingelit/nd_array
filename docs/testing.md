# Unit Testing Setup

The unit tests are implemented via Catch2.

## Test Categories & running tests

Tests are organized with tags:
- `[nd_array]` - nd_array tests
- `[nd_span]` - nd_span tests
- `[construction]` - Construction tests (variadic, initializer list, container, wrapping vectors)
- `[access]` - Element access tests
- `[copy]` - Copy semantics
- `[move]` - Move semantics
- `[operations]` - Operations like fill, apply
- `[reshape]` - Reshape tests
- `[transpose]` - Transpose and T tests
- `[flatten]` - Flatten tests
- `[squeeze]` - Squeeze tests
- `[subspan]` - Subspan tests (various dimensions, modifications, error handling)
- `[slice]` - Slice tests (dimension reduction, modifications, error handling)
- `[properties]` - Property query tests
- `[iterators]` - Iteration and begin/end access
- `[extents]` - Extents and stride access
- `[stride]` - Stride access
- `[span]` - Span to array conversions
- `[const]` - Const correctness tests
- `[c-interop]` - C API interoperability (wrapping C-arrays, std::array, std::vector)
- `[integration]` - Integration tests

Run specific tests:
```bash
./nd_array_tests [nd_array]
./nd_array_tests [nd_span]
./nd_array_tests [subspan]
./nd_array_tests [reshape]
./nd_array_tests [iterators]
```
