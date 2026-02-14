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
- `[subspan]` - Subspan tests (various dimensions, modifications, error handling)
- `[slice]` - Slice tests (dimension reduction, modifications, error handling)
- `[properties]` - Property query tests
- `[const]` - Const correctness tests
- `[c-interop]` - C API interoperability (wrapping C-arrays, std::array, std::vector)
- `[integration]` - Integration tests

Run specific tests:
```bash
./nd_array_tests [nd_array]
./nd_array_tests [nd_span]
./nd_array_tests [subspan]
```
