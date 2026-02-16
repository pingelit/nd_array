# nd_array

A high-performance C++ template class for n-dimensional arrays with minimal memory allocations.

The `nd_array` class provides an owning, heap-allocated N-dimensional array with a compact metadata layout and an mdspan-like interface.
It stores elements in a single contiguous allocation and provides efficient subviews via `nd_span`.
The `nd_span` class provides a non-owning view over multidimensional data, allowing wrapping raw C-arrays or other contiguous memory buffers with a modern C++ interface.

This library was created through my first dabble into coding with an AI agent.
It was a fun experiment to see how much I could build with the help of an AI agent on a weekend.

## Build quick start

```bash
cmake --preset windows-debug
cmake --build --preset windows-debug
```

## Documentation site

Documentation is built with MkDocs Material and MkDoxy.
See the site content in the `docs/` directory.

```bash
uv sync
uv run mkdocs serve
```

