# nd_array

A high-performance C++ template class for n-dimensional arrays with minimal memory allocations and mdspan-like interface.

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

