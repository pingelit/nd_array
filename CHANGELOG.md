# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Core API: `nd_array` and `nd_span`
- Example/Demo program
- Array features: subspans from const objects, container constructors, iterators, `stride()`/`extents()`, shape ops (`reshape`, `transpose`/`T`, `flatten`, `squeeze`), and deep-copy conversion from `nd_span` to `nd_array`.
- Tooling and docs: unit tests, mkdocs site, Doxygen docs, and version bumping config.
- Static analysis: clang-tidy integration with follow-up fixes.

### Changed

- CMake setup refined and clang-tidy targets made more specific.
- Identifier naming conventions and docstrings aligned, with clang-format applied.

### Fixed

- Const-casting issue in span-to-array conversion.
- Initializer list ordering issue.

