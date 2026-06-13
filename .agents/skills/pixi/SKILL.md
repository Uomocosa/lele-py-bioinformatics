---
name: pixi
description: Use when the user wants to run tests, install dependencies, build the project, or manage the Python environment via pixi. Covers pytest invocation, environment activation, dependency management, and common pixi commands for this project.
---

# Pixi Environment & Testing

This project uses **pixi** for environment and dependency management.

## 1. Running Tests

### Run All Tests

```bash
pixi run pytest
```

### Run a Specific Test File

```bash
pixi run pytest bio/Module/MyClass.py
```

### Run a Specific Test Function

```bash
pixi run pytest bio/Module/MyClass.py::test_usage -o "addopts="
```

### Run with Verbose/Print Output

```bash
pixi run pytest -s -v bio/Module/MyClass.py::test_usage
```

### Run Without Default Marker Filters

The project's `pyproject.toml` sets `addopts = "-m 'not verbose and not todo and not above10s and not unreliable and not infinite'"`. To override:

```bash
pixi run pytest -m "" bio/Module/MyClass.py -o "addopts="
```

### Available Test Markers

| Marker | Meaning |
|--------|---------|
| `verbose` | Tests with visible output |
| `todo` | Feature not yet implemented |
| `above10s` | Requires more than 10 seconds |
| `unreliable` | Depends on external factors |
| `infinite` | Will not finish |

## 2. Environment Management

### Activate Environment

```bash
pixi shell
```

### Install a New Dependency

```bash
pixi add <package-name>
```

### Install a Development Dependency

```bash
pixi add --dev <package-name>
```

### Update Lockfile

```bash
pixi update
```

## 3. Build

Build the package:

```bash
pixi run build
```
