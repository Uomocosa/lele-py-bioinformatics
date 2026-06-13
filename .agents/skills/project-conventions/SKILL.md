---
name: project-conventions
description: Python project conventions for this bioinformatics workspace — new-import-system, atomic *Method/ subpackages, dataclass/enum patterns, test_usage rules, import style (absolute only), logging with loguru, empty __init__.py conventions, PascalCase subpackages, __HELPER_DIR__ assets.
---

# Project Conventions (Python)

This project uses a custom `new-import-system` for absolute imports. All rules here override standard Python conventions.

## 1. Import System: new-import-system

### Top-level `__init__.py`

Every top-level package `__init__.py` must install it:

```python
import new_import_system
new_import_system.install(__file__)
```

### Import Rules

- **NEVER use relative imports** (`from .Module import ...` is forbidden)
- **Always use absolute imports** (e.g., `import my_package`, `from my_package.Module import MyClassMethod`)
- **No need to repeat module name** in function calls when the module name matches the function name:

```python
# Instead of:
my_package.Module.MyClassMethod.my_function(df, options)

# Do:
my_package.Module.MyClassMethod.my_function(df, options)
# OR if my_function is the only exported function:
my_package.Module.MyClassMethod(df, options)
```

### Empty `__init__.py` Files

All `__init__.py` files EXCEPT the top-level one must be **empty**:

```
my_package/
├── __init__.py          # Contains new-import-system installation
├── Module/
│   ├── __init__.py      # EMPTY
│   ├── MyClass.py
│   └── MyClassMethod/
│       ├── __init__.py  # EMPTY
│       └── my_method.py
```

## 2. Code Granularity & Organization

### Dataclasses, Enums, and Functions

Prefer dataclasses and enums for structured data. Regular functions are also welcome at the package level.

```python
from dataclasses import dataclass
from enum import IntEnum

@dataclass
class Config:
    csv_file: Path
    max_size: int = 100

class Status(IntEnum):
    UNKNOWN = -1
    INACTIVE = 0
    ACTIVE = 1

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['value'] > 0]
```

**Never use manual `__init__`** in classes that should be dataclasses.

### Method Subpackages: `ClassNameMethod/`

When a dataclass needs methods, create a subpackage named `ClassNameMethod/` and implement methods there.

```
my_package/Module/
├── MyClass.py              # Dataclass definition
└── MyClassMethod/
    ├── __init__.py         # EMPTY
    ├── my_method.py        # Method implementation
    └── process_data.py
```

**Dataclass references methods via the subpackage:**

```python
# my_package/Module/MyClass.py
from dataclasses import dataclass
import my_package

@dataclass
class MyClass:
    data: pd.DataFrame
    config: Config

    def my_method(self, options=MyClassMethod.my_method.Options()):
        return MyClassMethod.my_method(self.data, options)

    def another_method(self, options=MyClassMethod.another_method.Options()):
        return MyClassMethod.another_method(self.data, options)
```

### Function Organization in Method Modules

Each method file follows this pattern:

```python
# my_package/Module/MyClassMethod/my_method.py
from dataclasses import dataclass, field
import pandas as pd
import my_package

@dataclass
class Options:
    option_a: str = "default_value"
    n_points: int = 2

def my_method(df: pd.DataFrame, options: Options = Options()) -> pd.DataFrame:
    return df

def test_usage():
    from my_package.__global__ import DATA_CSV
    df = pd.read_csv(DATA_CSV)
    df = my_method(df, Options(option_a="value"))
    assert len(df) > 0
```

## 3. Testing Conventions

### `test_usage()` Function Pattern

Every module (function, dataclass, enum) must have at least one `test_usage()` function with **no arguments**:

```python
def test_usage():
    from my_package.__global__ import DATA_CSV
    config = Config(csv_file=DATA_CSV)
    instance = MyClass(config)
    instance.my_method()
    logger.info(f"Data shape: {instance.data.shape}")
```

### Multiple Test Functions

Use descriptive names when testing different aspects:

```python
def test_method_a():
    pass

def test_method_b():
    pass

def test_complete_workflow():
    pass
```

### Pytest Markers

Use pytest markers for test categorization:

```python
import pytest

@pytest.mark.above10s
def test_long_running():
    pass

@pytest.mark.skip(reason="Needed once for specific debugging")
def test_debug_one_time():
    pass

@pytest.mark.todo
def test_not_yet_implemented():
    pass

@pytest.mark.unreliable
def test_external_dependency():
    pass

@pytest.mark.verbose
def test_with_output():
    pass
```

## 4. Global Constants: `__global__.py`

Use `__global__.py` files for constants and shared configuration:

```python
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_DIR / 'DATA'
RESULTS_DIR = REPO_DIR / 'RESULTS'

from joblib import Memory
CACHE_MEMORY = Memory(location=".cache_dir", verbose=0)
```

Each subpackage can have its own `__global__.py`.

## 5. Code Style Rules

### No Comments (Unless Required)

Code should be self-documenting through clear naming.

### No `if __name__ == "__main__"`

Never use `if __name__ == "__main__":` blocks. Use test functions instead.

### Logging with loguru

```python
from loguru import logger

logger.debug(f"Processing {len(df)} rows")
logger.info("Operation completed successfully")
logger.warning("Missing data detected")
```

## 6. Directory Structure

```
my_package/
├── __init__.py              # new-import-system installation
├── __global__.py            # Global constants
├── ModuleA/
│   ├── __init__.py          # EMPTY
│   ├── __global__.py        # Module-specific constants
│   ├── Config.py            # Config dataclass
│   ├── MyClass.py           # Main dataclass
│   └── MyClassMethod/       # Methods for MyClass
│       ├── __init__.py      # EMPTY
│       ├── my_method.py
│       └── ...
```

### Top-Level Package Naming

The top-level package folder must be in `snake_case` and **must match `[project].name` in `pyproject.toml`**.

### Subpackage Naming (PascalCase)

All subpackages (folders with `__init__.py`) must use **PascalCase**:

```
✅ MyPackage/, ModuleA/, BioInformatics/, Utils/
❌ my_package/, module_a/, bio_informatics/
```

### `__HELPER_DIR__` Convention

Each top-level package should contain a `__HELPER_DIR__` subfolder for non-code assets with a `.gitkeep` file:

```
library/Package/
├── __init__.py
├── __HELPER_DIR__/
│   ├── .gitkeep
│   └── template.txt
└── __globals__.py
```

Import pattern:
```python
from library.Package.__globals__ import HELPER_DIR
```

## 7. Naming Conventions Summary

| Element | Convention | Example |
|---------|------------|---------|
| **Top-level package** | snake_case | `my_package/` |
| **Subpackages** | PascalCase | `Utils/`, `BioInformatics/` |
| **Classes** | PascalCase | `MyClass`, `Config` |
| **Enums** | PascalCase | `Status`, `LogLevel` |
| **Functions** | snake_case | `my_function`, `process_data` |
