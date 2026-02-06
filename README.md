## CI Status

| Python | Status |
|--------|--------|
| 3.8 | ![3.8](https://github.com/Uomocosa/lele-py-bioinformatics-bioinformatics/actions/workflows/test-3-8.yml/badge.svg) |
| 3.9 | ![3.9](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-9.yml/badge.svg) |
| 3.10 | ![3.10](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-10.yml/badge.svg) |
| 3.11 | ![3.11](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-11.yml/badge.svg) |
| 3.12 | ![3.12](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-12.yml/badge.svg) |
| 3.13 | ![3.13](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-13.yml/badge.svg) |
| 3.14 | ![3.14](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-14.yml/badge.svg) |

# How to use
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. `git clone https://github.com/Uomocosa/lele-py-bioinformatics`
3. `uv sync`
4. `uv pip install --upgrade torch --torch-backend=auto` (Optional, if you want to use GPU)
5. `uv tool install --editable . --python 3.11`
<!--6. `cacca_train`-->s

# My personal bioinformatics library
- I used this library and app for my master's thesis.
- It uses my [new-python-import-system](https://github.com/Uomocosa/new-python-import-system). The imports might seem more _magical_ than the usual.
