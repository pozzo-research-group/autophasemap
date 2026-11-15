## Installation

`autophasemap` supports Python 3.10 or newer. It depends on
[warping](https://github.com/kiranvad/warping), which compiles a C extension during
installation, so a C compiler must be available:

- macOS: Xcode command line tools (`xcode-select --install`)
- Linux: `gcc` (on a cluster, load it first, e.g. `module load gcc`)
- Windows: a working C compiler is harder to set up; one option is
  [MinGW with VSCode](https://code.visualstudio.com/docs/cpp/config-mingw), or use WSL.

### With pip

```bash
pip install git+https://github.com/pozzo-research-group/autophasemap.git
```

All dependencies, including `warping`, are installed automatically.

### For development

```bash
git clone https://github.com/pozzo-research-group/autophasemap.git
cd autophasemap
pip install -e ".[test]"
pytest
```

Add the `notebook` extra (`pip install -e ".[test,notebook]"`) to run `example.ipynb`.

### With conda

`environment.yml` creates an environment with Python, a C compiler from conda-forge and
an editable install of this repository (including the test and notebook extras):

```bash
git clone https://github.com/pozzo-research-group/autophasemap.git
cd autophasemap
conda env create --file environment.yml
conda activate autophasemap
```

### Reproducing the paper

The code and environment used for the paper are archived at
[pozzo-research-group/papers](https://github.com/pozzo-research-group/papers/tree/main/autophasemap)
(tag `v1.0.0-autophasemap`).
