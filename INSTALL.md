# PYRT Installation Guide

## Package Structure

The PYRT package has been reorganized into a professional pip-installable structure:

```
pyrt/
├── pyproject.toml          # Modern Python packaging configuration
├── requirements.txt        # Python dependencies
├── MANIFEST.in            # Distribution manifest
├── README.md              # Main documentation
├── INSTALL.md            # This file
├── docs/                  # Documentation directory
│   ├── dophot-usage.md
│   ├── terms-explained.md
│   ├── filter-estimation.md
│   ├── GRB_DATA_FORMAT.md
│   └── SZP-todo.md
├── pyrt/                  # Main package
│   ├── __init__.py
│   ├── core/              # Fitting engines (fotfit, termfit, zpnfit, grbfit)
│   ├── data/              # Data management (catalog, data_handling, match_stars)
│   ├── astrometry/        # Astrometric refinement modules
│   ├── regression/        # Stepwise regression
│   ├── utils/             # Utilities (config, plotting, file_utils, etc.)
│   └── cli/               # Command-line interface scripts
└── tests/                 # Test directory (for future tests)
```

## Installation Methods

### Method 1: Development Installation (Recommended for Development)

For local development and testing:

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Method 2: Direct Installation from Source

```bash
# In a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install directly
pip install .
```

### Method 3: Installation from Git Repository

Once pushed to GitHub:

```bash
pip install git+https://github.com/yourusername/pyrt.git
```

### Method 4: Installation from PyPI

Once published to PyPI:

```bash
pip install pyrt
# or if "pyrt" is taken:
pip install pyrt-astro  # or whatever name you choose
```

## System Requirements

### Python Requirements
- Python >= 3.10
- See `requirements.txt` for Python package dependencies

### External Tools (Optional but Recommended)

These tools are not installed via pip and must be installed separately:

1. **SExtractor** (required for source extraction)
   ```bash
   # Debian/Ubuntu
   sudo apt install sextractor
   ```

2. **astrometry.net** (required for initial WCS solutions)
   ```bash
   # Debian/Ubuntu
   sudo apt install astrometry.net
   ```

3. **IRAF** (optional, for precision aperture photometry)
   - Download from: https://iraf-community.github.io/

4. **Montage** (optional, for fast image reprojection)
   ```bash
   # Debian/Ubuntu
   sudo apt install montage
   ```

## Command-Line Tools

After installation, the following commands will be available:

### Main Tools
- `pyrt-dophot` - Main photometric calibration and astrometric refinement
- `pyrt-phcat` - Source extraction (SExtractor wrapper)
- `pyrt-cat2det` - Detection table preparation

### GRB Analysis
- `pyrt-grbphot` - GRB afterglow lightcurve fitting

### Astrometric Utilities
- `pyrt-zpn-to-tan` - Convert ZPN to TAN projection
- `pyrt-mproject` - Fast ZPN reprojection wrapper

### FITS Utilities
- `pyrt-ecsv-target` - Target-specific photometry
- `pyrt-fits-overlap` - Image overlap detection
- `pyrt-f2cj` - FITS to CJ conversion

## Usage Examples

### Basic Usage

After installation:

```bash
# Main photometric calibration
pyrt-dophot image.fits

# Source extraction
pyrt-phcat image.fits

# GRB lightcurve fitting
pyrt-grbphot data.ecsv
```

### Using as a Python Library

```python
# Import core fitting engines
from pyrt.core import fotfit, termfit, zpnfit

# Import data management
from pyrt.data import PhotometryData, Catalog

# Import utilities
from pyrt.utils import load_config

# Example: Create a photometric fitter
fitter = fotfit.fotfit()
```

## Verification

Test that the installation was successful:

```bash
# Check that commands are available
pyrt-dophot --help
pyrt-phcat --help

# Test Python imports
python3 -c "import pyrt; print(pyrt.__version__)"
python3 -c "from pyrt.core import fotfit; print('Success!')"
```

## Troubleshooting

### PEP 668 Error (Externally Managed Environment)

If you see an error about "externally-managed-environment":

**Solution 1**: Use a virtual environment (recommended)
```bash
python3 -m venv ~/.venvs/pyrt
source ~/.venvs/pyrt/bin/activate
pip install -e .
```

**Solution 2**: Use pipx for isolated installation
```bash
pipx install .
```

### Import Errors

If you encounter import errors:

1. Make sure you're in the correct directory
2. Check that all `__init__.py` files exist
3. Verify Python version is >= 3.10
4. Try reinstalling: `pip install --force-reinstall -e .`

### Missing External Tools

If SExtractor or astrometry.net are not found:

1. Install the system packages (see above)
2. Make sure they're in your PATH
3. Test: `which sex` and `which solve-field`

## Publishing to PyPI

### Check if "pyrt" is Available

```bash
pip search pyrt  # Note: search might be disabled
# Or check: https://pypi.org/project/pyrt/
```

If "pyrt" is taken, consider alternatives:
- `pyrt-astro`
- `pyrt-phot`
- `pyrt-photometry`

### Publishing Steps

1. **Update metadata in `pyproject.toml`**:
   - Update repository URLs
   - Verify author information
   - Choose correct license

2. **Build the package**:
   ```bash
   pip install build twine
   python -m build
   ```

3. **Test on Test PyPI first**:
   ```bash
   twine upload --repository testpypi dist/*
   ```

4. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

## Git Workflow

All file moves were done with `git mv` to preserve history:

```bash
# Check what was moved
git status

# Commit the restructuring
git add .
git commit -m "Restructure package for pip installation

- Reorganize into pyrt/ package structure
- Move core modules to pyrt/core/
- Move CLI scripts to pyrt/cli/
- Move documentation to docs/
- Add pyproject.toml for modern packaging
- Update all imports to use package namespacing
- Add console script entry points (pyrt-dophot, pyrt-phcat, etc.)
"

# Push to repository
git push origin main
```

## For Contributors

If you want to contribute:

```bash
# Clone the repository
git clone https://github.com/yourusername/pyrt.git
cd pyrt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest

# Format code
black pyrt/

# Type checking
mypy pyrt/
```

## Configuration

PYRT looks for configuration in:
1. `~/.config/dophot3/config`
2. Command-line arguments (override config file)

See `docs/dophot-usage.md` for detailed configuration options.

## Documentation

- Main README: `README.md`
- Usage guide: `docs/dophot-usage.md`
- Term syntax reference: `docs/terms-explained.md`
- Filter handling: `docs/filter-estimation.md`
- GRB data format: `docs/GRB_DATA_FORMAT.md`

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/pyrt/issues
- Email: martin.jelinek@asu.cas.cz
