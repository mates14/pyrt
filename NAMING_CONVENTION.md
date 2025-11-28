# PYRT Naming Convention

## Source Files vs Installed Commands

PYRT follows Python packaging best practices where **source file names** and **installed command names** are separate.

### Structure

```
Repository/Git:
pyrt/cli/dophot.py          ← Source file (clean name, in package context)

After Installation:
$ pyrt-dophot               ← Installed command (namespaced, no conflicts)
```

### Why This Approach?

**1. Avoid Name Conflicts**
- Many projects have a "dophot" or similar names
- `pyrt-dophot` clearly identifies it as part of PYRT
- No conflicts with other installed tools

**2. Clean Source Code**
- Source files have simple, clean names: `dophot.py`, `phcat.py`
- Within the `pyrt.cli` package context, these names make sense
- No need for long prefixes in imports: `from pyrt.cli import dophot`

**3. Flexible Deployment**
- Entry points in `pyproject.toml` map commands to source
- Can change command names without touching source code
- Easy to create aliases or alternative names

### Mapping Table

| Source File | Installed Command | Description |
|-------------|-------------------|-------------|
| `pyrt/cli/dophot.py` | `pyrt-dophot` | Main photometric calibration |
| `pyrt/cli/phcat.py` | `pyrt-phcat` | Source extraction |
| `pyrt/cli/cat2det.py` | `pyrt-cat2det` | Detection table preparation |
| `pyrt/cli/zpn_to_tan.py` | `pyrt-zpn-to-tan` | ZPN to TAN conversion |
| `pyrt/cli/mProjectPX.py` | `pyrt-mproject` | Fast reprojection |
| `pyrt/cli/combine-images` | `pyrt-combine-images` | Image combination |
| `pyrt/cli/cphead` | `pyrt-cphead` | Copy FITS headers |
| `pyrt/cli/edhead` | `pyrt-edhead` | Edit FITS headers |
| `pyrt/cli/sky2xy` | `pyrt-sky2xy` | Sky to pixel conversion |

### Entry Point Configuration

In `pyproject.toml`:

```toml
[project.scripts]
pyrt-dophot = "pyrt.cli.dophot:main"
#    ↑              ↑         ↑
# command name   module    function
```

This tells pip/setuptools:
- Create a command called `pyrt-dophot`
- When run, execute `main()` from `pyrt.cli.dophot` module

### Internal Tool Calls

When dophot calls other tools (like cat2det), it uses the **installed command name**:

```python
# In dophot.py
if shutil.which("pyrt-cat2det"):
    cmd = ["pyrt-cat2det"]  # ← Uses installed command name
else:
    # Fallback for running from source
    cmd = [sys.executable, "-m", "pyrt.cli.cat2det"]
```

This ensures:
- ✅ Works after installation (finds `pyrt-cat2det` in PATH)
- ✅ Works from source (falls back to module execution)
- ✅ Uses consistent command names

### For Users

After `pip install pyrt`, use commands with `pyrt-` prefix:

```bash
# Correct (installed commands)
pyrt-dophot image.fits
pyrt-phcat image.fits
pyrt-cat2det catalog.cat

# Not these (source file names)
dophot.py image.fits        # Won't work
phcat.py image.fits         # Won't work
```

### For Developers

When developing from source without installation:

```bash
# Run as Python module
python -m pyrt.cli.dophot image.fits
python -m pyrt.cli.phcat image.fits

# Or set PYTHONPATH and install in development mode
pip install -e .
pyrt-dophot image.fits      # Now works
```

### Backward Compatibility Note

The config file path remains `~/.config/dophot3/config` for backward compatibility with existing installations. This may be updated to `~/.config/pyrt/config` in a future major version.

### Benefits

1. **Namespace Protection**: `pyrt-*` commands clearly belong to PYRT
2. **Clean Sources**: Simple file names in source code
3. **No Conflicts**: Won't clash with other "dophot" tools
4. **Professional**: Follows Python packaging standards
5. **Flexible**: Can add aliases or change names easily

### Examples from Other Projects

Many Python projects follow this pattern:

| Project | Source | Installed Command |
|---------|--------|-------------------|
| black | `black/__main__.py` | `black` |
| pytest | `pytest/__main__.py` | `pytest` |
| pip | `pip/__main__.py` | `pip` |
| django | `django/core/management/__init__.py` | `django-admin` |

PYRT follows the same professional standard, just with explicit `pyrt-` prefix for clarity.
