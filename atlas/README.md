# ATLAS ECSV Binary (Optional)

## Description

This is a modified version of the original ATLAS refcat extraction tool with added support for ECSV (Enhanced Character Separated Values) output format.

The ECSV output provides **dramatically faster loading** of catalog data into Python/Astropy compared to traditional text formats.

## Building

Simple compilation with standard tools:

```bash
make atlas
```

Or manually:
```bash
gcc -o atlas atlas.c -lm
```

## Requirements

- C compiler (gcc, clang, etc.)
- libm (math library - standard on all Unix systems)

## Installation

After building, copy the `atlas` binary to somewhere in your PATH:

```bash
# Option 1: User bin directory
cp atlas ~/bin/

# Option 2: Local bin directory
sudo cp atlas /usr/local/bin/

# Option 3: Keep in pyrt directory
# (make sure it's in PATH or pyrt will find it locally)
```

## Usage

The tool is called automatically by PYRT's catalog module when using local ATLAS catalogs:

```python
from pyrt.catalog import Catalog

# Will use local atlas binary if available
cat = Catalog("atlas@localhost", ra=123.456, dec=45.678, radius=0.1)
```

## Note

This tool is **completely optional**. PYRT can also query ATLAS via VizieR:

```python
cat = Catalog("atlas@vizier", ra=123.456, dec=45.678, radius=0.1)
```

The local version is faster for repeated queries if you have local ATLAS refcat files.

## Modifications

This version differs from the original ATLAS refcat tool by adding ECSV output format support for efficient Python integration.
