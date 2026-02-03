# Catalog Module: Accessing Reference Star Catalogs

## What is it?

The `Catalog` class provides a unified interface for querying photometric reference star catalogs. It inherits from `astropy.table.Table`, so once you have a catalog, you work with it exactly like any astropy table -- indexing columns by name, slicing rows, stacking, etc.

When you create a `Catalog`, it immediately queries the specified service (local files, VizieR, MAST, or Gaia TAP) and returns the result as a table with normalized column names.

## Quickstart

```python
from pyrt.catalog.catalog import Catalog

# Query Gaia DR3 around RA=150, Dec=+2, stars brighter than 18 mag
cat = Catalog(
    catalog=Catalog.GAIA,
    ra=150.0,
    dec=2.0,
    width=0.25,
    height=0.25,
    mlim=18.0,
)

print(f"Got {len(cat)} stars")
print(cat['radeg', 'decdeg', 'G', 'BP', 'RP'][:5])
```

## Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `catalog` | `str` | (required) | Catalog identifier -- use the class constants listed below |
| `ra` | `float` | `None` | Field center RA in degrees |
| `dec` | `float` | `None` | Field center Dec in degrees |
| `width` | `float` | `0.25` | Field width in degrees |
| `height` | `float` | `0.25` | Field height in degrees |
| `mlim` | `float` | `17.0` | Magnitude limit (faint end) |

## Available catalogs

| Constant | Value | Service | Epoch | Filters |
|----------|-------|---------|-------|---------|
| `Catalog.ATLAS` | `'atlas@localhost'` | Local `atlas` binary | 2015.5 | Sloan g,r,i,z + J + Johnson B,V,R,I |
| `Catalog.ATLAS_VIZIER` | `'atlas@vizier'` | VizieR | 2015.5 | Same as ATLAS |
| `Catalog.PANSTARRS` | `'panstarrs'` | MAST | 2015.5 | g, r, i, z, y |
| `Catalog.GAIA` | `'gaia'` | Gaia TAP | 2016.0 | G, BP, RP + Johnson B,V,R,I (transformed) |
| `Catalog.MAKAK` | `'makak'` | Local FITS file | 2016.0 | Same as Gaia |
| `Catalog.SDSS` | `'sdss'` | VizieR | 2000.0 | Sloan u,g,r,i,z |
| `Catalog.USNOB` | `'usno'` | VizieR | 2000.0 | B1, R1, B2, R2, I |

## Columns you always get

Every catalog produces these columns (normalized names):

| Column | Description |
|--------|-------------|
| `radeg` | Right ascension in degrees |
| `decdeg` | Declination in degrees |
| `pmra` | Proper motion in RA (deg/yr) |
| `pmdec` | Proper motion in Dec (deg/yr) |

Plus magnitude columns specific to each catalog (see table above). Error columns are also available where the source catalog provides them (e.g. `dg`, `dr` for Pan-STARRS; `G_err`, `BP_err` for Gaia).

## Catalog properties

Since `Catalog` is an astropy Table, you can use it as one. It also exposes catalog-specific metadata through properties:

```python
cat.catalog_name   # e.g. 'gaia'
cat.epoch          # e.g. 2016.0
cat.description    # e.g. 'Gaia Data Release 3'
cat.query_params   # QueryParams(ra=150.0, dec=2.0, width=0.25, ...)

# Available filters as a dict of CatalogFilter objects:
for name, filt in cat.filters.items():
    print(f"{name}: wl={filt.effective_wl} A, system={filt.system}")
```

## Loading a catalog from a file

If you already have a catalog saved to disk (FITS, ECSV, etc.):

```python
cat = Catalog.from_file("my_catalog.fits")
```

This reads the file via `astropy.table.Table.read()` and wraps the result in a `Catalog` object.

## How pyrt uses the Catalog internally

In the standard pyrt pipeline (`pyrt-dophot`), the flow is:

1. **`cli/dophot.py`** calls `process_image_with_dynamic_limits()` from `core/match_stars.py`
2. **`core/match_stars.py`** instantiates `Catalog` using the image center coordinates and field size from the detection table metadata:
   ```python
   cat = Catalog(
       ra=det.meta['CTRRA'],
       dec=det.meta['CTRDEC'],
       width=enlarge * det.meta['FIELD'],
       height=enlarge * det.meta['FIELD'],
       mlim=maglim,
       catalog='makak' if options.makak else options.catalog,
   )
   ```
3. Catalog RA/Dec are transformed to pixel coordinates via WCS, then matched against detected sources using a KDTree.
4. **`core/data_handling.py`** receives the matched catalog and extracts magnitudes and colors for photometric fitting. It accesses columns by the filter name stored in `det.meta['PHFILTER']` (e.g. `cat['Johnson_V']`) and computes colors from the photometric schema (e.g. `cat['g'] - cat['r']`).

The catalog is typically queried twice: first with a conservative magnitude limit for an initial zeropoint estimate, then again with a refined limit based on that estimate.

## Dependencies

- **All catalogs**: `astropy`
- **Pan-STARRS**: `astroquery` (MAST service)
- **Gaia**: `astroquery` (Gaia TAP service)
- **VizieR-based** (ATLAS VizieR, USNO-B, SDSS): `astroquery` (VizieR service)
- **Local ATLAS**: the `atlas` command-line binary and catalog data files
- **MAKAK**: a pre-filtered FITS file at the configured path

## Examples

### Query Pan-STARRS and inspect colors

```python
from pyrt.catalog.catalog import Catalog

cat = Catalog(catalog=Catalog.PANSTARRS, ra=83.633, dec=22.014, mlim=19.0)

# g-r color
gr = cat['g'] - cat['r']
print(f"Median g-r = {np.median(gr):.2f}")
```

### Query multiple catalogs for the same field

```python
gaia = Catalog(catalog=Catalog.GAIA, ra=150.0, dec=2.0, mlim=17.0)
ps1  = Catalog(catalog=Catalog.PANSTARRS, ra=150.0, dec=2.0, mlim=17.0)

print(f"Gaia: {len(gaia)} stars (epoch {gaia.epoch})")
print(f"PS1:  {len(ps1)} stars (epoch {ps1.epoch})")
```

### Use the local ATLAS catalog

```python
cat = Catalog(
    catalog=Catalog.ATLAS,
    ra=210.0,
    dec=45.0,
    width=0.5,
    height=0.5,
    mlim=19.0,
)

# ATLAS provides Sloan + transformed Johnson magnitudes
print(cat['Sloan_g', 'Sloan_r', 'Johnson_B', 'Johnson_V'][:5])
```

### Access filter metadata

```python
cat = Catalog(catalog=Catalog.GAIA, ra=150.0, dec=2.0)

for name, f in cat.filters.items():
    print(f"{name:12s}  {f.effective_wl:6.0f} A  {f.system}")
# G              5890 A  AB
# BP             5050 A  AB
# RP             7730 A  AB
# Johnson_B      4353 A  Vega
# Johnson_V      5477 A  Vega
# Johnson_R      6349 A  Vega
# Johnson_I      8797 A  Vega
```
