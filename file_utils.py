import os
import astropy.io.fits
import astropy.wcs
import logging
from contextlib import suppress
from astropy.table import Table

def try_img(arg, verbose=False):
    """Try to open arg as a fits file, exit cleanly if it does not happen"""
    try:
        with suppress(astropy.wcs.FITSFixedWarning):
            fitsfile = astropy.io.fits.open(arg)
        logging.info(f"Argument {arg} is a fits file")
        return arg, fitsfile
    except (FileNotFoundError,OSError):
        logging.debug(f"Argument {arg} is not a fits file")
        return None, None

def try_sex(arg, verbose=False):
    """Try to open arg as a sextractor file, exit cleanly if it does not happen"""
    try:
        det = astropy.io.ascii.read(arg, format='sextractor')
        logging.info(f"Argument {arg} is a sextractor catalog")
        return arg, det
    except (FileNotFoundError,OSError,UnicodeDecodeError,astropy.io.ascii.core.InconsistentTableError):
        logging.debug(f"Argument {arg} is not a sextractor catalog")
        return None, None

# remove_meta must be True for compatibility reasons
def try_ecsv(arg, verbose=False, remove_meta=False):
    """Try to open arg as a sextractor file, exit cleanly if it does not happen"""
    try:
        det = astropy.io.ascii.read(arg, format='ecsv')
        if remove_meta:
            det.meta=None # certainly contains interesting info, but used to break the code
        logging.info(f"Argument {arg} is an ascii/ecsv catalog")
        return arg, det
    except (FileNotFoundError,OSError,UnicodeDecodeError):
        logging.debug(f"Argument {arg} is not an ascii/ecsv catalog")
        return None, None

def try_det(arg, verbose=False):
    """Try to open arg as an ecsv file, exit cleanly if it does not happen"""
    try:
        detfile = Table.read(arg, format="ascii.ecsv")
        logging.info(f"Argument {arg} is an ecsv table")
        return arg, detfile
    except (FileNotFoundError,OSError,UnicodeDecodeError,ValueError):
        pass
    try:
        detfile = Table.read(arg)
        logging.info(f"Argument {arg} is a table")
        return arg, detfile
    except (FileNotFoundError,OSError,UnicodeDecodeError,ValueError):
        logging.debug(f"Argument {arg} is not a table")
        return None, None

import astropy.table

def write_region_file(catalog: Table, filename: str,
                      color: str = "red", shape: str = "circle",
                      radius: float = 3.0, coord_system: str = "fk5") -> None:
    """
    Write a DS9 region file based on the provided catalog.

    Parameters:
    -----------
    catalog : astropy.table.Table
        The catalog containing the star positions.
    filename : str
        The name of the output region file.
    color : str, optional
        The color of the regions (default is "red").
    shape : str, optional
        The shape of the regions (default is "circle").
    radius : float, optional
        The radius of the regions in arcseconds (default is 3.0).
    coord_system : str, optional
        The coordinate system of the catalog (default is "fk5").

    Returns:
    --------
    None
    """
    keys = ['radeg','decdeg']
    with open(filename, "w") as region_file:
        # Write the header
        region_file.write(f"# Region file format: DS9 version 4.1\n")
        region_file.write(f"global color={color} dashlist=8 3 width=1 ")
        region_file.write(f"font=\"helvetica 10 normal roman\" select=1 ")
        region_file.write(f"highlite=1 dash=0 fixed=0 edit=1 move=1 ")
        region_file.write(f"delete=1 include=1 source=1\n")
        if coord_system is not None:
            region_file.write(f"{coord_system}\n")

        # Write a region for each star in the catalog
        for star in catalog:
            ra = star[keys[0]]
            dec = star[keys[1]]
            region_file.write(f"{shape}({ra:.7f},{dec:.7f},{radius}\") # color={color}\n")

    print(f"Region file '{filename}' has been created.")

def exportColumnsForDS9(columns, file="ds9.reg", size=10, width=3, color="red"):
    some_file = open(file, "w+")
    some_file.write(f"# Region file format: DS9 version 4.1\nglobal color={color} dashlist=8 3 width={width}"\
        " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
    for a, d in zip(columns[0], columns[1]):
        some_file.write(f"circle({a:.7f},{d:.7f},{size:.3f}\") # color={color} width={width}\n")
