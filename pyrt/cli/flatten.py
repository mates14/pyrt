#!/usr/bin/env python3
"""
flatten.py - fold the spatial and nonlinearity response of a dophot fit into MAG_AUTO

foo.ecsv -> foo-flat.ecsv with only the color response (and Z) left in RESPONSE,
so that many such files can go to one dophot run fitting the color response alone
(same as dophot --remove-spatial, but leaving the files for inspection/reuse).
"""

import os
import sys
import logging
import argparse

from astropy.table import Table

from pyrt.core import fotfit

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="dophot output .ecsv files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(levelname)s - %(message)s')

    failed = 0
    for fn in args.files:
        out = os.path.splitext(fn)[0] + "-flat.ecsv"
        try:
            det = Table.read(fn, format="ascii.ecsv")
            if 'RESPONSE' not in det.meta:
                raise KeyError("no RESPONSE in the header, not a dophot output?")
            fotfit.flatten_response(det)
            det.write(out, format="ascii.ecsv", overwrite=True)
            if args.verbose:
                print(f"{fn} -> {out}")
        except Exception as e:
            print(f"{fn}: {e}", file=sys.stderr)
            failed += 1
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
