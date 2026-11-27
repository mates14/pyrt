#!/usr/bin/env python3
"""
badpix - a list of bad pixels carried in the image's own header.

The pixels whose value cannot be trusted (an RTS hot pixel whose state was not told in this
frame, ...) stay in the image with their best guess, so single-frame tools (phcat, dophot)
need nothing; tools that combine frames (pyrt-combine, framestep) leave them out.

Header: BADPIX  = 'v1:<base64>'  (long string, CONTINUE cards), NBADPIX = count.
v1: linear indices i = y * NAXIS1 + x (0-based, of this image), sorted, the first and then
each difference to the previous one as an unsigned LEB128 varint, base64 without padding.
~4 characters a pixel when there are ~100 per Mpix.

Usage:  pyrt-badpix FILE.fits      prints x y (1-based FITS pixels), one per line
"""
import base64
import sys

import numpy as np

KEY, NKEY = 'BADPIX', 'NBADPIX'


def encode(y, x, nx):
    idx = np.unique(np.asarray(y, np.int64) * int(nx) + np.asarray(x, np.int64))
    d = np.diff(idx, prepend=0)
    out = bytearray()
    for v in d.tolist():
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return 'v1:' + base64.b64encode(bytes(out)).decode('ascii').rstrip('=')


def decode(s, nx):
    """(y, x) 0-based int arrays."""
    if not s:
        return np.zeros(0, int), np.zeros(0, int)
    ver, _, body = s.partition(':')
    if ver != 'v1':
        raise ValueError(f'unknown BADPIX encoding {ver!r}')
    raw = base64.b64decode(body + '=' * (-len(body) % 4))
    vals, v, sh = [], 0, 0
    for b in raw:
        v |= (b & 0x7F) << sh
        if b & 0x80:
            sh += 7
        else:
            vals.append(v); v, sh = 0, 0
    idx = np.cumsum(np.array(vals, np.int64))
    return (idx // nx).astype(int), (idx % nx).astype(int)


def to_header(header, y, x, nx=None, comment='bad pixels: pyrt.utils.badpix'):
    """Add (y, x) (0-based) to the header's list, merged with what is there."""
    nx = int(nx or header['NAXIS1'])
    y0, x0 = from_header(header, nx)
    y = np.concatenate([y0, np.asarray(y, int)]); x = np.concatenate([x0, np.asarray(x, int)])
    n = len(np.unique(y.astype(np.int64) * nx + x))
    if n == 0:
        return header
    header[KEY] = (encode(y, x, nx), comment)
    header[NKEY] = (n, 'pixels in BADPIX')
    return header


def from_header(header, nx=None):
    """(y, x) 0-based of the header's BADPIX, empty arrays when none."""
    s = header.get(KEY)
    if not s:
        return np.zeros(0, int), np.zeros(0, int)
    return decode(str(s), int(nx or header['NAXIS1']))


def main(argv=None):
    from astropy.io import fits
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return
    for f in argv:
        y, x = from_header(fits.getheader(f))
        if len(argv) > 1:
            print(f'# {f}: {len(y)}')
        for yy, xx in zip(y, x):
            print(xx + 1, yy + 1)


if __name__ == '__main__':
    main()
