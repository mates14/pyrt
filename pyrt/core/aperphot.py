"""Forced aperture photometry in numpy, a drop-in for IRAF digiphot.daophot.phot

The sky is IRAF's salgorithm=mode with the fitskypars defaults: pixels of the
annulus (centres within annulus..annulus+dannulus of the position, those above
datamax dropped), mode = 3 median - 2 mean (the mean if it is below the median),
the median being the upper-middle element as in IRAF, pixels further than 3 sigma
from the mode rejected and all of it repeated until none is rejected (at most
50 times).  The aperture sum weights a pixel by IRAF's partial-pixel ramp,
clip(ape - r + 0.5, 0, 1) of its centre distance r (overlap='exact' gives the true
circle-pixel overlap instead), the error is IRAF's
    merr = 1.0857 sqrt(flux/epadu + area stdev^2 + area^2 stdev^2/nsky) / flux.

Validated against IRAF on SBT full frames (forced photometry, 12k positions:
0.002 mag median difference down to S/N ~3, the same flatness against ATLAS),
on a D50 kinetic sum and on a 1024^2 frame with a 1.9 px aperture.

Coordinates are 1-based (FITS/IRAF/SExtractor convention)."""

import numpy as np
import astropy.table

MAXREJ = 50       # snreject
KREJ = 3.0        # sloreject = shireject
CHUNK = 5000     # objects processed at once (memory: CHUNK x box pixels)
NSUB = 32         # x samples per pixel of the aperture overlap


def _box(x, y, rmax, shape):
    """Pixel grid (1-based) of a box around each position, its validity and distances^2."""
    h = int(np.ceil(rmax)) + 1
    dy, dx = np.mgrid[-h:h + 1, -h:h + 1]
    ix = np.rint(x).astype(int)[:, None] + dx.ravel()[None, :]
    iy = np.rint(y).astype(int)[:, None] + dy.ravel()[None, :]
    inside = (ix >= 1) & (iy >= 1) & (ix <= shape[1]) & (iy <= shape[0])
    return ix, iy, inside


def _pixels(img, ix, iy):
    return img[np.clip(iy - 1, 0, img.shape[0] - 1), np.clip(ix - 1, 0, img.shape[1] - 1)]


def circle_overlap(ddx, ddy, r):
    """Area of each unit pixel (centre offset ddx, ddy from the circle centre) inside a circle of radius r"""
    xs = (np.arange(NSUB) + 0.5) / NSUB - 0.5
    area = np.zeros(ddx.shape)
    for s in xs:
        X = ddx + s
        half = np.sqrt(np.clip(r * r - X * X, 0, None))
        area += np.clip(np.minimum(ddy + 0.5, half) - np.maximum(ddy - 0.5, -half), 0, None)
    return area / NSUB


def sky_mode(v, k=KREJ, maxiter=MAXREJ):
    """IRAF 'mode' sky of each row of v (NaN = no pixel): mode, stdev, skew, nsky, nsrej"""
    v = v.copy()
    n0 = np.sum(np.isfinite(v), axis=1)
    rows = np.arange(len(v))
    for it in range(maxiter + 1):
        n = np.sum(np.isfinite(v), axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(v, axis=1) if v.shape[1] else np.full(len(v), np.nan)
            sd = np.sqrt(np.nansum((v - mean[:, None])**2, axis=1) / np.maximum(n - 1, 1))
        s = np.sort(v, axis=1)
        med = s[rows, np.clip(n // 2, 0, max(v.shape[1] - 1, 0))] if v.shape[1] else mean
        med = np.where(n > 0, med, np.nan)
        mode = np.where(mean < med, mean, 3 * med - 2 * mean)
        if it == maxiter:
            break
        with np.errstate(invalid='ignore'):
            bad = np.abs(v - mode[:, None]) > k * sd[:, None]
        if not bad.any():
            break
        v[bad] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        skew = np.nansum((v - mean[:, None])**3, axis=1) / np.maximum(n, 1) / sd**3
    return mode, sd, skew, n, n0 - n


def phot(img, x, y, ape, annulus, dannulus, epadu=1.0, datamax=None, ids=None, zmag=0.0, overlap='iraf'):
    """Forced photometry at (x, y) [1-based]; returns an astropy Table with IRAF phot's columns
    (ID XCENTER YCENTER XERR YERR MSKY STDEV SSKEW NSKY NSREJ SIER RAPERT SUM AREA FLUX MAG MERR PIER);
    MAG/MERR are NaN where IRAF writes INDEF."""
    img = np.asarray(img, dtype=float)
    x = np.asarray(x, float); y = np.asarray(y, float)
    rout = annulus + dannulus
    out = {k: np.full(len(x), np.nan) for k in ('MSKY', 'STDEV', 'SSKEW', 'SUM', 'AREA', 'FLUX', 'MAG', 'MERR')}
    for k in ('NSKY', 'NSREJ', 'SIER', 'PIER'):
        out[k] = np.zeros(len(x), int)
    for a in range(0, len(x), CHUNK):
        sl = slice(a, a + CHUNK)
        xs, ys = x[sl], y[sl]
        # sky annulus
        ix, iy, inside = _box(xs, ys, rout, img.shape)
        r2 = (ix - xs[:, None])**2 + (iy - ys[:, None])**2
        ann = (r2 >= annulus**2) & (r2 <= rout**2)
        val = _pixels(img, ix, iy)
        good = ann & inside & np.isfinite(val)
        if datamax is not None:
            good &= val <= datamax
        keep = ann.any(axis=0)
        v = np.where(good, val, np.nan)[:, keep]
        sky, sd, skew, nsky, nrej = sky_mode(v)
        sier = np.where(nsky == 0, 201, np.where((ann & ~inside).any(axis=1), 202, 0))
        del ix, iy, inside, r2, ann, val, good, v
        # aperture
        ix, iy, inside = _box(xs, ys, ape + 0.5, img.shape)
        if overlap == 'exact':
            w = circle_overlap(ix - xs[:, None], iy - ys[:, None], ape)
        else:
            w = np.clip(ape + 0.5 - np.hypot(ix - xs[:, None], iy - ys[:, None]), 0, 1)
        val = _pixels(img, ix, iy)
        badpix = (w > 0) & inside & (~np.isfinite(val) | ((val > datamax) if datamax is not None else False))
        partial = ((w > 0) & ~inside).any(axis=1)
        offimg = ~((w > 0) & inside).any(axis=1)
        wi = np.where(inside & np.isfinite(val), w, 0.0)
        area = wi.sum(axis=1)
        tot = np.sum(wi * np.where(np.isfinite(val), val, 0.0), axis=1)
        flux = tot - area * sky
        pier = np.where(offimg, 301, np.where(partial, 302, np.where(~np.isfinite(sky), 303,
                        np.where(badpix.any(axis=1), 305, 0))))
        with np.errstate(invalid='ignore', divide='ignore'):
            err = 1.0857 * np.sqrt(np.clip(flux, 0, None) / epadu + area * sd**2 + area**2 * sd**2 / nsky) / flux
            pos = flux > 0
            mag = np.where(pos, zmag - 2.5 * np.log10(np.where(pos, flux, 1.0)), np.nan)
            err = np.where(pos, err, np.nan)
        for k, val_ in (('MSKY', sky), ('STDEV', sd), ('SSKEW', skew), ('NSKY', nsky), ('NSREJ', nrej),
                        ('SIER', sier), ('SUM', tot), ('AREA', area), ('FLUX', flux), ('MAG', mag),
                        ('MERR', err), ('PIER', pier)):
            out[k][sl] = val_
    t = astropy.table.Table()
    t['ID'] = np.arange(1, len(x) + 1, dtype=np.int32) if ids is None else np.asarray(ids, dtype=np.int32)
    t['XCENTER'] = x
    t['YCENTER'] = y
    t['XERR'] = np.full(len(x), np.nan)     # no centering, as IRAF calgorithm=none (INDEF)
    t['YERR'] = np.full(len(x), np.nan)
    for k in ('MSKY', 'STDEV', 'SSKEW', 'NSKY', 'NSREJ', 'SIER'):
        t[k] = out[k]
    t['RAPERT'] = np.full(len(x), float(ape))
    for k in ('SUM', 'AREA', 'FLUX', 'MAG', 'MERR', 'PIER'):
        t[k] = out[k]
    return t
