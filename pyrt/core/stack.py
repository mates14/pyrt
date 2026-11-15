"""Robust coadd of reprojected frames: per-pixel weighted mean with outlier rejection.

Replaces Montage's mAdd (which has no clipping) after mProjectPP/pyrt-mproject: the
frames come as Montage writes them, each cut to its own overlap with the output grid
(its own NAXIS/CRPIX, same projection) with an *_area.fits companion.

  clipped_stack(frames, skel_header, output, ...)

frames: list of dicts {'image': projected (w*S*D), 'weight': projected (w*S) or None,
'w': scalar weight, 'zp': MAGZERO of the raw frame (None: no flux scaling for the test),
'gain': e-/ADU}.  The output keeps mAdd's meaning, the weighted mean of the unscaled
frames sum(W M D) / sum(W M) with W = area * w * S, now over the pixels M that pass:

  - each frame's sky (a coarse mesh median) is subtracted and its flux put on a common
    zeropoint FOR THE TEST ONLY: x_i = k_i (D_i - sky_i), k_i = 10^(0.4 (zp_ref - zp_i))
  - centre = per-pixel median of x over the frames covering the pixel
  - noise s_i^2 = k_i^2 (sky_rms_i^2 + max(D_i - sky_i, 0) / gain_i)  (the projected frames'
    own measured rms, so the reprojection's smoothing is in it)
  - reject x_i > centre + HI * s_i + AMPFRAC * |centre|  (cosmic rays, satellites, hot pixels)
    or     x_i < centre - LO * s_i - AMPFRAC * |centre|  (bad columns, dead pixels)
    AMPFRAC (SWarp's CLIP_AMPFRAC) keeps the cores of bright stars, which differ between
    frames with the seeing, out of the rejection;
  - a rejected pixel's neighbours go with it (GROW: the reprojection spreads a hit);
  - two passes, the second centred on the weighted mean of what survived the first.
Fewer than MIN_FRAMES frames on a pixel: no rejection there.  PYRT_STACK_DEBUG=x,y prints
the per-frame values at that output pixel.
"""

import os

import numpy as np
from astropy.io import fits

HI, LO = 4.0, 6.0
AMPFRAC = 0.3
GROW = 1
MIN_FRAMES = 5
MESH = 64          # [px] sky mesh
STRIP = 256        # [rows] processed at once


def _area_name(image):
    return image[:-5] + '_area.fits' if image.endswith('.fits') else image + '_area'


def _sky(data, mesh=MESH):
    """Coarse sky: median per mesh cell (NaN-aware), bilinear back to full size; and the
    robust rms of what is left."""
    from scipy.ndimage import zoom
    ny, nx = data.shape
    my, mx = max(1, ny // mesh), max(1, nx // mesh)
    cut = data[:my * mesh, :mx * mesh] if ny >= mesh and nx >= mesh else data
    if ny >= mesh and nx >= mesh:
        cells = cut.reshape(my, mesh, mx, mesh).swapaxes(1, 2).reshape(my, mx, -1)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                med = np.nanmedian(cells, axis=2)
        fill = np.nanmedian(med) if np.any(np.isfinite(med)) else 0.0
        med = np.where(np.isfinite(med), med, fill)
        sky = zoom(med, (ny / my, nx / mx), order=1, mode='nearest')[:ny, :nx]
        if sky.shape != data.shape:
            sky = np.pad(sky, ((0, ny - sky.shape[0]), (0, nx - sky.shape[1])), mode='edge')
    else:
        sky = np.full(data.shape, np.nanmedian(data))
    r = (data - sky)[np.isfinite(data)]
    rms = 1.4826 * np.median(np.abs(r - np.median(r))) if r.size else np.nan
    return sky.astype(np.float32), float(rms)


def _placement(hdr, skel):
    """Integer offset (dx, dy) of a Montage-cut frame in the output grid."""
    dx = skel['CRPIX1'] - hdr['CRPIX1']
    dy = skel['CRPIX2'] - hdr['CRPIX2']
    ix, iy = int(round(dx)), int(round(dy))
    if abs(dx - ix) > 0.01 or abs(dy - iy) > 0.01:
        raise ValueError(f"frame not on the output grid (offset {dx:.3f}, {dy:.3f})")
    return ix, iy


def clipped_stack(frames, skel, output, weight_output=None, reject_output=None, verbose=True):
    """frames: see the module doc; skel: astropy Header of the output grid.  Writes output
    (the coadd), optionally weight_output (sum of W over the kept pixels) and reject_output
    (rejected pixels per output pixel).  Returns a dict of statistics."""
    from scipy.ndimage import binary_dilation
    ny, nx = int(skel['NAXIS2']), int(skel['NAXIS1'])
    zps = [f['zp'] for f in frames if f.get('zp') is not None and np.isfinite(f['zp'])]
    zp_ref = float(np.median(zps)) if zps else None
    info = []
    for f in frames:
        h = fits.getheader(f['image'])
        ox, oy = _placement(h, skel)
        img = fits.getdata(f['image']).astype(np.float32)
        w_all = (fits.getdata(f['weight']).astype(np.float32) if f.get('weight') else
                 np.where(np.isfinite(img), np.float32(f['w']), np.float32(0)))
        try:
            area = fits.getdata(_area_name(f['image'])).astype(np.float32)
        except (OSError, FileNotFoundError):
            area = np.isfinite(img).astype(np.float32)
        with np.errstate(all='ignore'):
            D = np.where((w_all > 0) & np.isfinite(img), img / w_all, np.nan).astype(np.float32)
        sky, rms = _sky(D)
        k = 10 ** (0.4 * (zp_ref - f['zp'])) if zp_ref is not None and f.get('zp') is not None else 1.0
        W = np.nan_to_num(area * w_all, nan=0.0).astype(np.float32)
        W[~np.isfinite(D)] = 0
        info.append(dict(D=D, sky=sky, rms=rms, k=float(k), W=W, ox=ox, oy=oy,
                         gain=float(f.get('gain') or 1.0)))
        if verbose:
            print(f"stack: {f['image'].split('/')[-1]}: offset ({ox},{oy}) size {D.shape[1]}x{D.shape[0]}, "
                  f"sky rms {rms:.2f}, scale {k:.3f}")

    out = np.full((ny, nx), np.nan, np.float32)
    wsum = np.zeros((ny, nx), np.float32)
    nrej = np.zeros((ny, nx), np.int16)
    n = len(info)
    total_rej = np.zeros(n, int)
    for y0 in range(0, ny, STRIP):
        y1 = min(ny, y0 + STRIP)
        # strips overlap by GROW rows so that the dilation sees its neighbours
        ya, yb = max(0, y0 - GROW), min(ny, y1 + GROW)
        h = yb - ya
        X = np.full((n, h, nx), np.nan, np.float32)
        Dc = np.full((n, h, nx), np.nan, np.float32)
        Wc = np.zeros((n, h, nx), np.float32)
        S = np.full((n, h, nx), np.inf, np.float32)
        for i, f in enumerate(info):
            fy0, fy1 = ya - f['oy'], yb - f['oy']
            sy0, sy1 = max(0, fy0), min(f['D'].shape[0], fy1)
            if sy1 <= sy0:
                continue
            fx0, fx1 = max(0, -f['ox']), min(f['D'].shape[1], nx - f['ox'])
            if fx1 <= fx0:
                continue
            ty0, tx0 = sy0 - fy0, fx0 + f['ox']
            d = f['D'][sy0:sy1, fx0:fx1]
            s = f['sky'][sy0:sy1, fx0:fx1]
            sl = (i, slice(ty0, ty0 + (sy1 - sy0)), slice(tx0, tx0 + (fx1 - fx0)))
            Dc[sl] = d
            Wc[sl] = f['W'][sy0:sy1, fx0:fx1]
            X[sl] = f['k'] * (d - s)
            S[sl] = f['k'] * np.sqrt(f['rms'] ** 2 + np.clip(d - s, 0, None) / f['gain'])
        ok = np.isfinite(X) & (Wc > 0)
        ncov = ok.sum(axis=0)
        test = ncov >= MIN_FRAMES
        keep = ok.copy()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            centre = np.nanmedian(np.where(ok, X, np.nan), axis=0)
            for it in range(2):
                tol = AMPFRAC * np.abs(centre)
                bad = ok & test[None] & ((X > centre[None] + HI * S + tol[None]) |
                                         (X < centre[None] - LO * S - tol[None]))
                if GROW:
                    for i in range(n):
                        if bad[i].any():
                            bad[i] = binary_dilation(bad[i], iterations=GROW) & ok[i]
                keep = ok & ~bad
                if it == 0:
                    wk = np.where(keep, Wc, 0)
                    centre = np.where(wk.sum(axis=0) > 0,
                                      np.nansum(np.where(keep, X, 0) * wk, axis=0) / np.maximum(wk.sum(axis=0), 1e-30),
                                      centre)
        dbg = os.environ.get('PYRT_STACK_DEBUG')
        if dbg:
            px, py = map(int, dbg.split(','))
            if ya <= py < yb:
                r = py - ya
                print(f"stack debug ({px},{py}): ncov {ncov[r, px]} centre {centre[r, px]:.1f}")
                for i in range(n):
                    if ok[i, r, px]:
                        print(f"   frame {i:2d} x {X[i, r, px]:8.1f} s {S[i, r, px]:6.1f} D {Dc[i, r, px]:8.1f} "
                              f"W {Wc[i, r, px]:.3g} k {info[i]['k']:.3f} {'REJ' if not keep[i, r, px] else ''}")
        c0, c1 = y0 - ya, y0 - ya + (y1 - y0)
        wk = np.where(keep, Wc, 0)[:, c0:c1]
        ws = wk.sum(axis=0)
        num = np.nansum(np.where(keep[:, c0:c1], Dc[:, c0:c1], 0) * wk, axis=0)
        out[y0:y1] = np.where(ws > 0, num / np.maximum(ws, 1e-30), np.nan)
        wsum[y0:y1] = ws
        rej = (ok & ~keep)[:, c0:c1]
        nrej[y0:y1] = rej.sum(axis=0)
        total_rej += rej.reshape(n, -1).sum(axis=1)

    hdr = skel.copy()
    hdr['STACK'] = ('clipped', 'pyrt.core.stack: weighted mean with rejection')
    hdr['CLIPHI'] = (HI, 'rejection above centre [sigma]')
    hdr['CLIPLO'] = (LO, 'rejection below centre [sigma]')
    hdr['CLIPAMP'] = (AMPFRAC, 'tolerance as a fraction of the centre')
    hdr['NCLIPPED'] = (int(total_rej.sum()), 'frame pixels rejected')
    fits.writeto(output, out, hdr, overwrite=True)
    if weight_output:
        fits.writeto(weight_output, wsum, hdr, overwrite=True)
    if reject_output:
        fits.writeto(reject_output, nrej, hdr, overwrite=True)
    npix = np.array([np.isfinite(f['D']).sum() for f in info])
    frac = total_rej / np.maximum(npix, 1)
    if verbose:
        print(f"stack: {n} frames, {int(total_rej.sum())} pixels rejected "
              f"(per frame median {100*np.median(frac):.3f}%, max {100*np.max(frac):.3f}%)")
    return dict(n=n, rejected=total_rej, fraction=frac)
