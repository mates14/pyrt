"""Robust coadd of reprojected frames: per-pixel weighted mean with outlier rejection.

Replaces Montage's mAdd (which has no clipping) after mProjectPP/pyrt-mproject: the
frames come as Montage writes them, each cut to its own overlap with the output grid
(its own NAXIS/CRPIX, same projection) with an *_area.fits companion.

  clipped_stack(frames, skel_header, output, ...)

frames: list of dicts {'image': projected (w*S*D), 'weight': projected (w*S) or None,
'w': scalar weight, 'zp': MAGZERO of the raw frame (None: no flux scaling for the test),
'gain': e-/ADU}.  The output keeps mAdd's meaning, the weighted mean of the unscaled
frames with W = area * w * S, now over the pixels M that pass:
  sum(W M (D - sky)) / sum(W M) + sum(W sky) / sum(W)
(the sky term over all frames: a rejection must not step the level by that frame's sky offset;
with nothing rejected this is exactly mAdd's sum(W D) / sum(W)).  The test:

  - each frame's sky (a coarse mesh median) is subtracted and its flux put on a common
    zeropoint FOR THE TEST ONLY: x_i = k_i (D_i - sky_i), k_i = 10^(0.4 (zp_ref - zp_i))
  - centre = per-pixel median of x over the frames covering the pixel
  - noise s_i^2 = k_i^2 (sky_rms_i^2 + max(D_i - sky_i, 0) / gain_i)  (the projected frames'
    own measured rms, so the reprojection's smoothing is in it)
  - reject x_i > centre + HI * s_i + AMPFRAC * |centre|  (cosmic rays, satellites, hot pixels)
    or     x_i < centre - LO * s_i - AMPFRAC * |centre|  (bad columns, dead pixels)
    AMPFRAC (SWarp's CLIP_AMPFRAC, here 1.0: at 0.3 SBT bright stars lost 1-3% to core
    clipping) keeps the cores of bright stars, which differ between
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
AMPFRAC = 1.0       # SWarp uses 0.3; SBT's undersampled stars (3" px) vary more than that in their cores
GROW = 1
MIN_FRAMES = 5
MESH = 64          # [px] sky mesh
STRIP = 256        # [rows] processed at once


def _area_name(image):
    return image[:-5] + '_area.fits' if image.endswith('.fits') else image + '_area'


def _sky(data, mesh=MESH):
    """Coarse sky: the median per mesh cell (NaN-aware) as a small grid, and the robust rms of
    the frame around it.  _sky_at() interpolates the grid where needed (no full-size sky kept:
    an SBT frame is 4k x 4k)."""
    import warnings
    ny, nx = data.shape
    my, mx = max(1, ny // mesh), max(1, nx // mesh)
    if ny >= mesh and nx >= mesh:
        cells = data[:my * mesh, :mx * mesh].reshape(my, mesh, mx, mesh).swapaxes(1, 2).reshape(my, mx, -1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med = np.nanmedian(cells, axis=2)
        fill = np.nanmedian(med) if np.any(np.isfinite(med)) else 0.0
        med = np.where(np.isfinite(med), med, fill)
    else:
        med = np.full((1, 1), np.nanmedian(data))
    sky = _sky_at(med, mesh, 0, ny, 0, nx)
    r = (data - sky)[np.isfinite(data)]
    rms = 1.4826 * np.median(np.abs(r - np.median(r))) if r.size else np.nan
    return med.astype(np.float32), float(rms)


def _sky_at(med, mesh, y0, y1, x0, x1):
    """The sky grid interpolated (bilinear, cell centres, flat beyond the edge) on rows y0:y1,
    columns x0:x1 of the frame; separable, so cheap for a 4k-wide strip."""
    def axis(i0, i1, n):
        g = np.clip((np.arange(i0, i1) + 0.5) / mesh - 0.5, 0, n - 1)
        lo = np.floor(g).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        return lo, hi, (g - lo).astype(np.float32)
    ylo, yhi, fy = axis(y0, y1, med.shape[0])
    xlo, xhi, fx = axis(x0, x1, med.shape[1])
    rows = med[ylo] * (1 - fy)[:, None] + med[yhi] * fy[:, None]
    return (rows[:, xlo] * (1 - fx)[None] + rows[:, xhi] * fx[None]).astype(np.float32)


def _frame_rows(f, r0, r1):
    """Rows r0:r1 of a projected frame: the unweighted data D (NaN where not covered) and its
    weight W = area * w * S (0 there)."""
    img = np.asarray(f['hi'][0].section[r0:r1, :], dtype=np.float32)
    w_all = (np.asarray(f['hw'][0].section[r0:r1, :], dtype=np.float32) if f['hw'] is not None else
             np.where(np.isfinite(img), np.float32(f['w']), np.float32(0)))
    area = (np.asarray(f['ha'][0].section[r0:r1, :], dtype=np.float32) if f['ha'] is not None else
            np.isfinite(img).astype(np.float32))
    with np.errstate(all='ignore'):
        D = np.where((w_all > 0) & np.isfinite(img), img / w_all, np.nan).astype(np.float32)
    W = np.nan_to_num(area * w_all, nan=0.0).astype(np.float32)
    W[~np.isfinite(D)] = 0
    return D, W


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
        # rows are read with .section (seeks, no mapping: 25 SBT frames would otherwise sit in the
        # process as mapped pages)
        hi = fits.open(f['image'], memmap=False, lazy_load_hdus=True)
        hw = fits.open(f['weight'], memmap=False, lazy_load_hdus=True) if f.get('weight') else None
        try:
            ha = fits.open(_area_name(f['image']), memmap=False, lazy_load_hdus=True)
        except (OSError, FileNotFoundError):
            ha = None
        fr = dict(hi=hi, hw=hw, ha=ha, w=float(f['w']), ox=ox, oy=oy,
                  shape=(int(hi[0].header['NAXIS2']), int(hi[0].header['NAXIS1'])),
                  gain=float(f.get('gain') or 1.0))
        D, _ = _frame_rows(fr, 0, fr['shape'][0])
        fr['med'], fr['rms'] = _sky(D)
        fr['npix'] = int(np.isfinite(D).sum())
        del D
        fr['k'] = float(10 ** (0.4 * (zp_ref - f['zp']))) if zp_ref is not None and f.get('zp') is not None else 1.0
        info.append(fr)
        if verbose:
            print(f"stack: {f['image'].split('/')[-1]}: offset ({ox},{oy}) size {fr['shape'][1]}x{fr['shape'][0]}, "
                  f"sky rms {fr['rms']:.2f}, scale {fr['k']:.3f}")

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
        Sk = np.zeros((n, h, nx), np.float32)
        Wc = np.zeros((n, h, nx), np.float32)
        S = np.full((n, h, nx), np.inf, np.float32)
        for i, f in enumerate(info):
            fy0, fy1 = ya - f['oy'], yb - f['oy']
            sy0, sy1 = max(0, fy0), min(f['shape'][0], fy1)
            if sy1 <= sy0:
                continue
            fx0, fx1 = max(0, -f['ox']), min(f['shape'][1], nx - f['ox'])
            if fx1 <= fx0:
                continue
            ty0, tx0 = sy0 - fy0, fx0 + f['ox']
            d, wf = _frame_rows(f, sy0, sy1)
            d, wf = d[:, fx0:fx1], wf[:, fx0:fx1]
            s = _sky_at(f['med'], MESH, sy0, sy1, fx0, fx1)
            sl = (i, slice(ty0, ty0 + (sy1 - sy0)), slice(tx0, tx0 + (fx1 - fx0)))
            Dc[sl] = d
            Sk[sl] = s
            Wc[sl] = wf
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
        okc, keepc = ok[:, c0:c1], keep[:, c0:c1]
        # every frame rejected (a saturated core): keep them all there rather than leave a hole
        allrej = (okc.sum(axis=0) > 0) & (keepc.sum(axis=0) == 0)
        keepc = np.where(allrej[None], okc, keepc)
        Wcc, Dcc, Skc = Wc[:, c0:c1], Dc[:, c0:c1], Sk[:, c0:c1]
        wk = np.where(keepc, Wcc, 0)
        ws = wk.sum(axis=0)
        wa = np.where(okc, Wcc, 0)
        wsa = wa.sum(axis=0)
        # the mean of the sky-subtracted frames over what passed, plus the mean sky of ALL the
        # frames: dropping a frame must not step the level by its sky's offset (frames differ
        # in sky by hundreds of counts; a step over a grown 3x3 patch is noise in an aperture)
        num = np.nansum(np.where(keepc, Dcc - Skc, 0) * wk, axis=0)
        skym = np.nansum(np.where(okc, Skc, 0) * wa, axis=0) / np.maximum(wsa, 1e-30)
        out[y0:y1] = np.where(ws > 0, num / np.maximum(ws, 1e-30) + skym, np.nan)
        wsum[y0:y1] = ws
        rej = (okc & ~keepc)
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
    npix = np.array([f['npix'] for f in info])
    for f in info:
        for k in ('hi', 'hw', 'ha'):
            if f[k] is not None:
                f[k].close()
    frac = total_rej / np.maximum(npix, 1)
    if verbose:
        print(f"stack: {n} frames, {int(total_rej.sum())} pixels rejected "
              f"(per frame median {100*np.median(frac):.3f}%, max {100*np.max(frac):.3f}%)")
    return dict(n=n, rejected=total_rej, fraction=frac)
