"""Readouts summed into a frame (pyrt.utils.camera_config.readouts, phcat.get_accnum):
COADDS is the authority, written by the rts2ng Andor driver and by asarina wherever it sums
frames; the older keywords only without it."""
import astropy.io.fits as fits
import pytest

from pyrt.utils.camera_config import readouts


def H(**kw):
    h = fits.Header()
    for k, v in kw.items():
        h[k] = v
    return h


@pytest.mark.parametrize('hdr, n', [
    # rts2ng Andor driver since 2026-11-03
    (H(ACQMODE='Single Scan', ACCNUM=1, COADDS=1), 1),
    (H(ACQMODE='Accumulate', ACCNUM=20, COADDS=20), 20),
    (H(ACQMODE='Kinetic Series', ACCNUM=1, KINNUM=96, COADDS=96), 96),
    (H(ACQMODE='Kinetic Series', ACCNUM=2, KINNUM=96, COADDS=192), 192),
    # asarina, which copies the series header (COADDS=96) and must overwrite COADDS:
    # kinphot sub-frame of 4 scans, kinstep bin of 12 scans, framestep bin of 6 series frames
    (H(ACQMODE='Kinetic Series', ACCNUM=1, KINNUM=96, KINBIN=4, COADDS=4), 4),
    (H(ACQMODE='Accumulate', ACCNUM=12, KINNUM=96, KINBIN=12, COADDS=12), 12),
    (H(ACQMODE='Accumulate', ACCNUM=576, KINNUM=96, COADDS=576), 576),
    # SBT/FRAM bin of 6 single frames (no COADDS from the camera, asarina writes it)
    (H(ACQMODE='Accumulate', ACCNUM=6, COADDS=6), 6),
    # older frames, no COADDS
    (H(), 1),
    (H(ACQMODE='Single Scan', ACCNUM=1), 1),
    (H(ACQMODE='Accumulate', ACCNUM=20), 20),
    (H(ACQMODE='Kinetic Series', ACCNUM=1, KINNUM=96), 1),     # a 16-bit mean then
    (H(ACQMODE='Kinetic Series', ACCNUM=1, KINNUM=96, KINBIN=4), 4),
    (H(ACQMODE='3', ACCNUM=2), 2),
    # junk
    (H(COADDS=0, ACQMODE='Accumulate', ACCNUM=3), 3),
    (H(COADDS='x'), 1),
])
def test_readouts(hdr, n):
    assert readouts(hdr) == n


def test_get_accnum_reads_the_file(tmp_path):
    from pyrt.cli.phcat import get_accnum
    p = tmp_path / 'f.fits'
    fits.PrimaryHDU(header=H(ACQMODE='Kinetic Series', ACCNUM=1, KINNUM=96, COADDS=96)).writeto(p)
    assert get_accnum(str(p)) == 96
