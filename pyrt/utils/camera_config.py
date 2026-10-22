#!/usr/bin/env python3
"""
Per-camera parameters, resolved from the FITS header.

The camera is identified by pyrt.utils.chip_id (the same registry the
calibration library uses), the telescope by TELESCOP, the epoch by DATE-OBS.
Parameters come from INI files, read in this order (a later file wins within
the same section):

    pyrt/utils/cameras.ini          shipped with pyrt (calibration history in git)
    ~/.config/pyrt/cameras.ini      local additions and overrides

Section names, least to most specific:

    [DEFAULT]                       every camera
    [mi6166]                        the camera wherever it is mounted (detector)
    [mi6166/2026-10-02]             the camera from that date on
    [mi6166@SBT]                    the camera on that telescope (optics)
    [mi6166@SBT/2026-10-02]         ... from that date on

A section may name several cameras: [andor46,fli534@D50].  A camera the
registry does not know is named after CCD_NAME instead: [ccd:SROT1].  Epochs are
cumulative: all dated sections up to DATE-OBS apply in date order, so a
recalibration only lists what changed.  More specific sections win over less
specific ones, regardless of the file they come from.

Values the header may also carry (gain, ...) are taken by resolve():
command line > force.<key> > header > <key>.
"""

import os
import re
import logging
import configparser
from functools import lru_cache

from pyrt.utils.chip_id import get_camera_id, KNOWN_CAMERAS

PACKAGED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cameras.ini')
USER_FILE = '~/.config/pyrt/cameras.ini'

_SECTION_RE = re.compile(r'^(?P<cams>[^@/]+)(?:@(?P<tel>[^@/]+))?(?:/(?P<date>\d{4}-\d{2}-\d{2}))?$')


@lru_cache(maxsize=None)
def _load(files):
    """[(file, ConfigParser)] for the files that exist"""
    parsers = []
    for f in files:
        path = os.path.expanduser(f)
        if not os.path.exists(path):
            continue
        cp = configparser.ConfigParser(default_section=None, interpolation=None)
        cp.optionxform = str    # keep the case of FITS-like keys (PV2_3, CRPIX1)
        cp.read(path)
        parsers.append((path, cp))
    return parsers


def _matching_sections(section_names, camera, telescope, date):
    """Names of the sections that apply, least specific first"""
    found = []
    for name in section_names:
        if name == 'DEFAULT':
            found.append(((0, ''), name))
            continue
        m = _SECTION_RE.match(name)
        if m is None:
            logging.warning(f"camera_config: cannot parse section [{name}], ignored")
            continue
        if camera not in [c.strip() for c in m['cams'].split(',')]:
            continue
        tel = m['tel'].strip() if m['tel'] else None
        if tel is not None and tel != telescope:
            continue
        if m['date'] is not None and (date is None or m['date'] > date):
            continue
        level = 1 if tel is None else 2
        found.append(((level, m['date'] or ''), name))
    return [name for _, name in sorted(set(found))]


class CameraParams:
    """Merged parameters for one frame, with the origin of each value"""

    def __init__(self, camera, telescope, date, files=(PACKAGED_FILE, USER_FILE)):
        self.camera, self.telescope, self.date = camera, telescope, date
        self.values, self.source = {}, {}
        parsers = _load(tuple(files))
        names = {s for _, cp in parsers for s in cp.sections()}
        self.sections = _matching_sections(names, camera, telescope, date)
        for section in self.sections:
            for path, cp in parsers:
                if not cp.has_section(section):
                    continue
                for key, value in cp.items(section):
                    self.values[key] = value
                    self.source[key] = f"{os.path.basename(path)}[{section}]"

    @classmethod
    def from_header(cls, header, **kw):
        """Camera from CAMERA_ID (cat2det) or the registry, else ccd:CCD_NAME;
        telescope from TELESCOP, epoch from DATE-OBS"""
        camera = header.get('CAMERA_ID') or get_camera_id(header)
        ccd_name = str(header.get('CCD_NAME', '')).strip()
        if camera not in KNOWN_CAMERAS and ccd_name:
            camera = f"ccd:{ccd_name}"
        telescope = str(header.get('TELESCOP', '')).strip() or None
        date = str(header.get('DATE-OBS', ''))[:10] or None
        return cls(camera, telescope, date, **kw)

    def __contains__(self, key):
        return key in self.values

    def get(self, key, default=None, type=float):
        """Value converted by type, default when absent"""
        if key not in self.values:
            return default
        if type is bool:
            return self.values[key].strip().lower() in ('1', 'true', 'yes', 'on')
        return type(self.values[key])

    def resolve(self, key, header=None, header_key=None, cli=None, type=float):
        """(value, origin) for a value the header may carry as well:
        cli > force.<key> > header[header_key] > <key>; (None, None) if nowhere"""
        if cli is not None:
            return type(cli), 'cli'
        if f'force.{key}' in self.values:
            return self.get(f'force.{key}', type=type), self.source[f'force.{key}']
        if header is not None and header_key is not None and header.get(header_key) is not None:
            try:
                return type(header[header_key]), 'header'
            except (TypeError, ValueError):
                logging.warning(f"camera_config: unusable {header_key}={header[header_key]!r} in the header")
        if key in self.values:
            return self.get(key, type=type), self.source[key]
        return None, None

    @property
    def label(self):
        """Most specific section applied, for the output header (CAMCONF)"""
        specific = [s for s in self.sections if s != 'DEFAULT']
        return specific[-1] if specific else f"{self.camera}:none"

    def __repr__(self):
        return (f"CameraParams({self.camera}@{self.telescope} {self.date}: "
                f"{len(self.values)} values from {self.sections})")
