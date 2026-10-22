#!/usr/bin/env python3
"""
Camera Registry - single source of truth for camera identification.

This module identifies cameras from FITS headers, producing short,
human-readable IDs suitable for calibration file naming.

Naming convention:
    {manufacturer}{serial_digits}[v{version}]

Examples:
    andor46     - Andor iXon-Ultra, serial 10046
    mi2066      - Moravian Instruments G2, serial 002066
    mi6166      - Moravian Instruments G4, serial 06166
    mi2065      - Moravian G2, serial 002065, new electronics
    mi2065v1    - Moravian G2, serial 002065, old electronics
    fli534      - FLI IMG4710, serial suffix 258.534

Usage:
    from pyrt.utils.chip_id import get_camera_id, load_chip_id

    camera_id = get_camera_id(fits_header)
    # Returns e.g. "mi2066" or "unknown" if not recognized

CLI:
    pyrt-chip-id --list                     # list known cameras
    pyrt-chip-id -v *.fits                  # identify files
    find /images -name "*.fits" | pyrt-chip-id
"""

import sys
import argparse

# =============================================================================
# Camera Registry
# =============================================================================
# Format: (header_field, pattern_to_match, short_camera_id)
# Patterns are checked in order; first match wins.

CAMERA_PATTERNS = [
    # -------------------------------------------------------------------------
    # D50 Cameras (Ondrejov)
    # -------------------------------------------------------------------------

    # Andor iXon-Ultra (D50 C0)
    ("CCD_SER", "10046", "andor46"),
    ("CCD_TYPE", "ANDOR iXon-Ultra DU888_BV", "andor46"),

    # FLI IMG4710 (D50 C0) - serial 258.534, various header formats over years
    # NOTE: CCD_SER="CCD47-10" is ambiguous (both fli534 and fli785 have this chip)
    # so we rely on CCD_TYPE or the serial-specific CCD_SER patterns
    ("CCD_SER", "1--258.534", "fli534"),
    ("CCD_SER", "FLI_IMG4710_232304", "fli534"),  # sometimes in CCD_SER
    ("CCD_SER", "FLI_IMG4710", "fli534"),         # short version
    ("CCD_TYPE", "FLI_IMG4710_232304", "fli534"),
    ("CCD_TYPE", "FLI 258.534", "fli534"),
    ("CCD_TYPE", "FLI CCD47-10 rev.258.534", "fli534"),

    # Moravian C1-5000 with IMX250 (D50 C1) - config uses -p 50012
    ("CCD_SER", "C1MX05000-5001", "mi5001"),
    ("CCD_SER", "C1MX05000-050012", "mi5001"),    # possible full serial
    ("CCD_SER", "5001", "mi5001"),                 # gxccd numeric camera ID

    # Moravian G1-2000 (D50 C1, also old CWF1)
    ("CCD_SER", "G1SX2000-00315", "mi0315"),
    ("CCD_SER", "315", "mi0315"),                  # gxccd numeric camera ID

    # -------------------------------------------------------------------------
    # SBT Cameras (Ondrejov)
    # -------------------------------------------------------------------------

    # Moravian G4-16000 (SBT C1) - serial 06166
    ("CCD_SER", "G4KF16000-06166", "mi6166"),
    ("CCD_SER", "6166", "mi6166"),                 # gxccd numeric camera ID

    # Moravian G4-16000 (SBT C2) - serial 06167
    ("CCD_SER", "G4KF16000-06167", "mi6167"),
    ("CCD_SER", "6167", "mi6167"),                 # gxccd numeric camera ID

    # Moravian G2-1000BI (SBT C3) - full serial 002656
    ("CCD_SER", "G2EV1000-002656", "mi2656"),
    ("CCD_SER", "G2EV1000-00265", "mi2656"),       # truncated in some headers
    ("CCD_SER", "2656", "mi2656"),                 # gxccd numeric camera ID

    # Moravian G2-1600 - serial 002066
    ("CCD_SER", "G2KF1600-002066", "mi2066"),
    ("CCD_SER", "2066", "mi2066"),                 # gxccd numeric camera ID

    # Moravian G2-3200 MkII (SBT C3) - full serial 020008
    ("CCD_SER", "G2KF3202-020008", "mi0008"),
    ("CCD_SER", "G2KF3202-02000", "mi0008"),       # truncated in some headers
    ("CCD_SER", "20008", "mi0008"),                # gxccd numeric camera ID

    # -------------------------------------------------------------------------
    # CNF0/CWF1 Cameras (Ondrejov narrow/wide field)
    # -------------------------------------------------------------------------

    # FLI with CCD47-10 (CNF0) - serial 259.785
    ("CCD_SER", "FLI-2027-04", "fli785"),
    ("CCD_SER", "1--259.785", "fli785"),
    ("CCD_TYPE", "FLI 259.785", "fli785"),
    ("CCD_TYPE", "FLI CCD47-10 rev.259.785", "fli785"),
    ("CCD_TYPE", "FLI KAF1600 or similar array rev.259.785", "fli785"),

    # Moravian G2-1000BI (CNF0)
    ("CCD_SER", "G2EV01000-02596", "mi2596"),
    ("CCD_SER", "2596", "mi2596"),                 # gxccd numeric camera ID

    # Moravian G2-1600 (CNF0, CWF1) - serial 002065, new electronics
    ("CCD_SER", "G2KF1600-002065", "mi2065"),
    ("CCD_SER", "2065", "mi2065"),                 # gxccd numeric camera ID

    # Moravian G2-1600 (CWF1) - serial 902065, old electronics
    ("CCD_SER", "G2KF1600-902065", "mi2065v1"),
    ("CCD_SER", "902065", "mi2065v1"),             # gxccd numeric camera ID

    # -------------------------------------------------------------------------
    # BOOTES Cameras
    # -------------------------------------------------------------------------

    # Andor cameras at BOOTES
    ("CCD_SER", "1708", "andor1708"),
    ("CCD_SER", "2499", "andor2499"),
    ("CCD_SER", "3567", "andor3567"),

    # -------------------------------------------------------------------------
    # BOOTES accessory cameras. Their drivers often leave CCD_SER / CCD_TYPE
    # blank and record the identity in CCD_NAME / SERIAL / CHIP / DESCRIPTION
    # instead. These patterns come after the CCD_SER ones, so a populated serial
    # still wins (e.g. the SBT/CNF Moravians are matched by CCD_SER above).
    # -------------------------------------------------------------------------
    ("CCD_TYPE", "Alta KAF16803D7", "alta03d7"),
    # Apogee Alta (KAF-16803), CCD_SER blank/"0"
    ("CCD_NAME", "Alta", "alta03d7"),
    # Moravian G2-1600 (KAF-1603E) accessory; serial truncated to "G2KF1600-0"
    ("SERIAL", "G2KF1600-0", "mi1600"),
    ("DESCRIPTION", "G2-1600", "mi1600"),

    # -------------------------------------------------------------------------
    # Historical FLI cameras (no longer in use)
    # -------------------------------------------------------------------------

    # FLI IMG6303 with KAF-6303 (SBT C4)
    ("CCD_TYPE", "FLI_IMG6303_205004", "fli05004"),

    # FLI MaxCam CM2-1 with EEV CCD47-10 (BART 25cm primary focus)
    ("CCD_TYPE", "FLI_MaxCam_CM2-1_202704", "fli02704"),

    # FLI MaxCam CM8 cameras (3 units)
    ("CCD_TYPE", "FLI_MaxCam_CM8_212704", "fli12704"),
    ("CCD_TYPE", "FLI_MaxCam_CM8_224904", "fli24904"),    # ex Spain
    ("CCD_TYPE", "FLI_MaxCam_CM8_225004", "fli25004"),
]

# Compound patterns: require multiple fields to match
# Format: ({"field1": "value1", "field2": "value2"}, camera_id)
# These are checked BEFORE simple patterns to handle truncated serials
COMPOUND_PATTERNS = [
    # CCD47-10 chip is ambiguous - both fli534 and fli785 have it
    # Disambiguate by CCD_TYPE
    ({"CCD_SER": "CCD47-10", "CCD_TYPE": "FLI_IMG4710_232304"}, "fli534"),
    ({"CCD_SER": "CCD47-10", "CCD_TYPE": "FLI 258.534"}, "fli534"),
    ({"CCD_SER": "CCD47-10", "CCD_TYPE": "FLI 259.785"}, "fli785"),

    # G4-16000 with truncated serial - distinguish by INSTRUME
    ({"CCD_SER": "G4KF16000-0616", "INSTRUME": "SBT camera C1"}, "mi6166"),
    ({"CCD_SER": "G4KF16000-0616", "INSTRUME": "SBT camera C2"}, "mi6167"),

    # G2-1600 with truncated serial - truly ambiguous, use generic ID
    # (cameras are mobile, can't reliably distinguish by position)
]

# Fallback for truncated/ambiguous serials that can't be resolved
TRUNCATED_FALLBACKS = {
    "G4KF16000-0616": "mi616x",   # ambiguous G4-16000 (C1 or C2?)
    "G4KF16000-0620": "mi620x",   # G4-16000 serial 0620x (FRAM NF4, BART C0)
    "G2KF1600-00206": "mi0206",   # ambiguous G2-1600 (mi2065 or mi2066?)
    "CCD47-10": "fli47x",         # ambiguous FLI with CCD47-10 (fli534 or fli785?)
}

# Set of all valid camera IDs (auto-generated from patterns + fallbacks)
KNOWN_CAMERAS = (
    {pat[2] for pat in CAMERA_PATTERNS} |
    {pat[1] for pat in COMPOUND_PATTERNS} |
    set(TRUNCATED_FALLBACKS.values())
)

# Human-readable camera descriptions
CAMERA_DESCRIPTIONS = {
    # D50
    "andor46": "Andor iXon-Ultra DU888 (D50 C0)",
    "fli534": "FLI IMG4710 EEV CCD47-10 (D50 C0)",
    "mi5001": "Moravian C1-5000 IMX250 (D50 C1)",
    "mi0315": "Moravian G1-2000 Sony ICX274 (Makak zenith camera)",
    # SBT
    "mi6166": "Moravian G4-16000 Kodak KAF-16803 (SBT C1)",
    "mi6167": "Moravian G4-16000 Kodak KAF-16803 (SBT C2)",
    "mi2656": "Moravian G2-1000BI EEV CCD47-10 (now Perek-2m, ex SBT)",
    "mi2066": "Moravian G2-1600 Kodak KAF-1603 (backup, mobile)",
    "mi0008": "Moravian G2-3200 MkII Kodak KAF-3200 (SBT C3, ex Perek-2m)",
    # CNF0/CWF1 (BART)
    "fli785": "FLI MAXcam EEV CCD47-10 (BART/CNF0)",
    "mi2596": "Moravian G2-1000BI EEV (CNF0)",
    "mi2065": "Moravian G2-1600 Kodak KAF-1603 new elec (backup, mobile)",
    "mi2065v1": "Moravian G2-1600 Kodak KAF-1603 old elec (backup, mobile)",
    # BOOTES
    "andor1708": "Andor (BOOTES)",
    "andor2499": "Andor (BOOTES)",
    "andor3567": "Andor (BOOTES, also COLORES spectrograph)",
    "alta03d7": "Apogee Alta KAF-16803D7 (BOOTES accessory)",
    "mi1600": "Moravian G2-1600 KAF-1603E (BOOTES accessory, truncated serial)",
    # Generic family fallbacks (CCD_NAME only, specific unit unknown)
    "andor": "Andor (family; specific serial not recorded)",
    "mi": "Moravian Instruments (family; specific serial not recorded)",
    "alta": "Apogee Alta (family; specific serial not recorded)",
    "fli": "FLI (family; specific serial not recorded)",
    # Ambiguous (couldn't resolve)
    "mi616x": "Moravian G4-16000 (SBT, truncated serial - C1 or C2?)",
    "mi620x": "Moravian G4-16000 (FRAM NF4, also BART C0; truncated serial 0620x)",
    "mi0206": "Moravian G2-1600 (truncated serial - mi2065 or mi2066?)",
    "fli47x": "FLI with CCD47-10 (ambiguous - fli534 or fli785?)",
    # Historical FLI cameras (no longer in use)
    "fli05004": "FLI IMG6303 KAF-6303 (SBT C4, historical)",
    "fli02704": "FLI MaxCam CM2-1 EEV CCD47-10 (BART 25cm, historical)",
    "fli12704": "FLI MaxCam CM8 (historical)",
    "fli24904": "FLI MaxCam CM8 (ex Spain, historical)",
    "fli25004": "FLI MaxCam CM8 (historical)",
}


# =============================================================================
# Lookup Functions
# =============================================================================

def _get_header_value(header, field) -> str:
    """Get header value as stripped string, or None."""
    try:
        value = header.get(field)
        if value is None:
            return None
        return str(value).strip()
    except (KeyError, TypeError):
        return None


def get_camera_id(header) -> str:
    """
    Get short camera ID from FITS header.

    Args:
        header: FITS header (dict-like, e.g. astropy.io.fits header)

    Returns:
        Short camera ID string (e.g. "mi2066", "andor46")
        Returns "unknown" if camera not recognized.
    """
    # 1. Check compound patterns first (for truncated serials with disambiguating info)
    for conditions, camera_id in COMPOUND_PATTERNS:
        all_match = True
        for field, pattern in conditions.items():
            value = _get_header_value(header, field)
            if value != pattern:
                all_match = False
                break
        if all_match:
            return camera_id

    # 2. Check simple patterns
    for field, pattern, camera_id in CAMERA_PATTERNS:
        value = _get_header_value(header, field)
        if value == pattern:
            return camera_id

    # 3. Check truncated serial fallbacks
    ccd_ser = _get_header_value(header, "CCD_SER")
    if ccd_ser and ccd_ser in TRUNCATED_FALLBACKS:
        return TRUNCATED_FALLBACKS[ccd_ser]

    # 4. Family fallback from CCD_NAME when the specific unit can't be pinned
    #    (e.g. blank serial). Keeps camera families apart instead of collapsing
    #    everything to 'unknown'; never claims a specific serial it doesn't have.
    fam = _family_from_ccd_name(header)
    if fam:
        return fam

    return "unknown"


def _family_from_ccd_name(header) -> str:
    """Generic camera family ('andor'/'mi'/'alta'/'fli') from CCD_NAME, or None."""
    name = _get_header_value(header, "CCD_NAME")
    if not name:
        return None
    n = name.strip().lower()
    for fam in ("andor", "alta", "fli", "mi"):
        if n.startswith(fam):
            return fam
    return None


# Legacy alias for compatibility
load_chip_id = get_camera_id


def get_camera_id_verbose(header) -> tuple:
    """
    Get camera ID with diagnostic info.

    Returns:
        (camera_id, matched_field, matched_value) or
        ("unknown", None, header_dump) if not recognized
    """
    # 1. Check compound patterns first
    for conditions, camera_id in COMPOUND_PATTERNS:
        all_match = True
        matched_info = []
        for field, pattern in conditions.items():
            value = _get_header_value(header, field)
            if value != pattern:
                all_match = False
                break
            matched_info.append(f"{field}={value}")
        if all_match:
            return (camera_id, "compound", " + ".join(matched_info))

    # 2. Check simple patterns
    for field, pattern, camera_id in CAMERA_PATTERNS:
        value = _get_header_value(header, field)
        if value == pattern:
            return (camera_id, field, value)

    # 3. Check truncated serial fallbacks
    ccd_ser = _get_header_value(header, "CCD_SER")
    if ccd_ser and ccd_ser in TRUNCATED_FALLBACKS:
        return (TRUNCATED_FALLBACKS[ccd_ser], "CCD_SER (fallback)", ccd_ser)

    # 4. Family fallback from CCD_NAME
    fam = _family_from_ccd_name(header)
    if fam:
        return (fam, "CCD_NAME (family)", _get_header_value(header, "CCD_NAME"))

    # Not found - dump relevant fields for debugging
    debug_info = {}
    for field in ("CCD_SER", "CCD_TYPE", "CCD_NAME", "SERIAL", "CHIP",
                  "DESCRIPTION", "CCD_CHIP", "INSTRUME", "DETECTOR"):
        try:
            val = header.get(field)
            if val is not None:
                debug_info[field] = str(val).strip()
        except (KeyError, TypeError):
            pass

    return ("unknown", None, debug_info)


def list_cameras() -> list:
    """Return sorted list of all known camera IDs."""
    return sorted(KNOWN_CAMERAS)


def camera_info(camera_id: str) -> dict:
    """
    Get info about a camera by its short ID.

    Returns dict with matching patterns and description, or empty dict if unknown.
    """
    if camera_id not in KNOWN_CAMERAS:
        return {}

    # Collect patterns from simple patterns
    patterns = [(f, p) for f, p, cid in CAMERA_PATTERNS if cid == camera_id]

    # Collect from compound patterns
    for conditions, cid in COMPOUND_PATTERNS:
        if cid == camera_id:
            for f, p in conditions.items():
                patterns.append((f, p))

    # Collect from fallbacks
    for ser, cid in TRUNCATED_FALLBACKS.items():
        if cid == camera_id:
            patterns.append(("CCD_SER", ser + " (truncated)"))

    return {
        "id": camera_id,
        "description": CAMERA_DESCRIPTIONS.get(camera_id, ""),
        "patterns": patterns,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Camera registry lookup tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                     # List all known cameras
  %(prog)s --info mi2066              # Show patterns for a camera
  %(prog)s -v *.fits                  # Identify files verbosely
  find . -name "*.fits" | %(prog)s    # Read files from stdin
        """
    )
    parser.add_argument("files", nargs="*", help="FITS files to identify")
    parser.add_argument("--list", action="store_true", help="List all known cameras")
    parser.add_argument("--info", type=str, metavar="ID", help="Show info for a camera ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.list:
        print("Known cameras:")
        for cam in list_cameras():
            info = camera_info(cam)
            desc = info.get("description", "")
            if desc:
                print(f"  {cam:12s} - {desc}")
            else:
                patterns = ", ".join(f"{f}={p}" for f, p in info["patterns"][:2])
                print(f"  {cam:12s} ({patterns})")
        return 0

    if args.info:
        info = camera_info(args.info)
        if info:
            print(f"Camera: {info['id']}")
            if info.get("description"):
                print(f"Description: {info['description']}")
            print("Matching patterns:")
            for field, pattern in info["patterns"]:
                print(f"  {field} = '{pattern}'")
        else:
            print(f"Unknown camera: {args.info}")
            return 1
        return 0

    # Collect files from args and stdin
    files = list(args.files)
    if not sys.stdin.isatty():
        files.extend(line.strip() for line in sys.stdin if line.strip())

    if not files:
        parser.print_help()
        return 1

    try:
        from astropy.io import fits
    except ImportError:
        print("Error: astropy required. Install with: pip install astropy", file=sys.stderr)
        return 1

    for filepath in files:
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                if args.verbose:
                    camera_id, field, value = get_camera_id_verbose(header)
                    if field:
                        print(f"{filepath}\t{camera_id}\t({field}={value})")
                    else:
                        print(f"{filepath}\t{camera_id}\t{value}")
                else:
                    print(f"{filepath}\t{get_camera_id(header)}")
        except Exception as e:
            print(f"{filepath}\tERROR: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
