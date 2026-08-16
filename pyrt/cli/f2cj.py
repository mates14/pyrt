#!/usr/bin/python3

import os
import numpy as np
import sys
import argparse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from PIL import Image, ImageDraw, ImageFont

def log_function_with_fixed_c(x, A, B, C_fixed):
    return A + B * np.log10(x - C_fixed)

def apply_color_palette(data, palette='none', inverted=False):
    """Apply color palette to grayscale data."""
    if palette == 'none':
        # Grayscale mode
        if inverted:
            data = 255 - data
        return data
    elif palette == 'heat':
        # Heat palette: RGB channels rise at different rates
        # Red: rises at double rate, reaches 255 at middle (128)
        # Green: rises steadily from 0 to 255
        # Blue: starts at middle, rises from 0 to 255
        #red = np.clip(data*3/2, 0, 255).astype(np.uint8)
        blue = np.clip(data.astype(np.uint8)*3//2,0,255)
        green = data.astype(np.uint8)
        red = np.clip(data.astype(np.uint8)*3//2-127,0,255) # np.clip(data*3/2-127, 0, 255).astype(np.uint8)
        
#        if inverted:  # Cool palette is inverted heat
#            red = 255 - red
#            green = 255 - green
#            blue = 255 - blue
        
        # Stack RGB channels
        return np.stack([red, green, blue], axis=-1)

def load_font(size):
    """Load the best available TTF font, falling back to PIL's built-in."""
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

def expand_label(label_text, fits_header):
    """Expand FITS header values in label text."""
    if not label_text:
        return ""
    
    import re
    from datetime import datetime
    
    expanded = label_text
    
    # Handle %H:%M expansion from DATE-OBS
    if '%H:%M' in expanded:
        try:
            if 'DATE-OBS' in fits_header:
                date_obs = str(fits_header['DATE-OBS'])
                # Parse ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS
                if 'T' in date_obs:
                    dt = datetime.fromisoformat(date_obs.replace('T', ' ').split('.')[0])
                else:
                    dt = datetime.fromisoformat(date_obs.split('.')[0])
                expanded = expanded.replace('%H:%M', dt.strftime('%H:%M'))
            else:
                expanded = expanded.replace('%H:%M', '??:??')
        except:
            expanded = expanded.replace('%H:%M', '??:??')
    
    # Handle other common expansions
    expanded = re.sub(r'%([A-Z0-9_-]+)', lambda m: str(fits_header.get(m.group(1), '?')), expanded)
    
    return expanded

def add_label_to_image(img, label_text, fits_header):
    """Add text label at bottom of image."""
    if not label_text:
        return img
    
    # Expand FITS header values in label
    expanded_label = expand_label(label_text, fits_header)
    
    draw = ImageDraw.Draw(img)
    font = load_font(24)

    # Get image dimensions and text size
    img_width, img_height = img.size
    bbox = draw.textbbox((0, 0), expanded_label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position text at bottom center
    x = (img_width - text_width) // 2
    y = img_height - text_height - 10
    
    # Draw text with black outline for visibility
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x+dx, y+dy), expanded_label, font=font, fill='black')
    draw.text((x, y), expanded_label, font=font, fill='white')
    
    return img

def create_fallback_image(width=800, height=600, error_message="Error processing FITS file"):
    """Create a white fallback image with error message."""
    try:
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        font = load_font(24)

        # Center the error message
        bbox = draw.textbbox((0, 0), error_message, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), error_message, font=font, fill='red')
        return img
    except:
        # Absolute fallback - create minimal white image
        return Image.new('RGB', (width, height), 'white')

def safe_apply_color_palette(data, palette='none', inverted=False):
    """Safe version of apply_color_palette that handles errors."""
    try:
        return apply_color_palette(data, palette, inverted)
    except Exception as e:
        print(f"Warning: Color palette application failed: {e}")
        # Return grayscale as fallback
        if inverted:
            data = 255 - data
        return data

def safe_add_label_to_image(img, label_text, fits_header):
    """Safe version of add_label_to_image that handles errors."""
    try:
        return add_label_to_image(img, label_text, fits_header)
    except Exception as e:
        print(f"Warning: Label addition failed: {e}")
        return img

def decode_catalog_name(name):
    """Decode literal '\\uXXXX' escapes (e.g. greek letters) in catalog name strings."""
    name = name.strip()
    if '\\u' in name:
        try:
            return name.encode('ascii').decode('unicode_escape')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return name
    return name

def field_center_and_radius(wcs, img_width, img_height):
    """Sky center and a matching radius (deg) with margin, for pre-filtering catalogs."""
    center = wcs.pixel_to_world(img_width / 2.0, img_height / 2.0)
    corner = wcs.pixel_to_world(0, 0)
    # Margin beyond the corner distance so objects just outside the frame
    # (whose center falls inside once radius/label extent is drawn) still match.
    radius = center.separation(corner).deg * 1.2
    return center, radius

def match_catalog_to_frame(wcs, center, field_radius, img_width, img_height, ra, dec):
    """Return (index, x, y) for catalog rows within field_radius whose projected
    pixel position also falls inside the image bounds."""
    sep = center.separation(SkyCoord(ra * u.deg, dec * u.deg))
    idx = np.nonzero(sep.deg < field_radius)[0]
    if not len(idx):
        return []
    xs, ys = wcs.all_world2pix(ra[idx], dec[idx], 0)
    out = []
    for k, i in enumerate(idx):
        x, y = float(xs[k]), float(ys[k])
        if 0 <= x < img_width and 0 <= y < img_height:
            out.append((i, x, y))
    return out

def get_sky_annotations(header, img_width, img_height, catalog_dir, star_maglimit=None,
                         gcvs_maglimit=None):
    """Match NGC/IC objects, named bright stars and GCVS variable stars against
    the image's WCS.

    catalog_dir is expected to contain astrometry.net's ngc2000.fits,
    ngc2000names.fits and brightstars.fits, plus an optional gcvs.fits
    (see fetch_gcvs.py). Uses astropy's WCS (wcslib) directly rather than
    astrometry.net's own anwcs/plotstuff, since the latter doesn't handle
    our ZPN-projected WCS solutions.

    Returns a list of dicts: {kind, x, y, radius_px, labels}.
    """
    wcs = WCS(header)
    if not wcs.has_celestial:
        raise ValueError('FITS header has no celestial WCS')

    center, field_radius = field_center_and_radius(wcs, img_width, img_height)
    pixscale_arcsec = np.mean(proj_plane_pixel_scales(wcs)) * 3600.0

    annotations = []

    catalog_dir = os.path.expanduser(catalog_dir)
    ngc_path = os.path.join(catalog_dir, 'ngc2000.fits')
    names_path = os.path.join(catalog_dir, 'ngc2000names.fits')
    bright_path = os.path.join(catalog_dir, 'brightstars.fits')
    gcvs_path = os.path.join(catalog_dir, 'gcvs.fits')

    if not any(os.path.exists(p) for p in (ngc_path, bright_path, gcvs_path)):
        raise FileNotFoundError(
            f"No catalog files found under '{catalog_dir}' "
            f"(expected ngc2000.fits, brightstars.fits and/or gcvs.fits)")

    if os.path.exists(ngc_path):
        ngc = fits.getdata(ngc_path, 1)

        namemap = {}
        if os.path.exists(names_path):
            names = fits.getdata(names_path, 1)
            for obj, nm in zip(names['Object'], names['Name']):
                nm = nm.strip()
                if not nm:
                    continue
                isic = nm.startswith('I')
                try:
                    num = int(nm.replace('I', '').strip())
                except ValueError:
                    continue
                namemap.setdefault((isic, num), []).append(obj.strip())

        for i, x, y in match_catalog_to_frame(wcs, center, field_radius, img_width, img_height,
                                              ngc['ra'], ngc['dec']):
            designation = ngc['name'][i].strip()
            isic = designation.startswith('IC')
            num = int(ngc['ngcnum'][i])
            labels = [designation] + namemap.get((isic, num), [])
            radius_px = float(ngc['radius'][i]) * 3600.0 / pixscale_arcsec
            annotations.append(dict(kind='ngc', x=x, y=y,
                                    radius_px=radius_px, labels=labels))

    if os.path.exists(bright_path):
        bright = fits.getdata(bright_path, 1)
        for i, x, y in match_catalog_to_frame(wcs, center, field_radius, img_width, img_height,
                                              bright['ra'], bright['dec']):
            if star_maglimit is not None and bright['vmag'][i] > star_maglimit:
                continue
            labels = [decode_catalog_name(n) for n in (bright['name1'][i], bright['name2'][i])
                     if n.strip()]
            if not labels:
                # Bright-star catalog is name-only; skip unnamed entries.
                continue
            annotations.append(dict(kind='bright', x=x, y=y,
                                    radius_px=0.0, labels=labels))

    if os.path.exists(gcvs_path):
        gcvs = fits.getdata(gcvs_path, 1)
        for i, x, y in match_catalog_to_frame(wcs, center, field_radius, img_width, img_height,
                                              gcvs['ra'], gcvs['dec']):
            magmax = float(gcvs['magmax'][i])
            if gcvs_maglimit is not None and (magmax < 0 or magmax > gcvs_maglimit):
                continue
            vartype = gcvs['vartype'][i].strip()
            label = gcvs['name'][i].strip()
            if vartype:
                label += f' ({vartype})'
            annotations.append(dict(kind='gcvs', x=x, y=y,
                                    radius_px=0.0, labels=[label]))

    return annotations

def draw_sky_annotations(img, annotations, colors, fontsize=14):
    """Draw catalog circles/markers with labels onto img.

    colors maps annotation 'kind' -> PIL color; kinds without an entry
    fall back to colors['default'].
    """
    if not annotations:
        return img
    if img.mode == 'L':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    font = load_font(int(fontsize))
    for ann in annotations:
        x, y, r = ann['x'], ann['y'], ann['radius_px']
        label = ' / '.join(ann['labels'])
        color = colors.get(ann['kind'], colors['default'])
        if r >= 3:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
            ty = y + r + 2
        else:
            m = 5
            draw.line([x - m, y, x + m, y], fill=color, width=1)
            draw.line([x, y - m, x, y + m], fill=color, width=1)
            ty = y + m + 2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x + dx, ty + dy), label, font=font, fill='black')
        draw.text((x, ty), label, font=font, fill=color)
    return img

def get_simbad_annotations(header, img_width, img_height, maglimit=12.0):
    """Live SIMBAD cone-search for named objects in the field, filtered to
    maglimit to avoid drowning the image in faint catalog-only entries.

    Returns a list of dicts in the same shape as get_sky_annotations().
    """
    from astroquery.simbad import Simbad

    wcs = WCS(header)
    if not wcs.has_celestial:
        raise ValueError('FITS header has no celestial WCS')
    center, field_radius = field_center_and_radius(wcs, img_width, img_height)

    simbad = Simbad()
    simbad.add_votable_fields('otype', 'V')
    result = simbad.query_region(center, radius=field_radius * u.deg)
    if result is None or len(result) == 0:
        return []

    annotations = []
    for row in result:
        v = row['V']
        if hasattr(v, 'mask') and v.mask:
            continue
        v = float(v)
        if maglimit is not None and v > maglimit:
            continue
        sc = SkyCoord(row['ra'] * u.deg, row['dec'] * u.deg)
        x, y = wcs.all_world2pix(sc.ra.deg, sc.dec.deg, 0)
        x, y = float(x), float(y)
        if not (0 <= x < img_width and 0 <= y < img_height):
            continue
        label = str(row['main_id'])
        annotations.append(dict(kind='simbad', x=x, y=y, radius_px=0.0, labels=[label]))
    return annotations

def safe_annotate_sky(img, header, catalog_dir=None, star_maglimit=None, gcvs_maglimit=None,
                       colors=None, fontsize=14, do_simbad=False, simbad_maglimit=12.0):
    """Safe wrapper: annotate img with sky catalog objects, never raising."""
    anns = []
    if catalog_dir:
        try:
            local = get_sky_annotations(header, img.width, img.height, catalog_dir,
                                        star_maglimit=star_maglimit, gcvs_maglimit=gcvs_maglimit)
            print(f"Annotated {len(local)} object(s) from {catalog_dir}")
            anns += local
        except Exception as e:
            print(f"Warning: local catalog annotation failed: {e}")
    if do_simbad:
        try:
            sim = get_simbad_annotations(header, img.width, img.height, maglimit=simbad_maglimit)
            print(f"Annotated {len(sim)} object(s) from SIMBAD")
            anns += sim
        except Exception as e:
            print(f"Warning: SIMBAD annotation failed: {e}")
    if not anns:
        return img
    return draw_sky_annotations(img, anns, colors=colors, fontsize=fontsize)

def main():
    parser = argparse.ArgumentParser(description='Convert FITS file to JPEG with logarithmic scaling')
    parser.add_argument('fits_file', help='Input FITS file')
    parser.add_argument('-o', '--output', help='Output filename (default: auto-generated from input filename)')
    parser.add_argument('-c', '--color', choices=['heat', 'cool'], help='Color palette: heat or cool')
    parser.add_argument('-i', '--inverted', action='store_true', help='Invert colors (grayscale) or use cool palette (with -c heat)')
    parser.add_argument('-l', '--label', help='Label text to add at bottom of image (supports FITS header expansion like %%H:%%M)')
    parser.add_argument('-F', '--fits-out', action='store_true', help='Save as FITS with full header and 8-bit grayscale data (for Aladin)')
    parser.add_argument('-a', '--annotate', action='store_true', help='Annotate image with NGC/IC objects, named bright stars and GCVS variables from the WCS (requires --catalog-dir and/or --simbad)')
    parser.add_argument('--catalog-dir', dest='catalog_dir', help='Path to local catalogs directory (ngc2000.fits, ngc2000names.fits, brightstars.fits, gcvs.fits -- see fetch_gcvs.py)')
    parser.add_argument('--ann-maglimit', dest='ann_maglimit', type=float, help='Only label bright stars at or brighter than this V magnitude')
    parser.add_argument('--gcvs-maglimit', dest='gcvs_maglimit', type=float, help='Only label GCVS variables at or brighter than this magnitude at maximum')
    parser.add_argument('--ann-color', dest='ann_color', default='yellow', help='Annotation color for NGC/IC and bright stars (default: %(default)s)')
    parser.add_argument('--ann-var-color', dest='ann_var_color', default='cyan', help='Annotation color for GCVS variable stars (default: %(default)s)')
    parser.add_argument('--ann-fontsize', dest='ann_fontsize', default=14, type=float, help='Annotation label font size (default: %(default)s)')
    parser.add_argument('--simbad', action='store_true', help='Also query SIMBAD live for named objects in the field (requires network; opt-in since it is slow/rate-limited for batch use)')
    parser.add_argument('--simbad-maglimit', dest='simbad_maglimit', type=float, default=12.0, help='Only label SIMBAD hits at or brighter than this V magnitude (default: %(default)s)')
    parser.add_argument('--ann-simbad-color', dest='ann_simbad_color', default='orange', help='Annotation color for SIMBAD hits (default: %(default)s)')

    args = parser.parse_args()
    fits_file = args.fits_file
    save_as_fits = args.fits_out

    # Generate output filename first (needed for error cases)
    if args.output:
        output_filename = args.output
        # Auto-detect fits output from extension
        if output_filename.endswith('.fits') or output_filename.endswith('.fit'):
            save_as_fits = True
    elif save_as_fits:
        output_filename = fits_file.replace('.fits', '_8bit.fits').replace('.fit', '_8bit.fit')
        if output_filename == fits_file:
            output_filename = fits_file + '_8bit.fits'
    else:
        output_filename = fits_file.replace('.fits', '.jpg').replace('.fit', '.jpg')
        if output_filename == fits_file:  # No .fits extension found
            output_filename = fits_file + '.jpg'
    
    try:
        # Load FITS file
        with fits.open(fits_file) as f:
            data = f[0].data.astype(float)
            header = f[0].header
            
            # Get original image dimensions for fallback
            if data.ndim == 2:
                original_height, original_width = data.shape
            else:
                original_height, original_width = 600, 800

        # Filter out saturated pixels (above 60000) for quantile calculation
        saturation_level = 60000
        unsaturated_data = data[data <= saturation_level]
        
        if len(unsaturated_data) == 0:
            print("Warning: All pixels are saturated! Using full dataset.")
            unsaturated_data = data
        else:
            print(f"Filtered out {len(data) - len(unsaturated_data)} saturated pixels (>{saturation_level})")
        
        # Calculate quantiles on unsaturated data only
        quantiles = []
        fractions = [0.1, 0.5, 0.9, 0.9995]
        for q in fractions:
            quantiles.append(np.quantile(unsaturated_data, q))
        
        print(f"Quantiles: {dict(zip(fractions, quantiles))}")
        
        # Prepare data for fitting: map quantiles to target range
        # 1% -> 1% of range (2.55), 90% -> 10% of range (25.5), 99% -> 100% of range (255)
        x_data = np.array(quantiles)
        #y_data_log = np.log10(np.array([2.55, 25.5, 255]))
        y_data_log = np.log10(np.array([1, 255./8, 255./4, 255.]))
        
        # Fix C to 0.1% below the minimum (noise floor), matching gnuplot approach
        C_fixed = quantiles[0] - (quantiles[2] - quantiles[0]) / 1000
        
        # Create wrapper function with fixed C for curve_fit
        def fit_func(x, A, B):
            return log_function_with_fixed_c(x, A, B, C_fixed)
        
        # Initial parameter guesses: A=1, B=1 (C is now fixed)
        initial_guess = [1.0, 1.0]
        
        # Debug output: show fitting data points for gnuplot
        print("# Debug: Fitting data points (x y) for gnuplot:")
        for i in range(len(x_data)):
            print(f"{x_data[i]:.6f} {y_data_log[i]:.6f}")
        print(f"# Fixed C={C_fixed:.6f}, Initial guess: A={initial_guess[0]}, B={initial_guess[1]}")
        
        # Perform logarithmic fit
        popt, pcov = curve_fit(fit_func, x_data, y_data_log, p0=initial_guess)
        A, B = popt
        print(f"Fitted parameters: A={A:.4f}, B={B:.4f}, C={C_fixed:.4f} (fixed)")
        
        # Apply the transformation to all pixel values
        # Clamp values to avoid log of negative numbers (x - C_fixed must be positive)
        data_clamped = np.maximum(data, C_fixed + 1e-10)
        
        # Apply the fitted logarithmic transformation
        transformed = 10 ** log_function_with_fixed_c(data_clamped, A, B, C_fixed)
        
        # Clamp to 0-255 range and convert to uint8
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        
        if save_as_fits:
            # Save as FITS with full original header and 8-bit grayscale data
            out_header = header.copy()
            out_header['BITPIX'] = 8
            out_hdu = fits.PrimaryHDU(data=transformed, header=out_header)
            out_hdu.writeto(output_filename, overwrite=True)
            print(f"FITS (8-bit) saved as: {output_filename}")
            return

        # Apply color palette and inversion
        palette = args.color or 'none'
        is_inverted = args.inverted

        # Handle cool palette as inverted heat
        if args.color == 'cool':
            palette = 'heat'
            is_inverted = True

        colored_data = safe_apply_color_palette(transformed, palette, is_inverted)

        # Create JPEG
        if palette == 'none':
            # Grayscale image
            img = Image.fromarray(colored_data, mode='L')
        else:
            # Color image
            img = Image.fromarray(colored_data, mode='RGB')

        # Annotate with NGC/IC objects, named bright stars, GCVS variables and/or SIMBAD
        if args.annotate:
            if not args.catalog_dir and not args.simbad:
                print("Warning: --annotate requires --catalog-dir and/or --simbad; skipping annotation")
            else:
                colors = dict(default=args.ann_color, ngc=args.ann_color, bright=args.ann_color,
                              gcvs=args.ann_var_color, simbad=args.ann_simbad_color)
                img = safe_annotate_sky(img, header, catalog_dir=args.catalog_dir,
                                        star_maglimit=args.ann_maglimit,
                                        gcvs_maglimit=args.gcvs_maglimit,
                                        colors=colors, fontsize=args.ann_fontsize,
                                        do_simbad=args.simbad,
                                        simbad_maglimit=args.simbad_maglimit)

        # Add label if specified
        if args.label:
            img = safe_add_label_to_image(img, args.label, header)

        img.save(output_filename, 'JPEG', quality=95)
        print(f"JPEG saved as: {output_filename}")
        
    except Exception as e:
        # Create fallback white image with error message
        print(f"Error processing FITS file: {e}")
        try:
            # Try to determine appropriate size from existing file if possible
            fallback_width = original_width if 'original_width' in locals() else 800
            fallback_height = original_height if 'original_height' in locals() else 600
        except:
            fallback_width, fallback_height = 800, 600
        
        error_msg = f"Error: {str(e)[:50]}..." if len(str(e)) > 50 else str(e)
        img = create_fallback_image(fallback_width, fallback_height, error_msg)
        
        # Try to add original label even on error image
        if args.label:
            try:
                # Create minimal header for label expansion
                fallback_header = {'DATE-OBS': '2000-01-01T00:00:00'}
                img = safe_add_label_to_image(img, args.label, fallback_header)
            except:
                pass
        
        if save_as_fits:
            # Write a minimal blank 8-bit FITS as fallback
            blank = np.zeros((fallback_height, fallback_width), dtype=np.uint8)
            fits.PrimaryHDU(data=blank).writeto(output_filename, overwrite=True)
        else:
            img.save(output_filename, 'JPEG', quality=95)
        print(f"Fallback image saved as: {output_filename}")

if __name__ == "__main__":
    main()
