#!/usr/bin/python3

import numpy as np
import sys
import argparse
from astropy.io import fits
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
    
    # Try to load a decent font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
    
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
        
        # Try to load a font for error message
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 24)
            except:
                font = ImageFont.load_default()
        
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

def main():
    parser = argparse.ArgumentParser(description='Convert FITS file to JPEG with logarithmic scaling')
    parser.add_argument('fits_file', help='Input FITS file')
    parser.add_argument('-o', '--output', help='Output JPEG filename (default: auto-generated from input filename)')
    parser.add_argument('-c', '--color', choices=['heat', 'cool'], help='Color palette: heat or cool')
    parser.add_argument('-i', '--inverted', action='store_true', help='Invert colors (grayscale) or use cool palette (with -c heat)')
    parser.add_argument('-l', '--label', help='Label text to add at bottom of image (supports FITS header expansion like %%H:%%M)')
    
    args = parser.parse_args()
    fits_file = args.fits_file
    
    # Generate output filename first (needed for error cases)
    if args.output:
        output_filename = args.output
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
        
        img.save(output_filename, 'JPEG', quality=95)
        print(f"Fallback image saved as: {output_filename}")

if __name__ == "__main__":
    main()
