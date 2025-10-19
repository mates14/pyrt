import numpy as np
import astropy.wcs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_residual_plots(data, output_base, ffit, afit, plot_type='photometry'):
    """
    Create residual plots similar to the gnuplot script.

    Args:
        data: PhotometryData object containing the star data
        output_base: Base filename for output (without extension)
        plot_type: 'photometry' or 'astrometry' to determine plot type
    """
    # Get the current mask and data arrays
    #current_mask = data.get_current_mask()
    data.use_mask('photometry')
    current_mask = data.get_current_mask()
    data.use_mask('default')

    fd = data.get_fitdata('x', 'y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'dy', 'ra', 'dec', 'image_x', 'image_y', 'cat_x', 'cat_y', 'airmass')

    # Calculate astrometric residuals (if available)
    try:
        imgwcs = astropy.wcs.WCS(afit.wcs())
        astx, asty = imgwcs.all_world2pix(fd.ra, fd.dec, 1)
        ast_residuals = np.sqrt((astx - fd.coord_x)**2 + (asty - fd.coord_y)**2)
    except (KeyError,AttributeError):
        ast_residuals = np.zeros_like(fd.x)  # If astrometric data is not available

    # Calculate model magnitudes - use elegant fotparams property
    model_mags = ffit.model(np.array(ffit.fitvalues), fd.fotparams)

    # Calculate center coordinates for radius calculation
    X0 = np.mean(fd.coord_x)
    Y0 = np.mean(fd.coord_y)
    radius = np.sqrt((fd.coord_y - Y0)**2 + (fd.coord_x - X0)**2)

    # Set up the plotting parameters
#    plt.style.use('dark_background')
    fig = plt.figure(figsize=(19.2, 10.8))
    gs = GridSpec(5, 2, figure=fig)

    # List of parameters to plot
    params = [
    #    ('Color x Airmass', fd.adif * fd.color1),
        ('Radius', radius),
        ('Color3', fd.color3),
    #    ('Color4', fd.color4),
        ('Color1', fd.color1),
        ('Color2', fd.color2),
        ('CoordX', fd.coord_x),
        ('CoordY', fd.coord_y),
        ('Catalog', fd.x),
        ('Airmass', fd.adif)
    ]

    aparams = [
    #    ('Color x Airmass', fd.adif * fd.color1),
        ('CoordX', fd.coord_x),
        ('CoordY', fd.coord_y)
    ]

    # Color mapping for points based on errors
    error_colors = plt.cm.hsv(np.log10(fd.dy))

    if plot_type == 'photometry':
        # Magnitude residuals, up higher values, should me "measured brighter"
        residuals = fd.x - model_mags
        # I'd like to have 10-sigma here
        sigma = ffit.sigma
        ylim = (-sigma*7, sigma*7)

        for idx, (label, param) in enumerate(params):
            ax = fig.add_subplot(gs[idx // 2 + 1, idx % 2 ])

            # Plot masked points in gray
            ax.scatter(param[~current_mask], residuals[~current_mask],
                      c='gray', alpha=0.3, s=2)

            # Plot unmasked points with color coding
            sc = ax.scatter(param[current_mask], residuals[current_mask],
                          c=fd.dy[current_mask], cmap='hsv', s=2)

            ax.set_ylabel('Magnitude Residuals')
            ax.set_xlabel(label)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.2)

#    else:  # astrometry

        try:
            dx = fd.image_x - astx
            dy = fd.image_y - asty
            sigma = afit.sigma
        except:
            dx=np.zeros_like(fd.image_x)
            dy=np.zeros_like(fd.image_x)
            sigma=0.01

        ylim = (-sigma*7, sigma*7)

#        for idx, (label, param) in enumerate(params):
        for idx, (label, param) in enumerate(aparams):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            # Plot X residuals
            ax.scatter(param[~current_mask], dx[~current_mask],
                      c='#ffaaaa', alpha=0.3, s=2, label='X (masked)')
            ax.scatter(param[~current_mask], dy[~current_mask],
                      c='#aaffaa', alpha=0.3, s=2, label='Y (masked)')

            ax.scatter(param[current_mask], dx[current_mask],
                      c='#aa0000', s=2, label='X')
            ax.scatter(param[current_mask], dy[current_mask],
                      c='#00aa00', s=2, label='Y')

            if idx == 0:  # Only show legend for first plot
                ax.legend(markerscale=3)

            ax.set_ylabel('Position Residuals')
            ax.set_xlabel(label)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save the plot
    output_filename = f"{output_base}-{'ast' if plot_type == 'astrometry' else 'phot'}.png"
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    plt.close()

    return output_filename


def create_correction_volume_plots(data, output_base, ffit):
    """
    Create plots showing the volume/shape of corrections that depend on input coordinates.
    
    Evaluates the model for each identification with the coordinate set to zero vs normal,
    showing the differences as a function of the varied coordinate.
    
    Args:
        data: PhotometryData object containing the star data
        output_base: Base filename for output (without extension)
        ffit: Fitted photometric model
    """
    # Get the current data
    data.use_mask('photometry')
    current_mask = data.get_current_mask()
    data.use_mask('default')
    
    fd = data.get_fitdata('x', 'y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'dy', 'ra', 'dec', 'image_x', 'image_y', 'cat_x', 'cat_y', 'airmass')
    
    # Calculate model magnitudes with original coordinates
    model_original = ffit.model(np.array(ffit.fitvalues), fd.fotparams)
    
    # Coordinate names and their indices in fotparams
    # fotparams format: (mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err, cat_x, cat_y, airmass_abs)
    coordinate_specs = [
        ('coord_x', 2, fd.coord_x, 'Coordinate X'),
        ('coord_y', 3, fd.coord_y, 'Coordinate Y'),
        ('color1', 4, fd.color1, 'Color 1'),
        ('color2', 5, fd.color2, 'Color 2'), 
        ('color3', 6, fd.color3, 'Color 3'),
        ('color4', 7, fd.color4, 'Color 4'),
        ('airmass', 1, fd.adif, 'Airmass'),
    ]
    
    # Additional coordinate information for subpixel plots
    subpixel_x = fd.coord_x - np.floor(fd.coord_x)
    subpixel_y = fd.coord_y - np.floor(fd.coord_y)
    
    # Calculate center coordinates for radius calculation
    X0 = np.mean(fd.coord_x)
    Y0 = np.mean(fd.coord_y)
    radius = np.sqrt((fd.coord_y - Y0)**2 + (fd.coord_x - X0)**2)
    
    # Add radius and magnitude to coordinate specs
    coordinate_specs.extend([
        ('radius', None, radius, 'Radius'),
        ('magnitude', 0, fd.x, 'Catalog Magnitude')
    ])
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(6, 6, figure=fig)
    
    plot_idx = 0
    
    for coord_name, param_idx, coord_values, label in coordinate_specs:
        if plot_idx >= 15:  # Limit to available subplot positions
            break
            
        # Create modified fotparams with this coordinate set to zero
        modified_fotparams = list(fd.fotparams)
        if param_idx is not None:
            if coord_name == 'magnitude':
                # For magnitude, use the mean value instead of zero
                modified_fotparams[param_idx] = np.full_like(coord_values, np.mean(coord_values))
            else:
                modified_fotparams[param_idx] = np.zeros_like(coord_values)
        
        # Calculate model with zeroed coordinate
        if param_idx is not None:
            model_zeroed = ffit.model(np.array(ffit.fitvalues), tuple(modified_fotparams))
            correction = model_original - model_zeroed
        else:
            # For derived quantities like radius, we can't directly zero them in the model
            # Instead, we'll show the correlation with the original residuals
            residuals = fd.x - model_original
            correction = residuals
        
        # Calculate percentile-based range (zscale approach) for y-axis scaling
        correction_active = correction[current_mask]
        if len(correction_active) > 0:
            # Use 1st and 99th percentiles for robust range estimation
            p1, p99 = np.percentile(correction_active, [1, 99])
            data_range = p99 - p1
            # Add minimum buffer of 5% of range or 0.01, whichever is larger
            buffer = max(0.01, data_range * 0.05)
            ylim = (p1 - buffer, p99 + buffer)
        else:
            ylim = (-0.01, 0.01)  # Default small range if no active points
        
        # Create subplot - using 2 columns per plot for better spacing
        row = plot_idx // 3
        col = (plot_idx % 3) * 2
        ax = fig.add_subplot(gs[row, col:col+2])
        
        # Plot masked points in gray
        ax.scatter(coord_values[~current_mask], correction[~current_mask],
                  c='gray', alpha=0.3, s=2, label='Masked')
        
        # Plot unmasked points with color coding by error
        sc = ax.scatter(coord_values[current_mask], correction[current_mask],
                      c=fd.dy[current_mask], cmap='hsv', s=2, label='Active')
        
        ax.set_ylabel('Model Correction [mag]')
        ax.set_xlabel(label)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.2)
        
        # Add colorbar for first plot
        if plot_idx == 0:
            plt.colorbar(sc, ax=ax, label='Photometric Error')
        
        plot_idx += 1
    
    # Create x/y map of corrections - larger spatial plots at bottom
    # Average correction per coordinate
    avg_correction = np.mean([model_original - ffit.model(np.array(ffit.fitvalues), 
                                                         (fd.fotparams[0], fd.fotparams[1], np.zeros_like(fd.coord_x), fd.fotparams[3], 
                                                          fd.fotparams[4], fd.fotparams[5], fd.fotparams[6], fd.fotparams[7], 
                                                          fd.fotparams[8], fd.fotparams[9], fd.fotparams[10], fd.fotparams[11], fd.fotparams[12], fd.fotparams[13])),
                             model_original - ffit.model(np.array(ffit.fitvalues), 
                                                         (fd.fotparams[0], fd.fotparams[1], fd.fotparams[2], np.zeros_like(fd.coord_y), 
                                                          fd.fotparams[4], fd.fotparams[5], fd.fotparams[6], fd.fotparams[7], 
                                                          fd.fotparams[8], fd.fotparams[9], fd.fotparams[10], fd.fotparams[11], fd.fotparams[12], fd.fotparams[13]))], axis=0)
    
    # X/Y spatial map - use 3 columns width
    ax = fig.add_subplot(gs[3:6, 0:3])
    sc = ax.scatter(fd.coord_x[current_mask], fd.coord_y[current_mask], 
                   c=avg_correction[current_mask], cmap='RdBu_r', s=8)
    ax.scatter(fd.coord_x[~current_mask], fd.coord_y[~current_mask], 
              c='gray', alpha=0.3, s=4)
    ax.set_xlabel('Coordinate X')
    ax.set_ylabel('Coordinate Y')
    ax.set_title('X/Y Spatial Map of Corrections')
    plt.colorbar(sc, ax=ax, label='Average XY Correction')
    
    # Subpixel X/Y map - compute actual subpixel correction
    # This is the difference between model at fractional vs integer positions
    ax = fig.add_subplot(gs[3:6, 3:6])
    
    # Create fotparams with integer-floored coordinates
    rounded_fotparams = list(fd.fotparams)
    rounded_fotparams[2] = np.floor(fd.coord_x)  # Floor coord_x to integers
    rounded_fotparams[3] = np.floor(fd.coord_y)  # Floor coord_y to integers
    
    # Calculate model with rounded coordinates
    model_rounded = ffit.model(np.array(ffit.fitvalues), rounded_fotparams)
    
    # Subpixel correction is the difference: fractional position - integer position
    subpixel_correction = model_original - model_rounded
    subpixel_correction = np.clip(subpixel_correction, -0.1, 0.1)
    
    sc = ax.scatter(subpixel_x[current_mask], subpixel_y[current_mask], 
                   c=subpixel_correction[current_mask], cmap='RdBu_r', s=8)
    ax.scatter(subpixel_x[~current_mask], subpixel_y[~current_mask], 
              c='gray', alpha=0.3, s=4)
    ax.set_xlabel('Subpixel X')
    ax.set_ylabel('Subpixel Y')
    ax.set_title('Subpixel X/Y Map of Corrections')
    plt.colorbar(sc, ax=ax, label='Subpixel Correction')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"{output_base}-corrections.png"
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_filename
