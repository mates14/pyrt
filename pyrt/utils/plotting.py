import numpy as np
import astropy.wcs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import hsv_to_rgb

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
        ('Airmass', fd.airmass)
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

        # Create reference fotparams with ALL parameters at reference values (median)
        reference_fotparams = [
            np.full_like(fd.fotparams[i], np.median(fd.fotparams[i]))
            for i in range(len(fd.fotparams))
        ]

        # Now set ONLY the parameter of interest to its actual values
        if param_idx is not None:
            reference_fotparams[param_idx] = coord_values

            # Calculate model with only this parameter varying
            model_only_this = ffit.model(np.array(ffit.fitvalues), tuple(reference_fotparams))

            # Isolated effect: observations - F(all) + F(only this param)
            correction = fd.x - model_original + model_only_this
        else:
            # For derived quantities like radius, we can't directly vary them in the model
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
    # Calculate isolated effects for coord_x and coord_y

    # For coord_x: reference model with only coord_x varying
    ref_params_x = [np.full_like(fd.fotparams[i], np.median(fd.fotparams[i]))
                    for i in range(len(fd.fotparams))]
    ref_params_x[2] = fd.coord_x  # Keep coord_x actual values
    model_only_x = ffit.model(np.array(ffit.fitvalues), tuple(ref_params_x))
    correction_x = fd.x - model_original + model_only_x

    # For coord_y: reference model with only coord_y varying
    ref_params_y = [np.full_like(fd.fotparams[i], np.median(fd.fotparams[i]))
                    for i in range(len(fd.fotparams))]
    ref_params_y[3] = fd.coord_y  # Keep coord_y actual values
    model_only_y = ffit.model(np.array(ffit.fitvalues), tuple(ref_params_y))
    correction_y = fd.x - model_original + model_only_y

    # Average the X and Y spatial corrections
    avg_correction = np.mean([correction_x, correction_y], axis=0)
    
    # X/Y spatial map - use 3 columns width with HSV color encoding
    ax = fig.add_subplot(gs[3:6, 0:3])

    # Prepare HSV colors for active points
    # Hue: encode correction value (use 0-240° range: red->yellow->green->cyan->blue)
    # Can be changed to 240-360° (blue->violet->red) by using hue_range=(240/360, 1.0)
    correction_active = avg_correction[current_mask]
    errors_active = fd.dy[current_mask]

    # Normalize corrections to hue range [0, 0.667] (0-240 degrees)
    # For blue->violet->red range, use [0.667, 1.0] instead
    hue_range = (0.0, 0.667)  # Change to (0.667, 1.0) for 240-360° range
    corr_min, corr_max = np.percentile(correction_active, [1, 99])
    if corr_max > corr_min:
        hue_normalized = (correction_active - corr_min) / (corr_max - corr_min)
    else:
        hue_normalized = np.full_like(correction_active, 0.5)
    hue = hue_range[0] + hue_normalized * (hue_range[1] - hue_range[0])

    # Saturation: encode precision (inverse of uncertainty)
    # High precision (small error) = high saturation (vivid colors)
    # Low precision (large error) = low saturation (washed out colors)
    error_min, error_max = np.percentile(errors_active, [1, 99])
    if error_max > error_min:
        # Invert so small errors -> high saturation
        saturation = 1.0 - (errors_active - error_min) / (error_max - error_min)
        saturation = np.clip(saturation, 0.3, 1.0)  # Keep some minimum saturation
    else:
        saturation = np.ones_like(errors_active)

    # Value: constant maximum brightness
    value = np.ones_like(hue)

    # Stack HSV and convert to RGB
    hsv_colors = np.stack([hue, saturation, value], axis=-1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    # Clip to valid range to handle floating point errors and outliers
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

    # Plot active points with HSV colors
    ax.scatter(fd.coord_x[current_mask], fd.coord_y[current_mask],
               c=rgb_colors, s=8, edgecolors='none')

    # Plot masked points in gray
    ax.scatter(fd.coord_x[~current_mask], fd.coord_y[~current_mask],
              c='gray', alpha=0.3, s=4)

    ax.set_xlabel('Coordinate X')
    ax.set_ylabel('Coordinate Y')
    ax.set_title('X/Y Spatial Map of Corrections (Hue=Correction, Saturation=Precision)')

    # Create a custom colorbar showing the hue scale
    # Make a small sample of colors for the colorbar
    n_colors = 256
    cbar_hue = np.linspace(hue_range[0], hue_range[1], n_colors)
    cbar_sat = np.ones(n_colors)
    cbar_val = np.ones(n_colors)
    cbar_hsv = np.stack([cbar_hue, cbar_sat, cbar_val], axis=-1)
    cbar_rgb = hsv_to_rgb(cbar_hsv.reshape(1, -1, 3)).reshape(-1, 3)
    # Clip to valid range to handle floating point errors
    cbar_rgb = np.clip(cbar_rgb, 0.0, 1.0)

    # Create a dummy mappable for colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=corr_min, vmax=corr_max)
    sm = ScalarMappable(norm=norm, cmap=plt.cm.colors.ListedColormap(cbar_rgb))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Average XY Correction [mag]')
    
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

    # Create a 2D grid to evaluate the model's subpixel sensitivity function
    # Evaluate model(fractional) - model(0,0) to show only subpixel variation
    grid_resolution = 50
    subpix_grid_x, subpix_grid_y = np.meshgrid(
        np.linspace(0, 1, grid_resolution),
        np.linspace(0, 1, grid_resolution)
    )

    # Use a fixed base position for grid evaluation
    base_coord_x = 1024.0
    base_coord_y = 1024.0

    # Create fotparams for grid evaluation
    n_grid = grid_resolution * grid_resolution
    grid_fotparams = [
        np.full(n_grid, np.median(fd.fotparams[0].flat)),  # magnitude
        np.full(n_grid, np.median(fd.fotparams[1].flat)),  # airmass
        (base_coord_x + subpix_grid_x).flatten(),  # coord_x with fractional part
        (base_coord_y + subpix_grid_y).flatten(),  # coord_y with fractional part
        np.full(n_grid, np.median(fd.fotparams[4].flat)),  # color1
        np.full(n_grid, np.median(fd.fotparams[5].flat)),  # color2
        np.full(n_grid, np.median(fd.fotparams[6].flat)),  # color3
        np.full(n_grid, np.median(fd.fotparams[7].flat)),  # color4
        np.full(n_grid, np.median(fd.fotparams[8].flat)),  # img
        np.full(n_grid, np.median(fd.fotparams[9].flat)),  # y
        np.full(n_grid, np.median(fd.fotparams[10].flat)), # err
        np.full(n_grid, np.median(fd.fotparams[11].flat)), # cat_x
        np.full(n_grid, np.median(fd.fotparams[12].flat)), # cat_y
        np.full(n_grid, np.median(fd.fotparams[13].flat)), # airmass_abs
    ]

    # Evaluate model on the grid
    try:
        model_grid = ffit.model(np.array(ffit.fitvalues), tuple(grid_fotparams))
        model_grid = model_grid.reshape(grid_resolution, grid_resolution)

        # Subtract the mean to show only variation (removes baseline offset)
        grid_correction = model_grid - np.mean(model_grid)

        # Plot the model subpixel sensitivity as a mesh/heatmap
        im = ax.pcolormesh(subpix_grid_x, subpix_grid_y, grid_correction,
                          cmap='RdBu_r', alpha=0.7, shading='auto',
                          vmin=-0.05, vmax=0.05)
        plt.colorbar(im, ax=ax, label='Model Subpixel Sensitivity [mag]')
    except Exception as e:
        print(f"Warning: Could not evaluate model grid: {e}")

    # Data points: show isolated subpixel effect
    # Formula: observations - F(all) + F(only_subpixel)
    # Where F(only_subpixel) ≈ F(fractional) - F(integer) when other params at reference
    # Simplified: observations - model_full + (model_fractional - model_rounded)
    subpixel_isolated = fd.x - model_original + (model_original - model_rounded)
    # Which simplifies to: observations - model_rounded
    data_residuals = subpixel_isolated
    data_residuals_clipped = np.clip(data_residuals, -0.1, 0.1)

    # Plot masked points as small gray dots
    ax.scatter(subpixel_x[~current_mask], subpixel_y[~current_mask],
              c='gray', alpha=0.3, s=2, edgecolors='none')

    # Plot active data points: data minus model at integer positions
    sc = ax.scatter(subpixel_x[current_mask], subpixel_y[current_mask],
                   c=data_residuals_clipped[current_mask], cmap='RdBu_r',
                   s=15, alpha=0.8, edgecolors='black', linewidths=0.3,
                   vmin=-0.05, vmax=0.05)

    ax.set_xlabel('Subpixel X')
    ax.set_ylabel('Subpixel Y')
    ax.set_title('Subpixel Sensitivity: Model (mesh) + Data (points)')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"{output_base}-corr.png"
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_filename
