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

    x, y, adif, coord_x, coord_y, color1, color2, color3, color4, img, dy, ra, dec, image_x, image_y, cat_x, cat_y = data.get_arrays(
        'x', 'y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'dy', 'ra', 'dec', 'image_x', 'image_y', 'cat_x', 'cat_y'
    )

    # Calculate astrometric residuals (if available)
    try:
        imgwcs = astropy.wcs.WCS(afit.wcs())
        astx, asty = imgwcs.all_world2pix( ra, dec, 1)
        ast_residuals = np.sqrt((astx - coord_x)**2 + (asty - coord_y)**2)
    except (KeyError,AttributeError):
        ast_residuals = np.zeros_like(x)  # If astrometric data is not available

    # Calculate model magnitudes
    model_input = (y, adif, coord_x, coord_y, color1, color2, color3, color4, img, x, dy, image_x, image_y)
    model_mags = ffit.model(np.array(ffit.fitvalues), model_input)

    # Calculate center coordinates for radius calculation
    X0 = np.mean(coord_x)
    Y0 = np.mean(coord_y)
    radius = np.sqrt((coord_y - Y0)**2 + (coord_x - X0)**2)

    # Set up the plotting parameters
#    plt.style.use('dark_background')
    fig = plt.figure(figsize=(19.2, 10.8))
    gs = GridSpec(5, 2, figure=fig)

    # List of parameters to plot
    params = [
    #    ('Color x Airmass', adif * color1),
        ('Radius', radius),
        ('Color3', color3),
    #    ('Color4', color4),
        ('Color1', color1),
        ('Color2', color2),
        ('CoordX', coord_x),
        ('CoordY', coord_y),
        ('Catalog', x),
        ('Airmass', adif)
    ]

    aparams = [
    #    ('Color x Airmass', adif * color1),
        ('CoordX', coord_x),
        ('CoordY', coord_y)
    ]

    # Color mapping for points based on errors
    error_colors = plt.cm.hsv(np.log10(dy))

    if plot_type == 'photometry':
        # Magnitude residuals, up higher values, should me "measured brighter"
        residuals = x - model_mags
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
                          c=dy[current_mask], cmap='hsv', s=2)

            ax.set_ylabel('Magnitude Residuals')
            ax.set_xlabel(label)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.2)

#    else:  # astrometry

        try:
            dx = image_x - astx
            dy = image_y - asty
            sigma = afit.sigma
        except:
            dx=np.zeros_like(image_x)
            dy=np.zeros_like(image_x)
            sigma=0.1

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
