def transform_catalog_magnitudes(det, cat, imgwcs):
    """
    Transform catalog magnitudes to the detector's photometric system using the stored model.
    
    Args:
        det: Detection table with metadata including the photometric model
        cat: Catalog data from get_atlas() or get_catalog()
        imgwcs: WCS object for coordinate transformation
        
    Returns:
        astropy.table.Table: Catalog with additional column 'mag_instrument' matching detector response
    """
    try:
        # Load the photometric model
        ffit = fotfit.fotfit()
        ffit.from_oneline(det.meta['RESPONSE'])
        
        # Transform catalog coordinates to pixel coordinates
        try:
            cat_x, cat_y = imgwcs.all_world2pix(cat['radeg'], cat['decdeg'], 1)
        except:
            print("Error transforming coordinates")
            return None
            
        # Prepare color indices from catalog data
        color1 = cat['Sloan_g'] - cat['Sloan_r']  # g-r
        color2 = cat['Sloan_r'] - cat['Sloan_i']  # r-i
        color3 = cat['Sloan_i'] - cat['Sloan_z']  # i-z
        color4 = np.zeros_like(cat['Sloan_r'])    # placeholder for J if needed
        
        # Base magnitude based on filter used in photometric solution
        base_filter = det.meta.get('REFILTER', 'Sloan_r')  # Default to Sloan_r if not specified
        base_mag = cat[base_filter]
        
        # Prepare input data for the photometric model
        model_input = (
            base_mag,                                    # catalog magnitude
            det.meta['AIRMASS'],                        # airmass
            (cat_x - det.meta['CTRX'])/1024,           # normalized X coordinate
            (cat_y - det.meta['CTRY'])/1024,           # normalized Y coordinate
            color1,                                     # g-r color
            color2,                                     # r-i color
            color3,                                     # i-z color
            color4,                                     # extra color term
            det.meta['IMGNO'],                         # image number
            np.zeros_like(base_mag),                   # y (not used)
            np.ones_like(base_mag)                     # errors (not used)
        )
        
        # Apply the model to get instrumental magnitudes
        cat_transformed = cat.copy()
        cat_transformed['mag_instrument'] = ffit.model(ffit.fixvalues, model_input)
        
        return cat_transformed
        
    except KeyError as e:
        print(f"Missing required metadata: {e}")
        return None
    except Exception as e:
        print(f"Error applying photometric model to catalog: {e}")
        return None

def match_detections_with_catalog(det, cat, options):
    """
    Match detections with transformed catalog data.
    
    Args:
        det: Detection table
        cat: Original catalog from get_atlas()
        options: Command line options containing matching parameters
    
    Returns:
        tuple: (matched_detections, matched_catalog)
    """
    # Get WCS from detection metadata
    imgwcs = astropy.wcs.WCS(det.meta)
    
    # Transform catalog magnitudes to instrumental system
    cat_transformed = transform_catalog_magnitudes(det, cat, imgwcs)
    if cat_transformed is None:
        return None, None
    
    # Convert catalog coordinates to pixel space for matching
    cat_x, cat_y = imgwcs.all_world2pix(cat_transformed['radeg'], 
                                       cat_transformed['decdeg'], 1)
    
    # Build KD-tree for spatial matching
    X = np.array([cat_x, cat_y]).transpose()
    Y = np.array([det['X_IMAGE'], det['Y_IMAGE']]).transpose()
    
    tree = KDTree(X)
    idlimit = options.idlimit if options.idlimit else det.meta.get('FWHM', 2.0)
    nearest_ind, nearest_dist = tree.query_radius(Y, r=idlimit, 
                                                return_distance=True)
    
    return det, cat_transformed, nearest_ind, nearest_dist

# Example usage in transients.py:
def process_image(arg, options):
    det = process_input_file(arg, options)
    if det is None:
        return
        
    # Get catalog data
    cat = get_atlas(det.meta['CTRRA'], det.meta['CTRDEC'], 
                   width=det.meta['FIELD'], 
                   height=det.meta['FIELD'],
                   mlim=options.maglim)
                   
    # Match detections with transformed catalog
    det, cat_transformed, nearest_ind, nearest_dist = \
        match_detections_with_catalog(det, cat, options)
    
    if det is None:
        return
        
    # Now compare instrumental magnitudes for transient detection
    for i, (indices, distances) in enumerate(zip(nearest_ind, nearest_dist)):
        if len(indices) == 0:
            # No catalog match - potential transient
            process_candidate(det[i], cat_transformed, options)
        else:
            # Compare magnitudes for variability
            cat_mag = cat_transformed['mag_instrument'][indices[0]]
            det_mag = det[i]['MAG_AUTO']
            if abs(cat_mag - det_mag) > options.siglim * det[i]['MAGERR_AUTO']:
                process_variable(det[i], cat_transformed[indices[0]], options)
