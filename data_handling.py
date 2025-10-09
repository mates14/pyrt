import numpy as np
import logging
from airmass import calculate_airmass_array

class FitData:
    """
    A wrapper for photometric fit data that behaves like a tuple but allows attribute access.

    DESIGN PHILOSOPHY:
    FitData represents an immutable snapshot of photometric data at a specific mask state.
    It provides a clean interface for fitting functions while maintaining tuple compatibility
    for legacy code. This class intentionally does NOT hold references to PhotometryData
    to avoid complex masking state dependencies.

    KEY DESIGN DECISIONS:
    - Immutable: Once created, data cannot change (no mask switching surprises)
    - Simple: Just data arrays + attribute access + tuple compatibility
    - Focused: Only handles fitting interface, not data management
    - Lightweight: Minimal overhead, created and discarded as needed

    ANTI-PATTERNS TO AVOID:
    - Making this a "view" of PhotometryData (breaks with mask switching)
    - Adding data management features (that's PhotometryData's job)
    - Lazy evaluation of properties (creates hidden state dependencies)

    USAGE PATTERN:
    fd = data.get_fitdata(...)     # Create snapshot with current mask
    ffit.fit(fd.fotparams)         # Use for fitting
    # fd is discarded, create new one after mask changes
    """

    def __init__(self, field_names, *arrays):
        self._arrays = arrays
        self._field_names = field_names
        # Create attribute access for named fields
        for i, name in enumerate(field_names):
            if i < len(arrays):
                setattr(self, name, arrays[i])

    def __getitem__(self, index):
        """Allow tuple-like indexing for backward compatibility."""
        return self._arrays[index]

    def __len__(self):
        """Return number of arrays."""
        return len(self._arrays)

    def __iter__(self):
        """Allow tuple unpacking for backward compatibility."""
        return iter(self._arrays)

    @property
    def fotparams(self):
        """Return tuple in the format expected by fotfit.model()."""
        # fotfit expects: mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err, cat_x, cat_y, airmass_abs
        # dophot3 passes: (y, adif, coord_x, coord_y, color1, color2, color3, color4, img, x, dy, image_x, image_y, airmass)
        return (self.y, self.adif, self.coord_x, self.coord_y, self.color1, self.color2,
                self.color3, self.color4, self.img, self.x, self.dy, self.image_x, self.image_y, self.airmass)

    @property
    def astparams(self):
        """Return tuple in the format expected by zpntest.model(), zpntest.fit(), etc."""
        # zpntest expects: image_x, image_y, ra, dec, image_dxy
        return (self.image_x, self.image_y, self.ra, self.dec, self.image_dxy)

class PhotometryData:
    """
    Manages photometric data with complex masking and filtering operations.

    DESIGN PHILOSOPHY:
    PhotometryData handles the complex data management aspects: storage, masking,
    filtering, coordinate transformations, and metadata. It provides methods to
    create immutable FitData snapshots for fitting operations.

    KEY RESPONSIBILITIES:
    - Data storage and organization (_data, _meta dictionaries)
    - Complex masking operations (multiple named masks, mask switching)
    - Data filtering and validation
    - Coordinate transformations and airmass calculations
    - Metadata management

    MASKING WORKFLOW COMPLEXITY:
    The masking system supports non-linear workflows like:
    1. data.use_mask('photometry') → get data → fit
    2. Calculate residuals on ALL data (different mask!)
    3. data.add_mask('combined', new_mask) → switch → refit
    4. Multiple concurrent mask states for different purposes

    SEPARATION OF CONCERNS:
    PhotometryData (data management) ↔ FitData (fitting interface)
    This separation prevents architectural fusion attempts that would create
    subtle bugs due to mask state dependencies and concurrent access patterns.

    WHY NOT FUSION WITH FitData?
    - PhotometryData: Complex, stateful, handles data management
    - FitData: Simple, immutable, handles fitting interface
    - Fusion would create timing bugs with mask switching
    - Current design follows Unix philosophy: "Do one thing well"
    """
    def __init__(self):
        self._data = {}
        self._meta = {}
        self._masks = {}
        self._current_mask = None
        self._filter_columns = []
        self._current_filter = None
        self._required_columns = ['y', 'adif', 'coord_x', 'coord_y', 'img', 'dy']
        self._image_counts = {}
        self._total_objects = 0

    def init_column(self, name):
        if name not in self._data and name != 'x':
            self._data[name] = []

    def append(self, **kwargs):
        for name, value in kwargs.items():
            if name != 'x':
                self.init_column(name)
                self._data[name].append(value)

    def extend(self, **kwargs):
        for name, value in kwargs.items():
            if name != 'x':
                self.init_column(name)
                self._data[name].extend(value)
                if name == 'img':
                    for img_no in value:
                        self._image_counts[img_no] = self._image_counts.get(img_no, 0) + 1

    def set_meta(self, key, value):
        self._meta[key] = value

    def get_meta(self, key, default=None):
        return self._meta.get(key, default)

    def finalize(self):
        for name in self._data:
            self._data[name] = np.array(self._data[name])

        # Check if any data was successfully processed
        if not self._data:
            logging.warning("No data was successfully processed - all images were skipped")
            return

        # Calculate total objects after converting to arrays
        self._total_objects = len(next(iter(self._data.values())))
        # total_objects = sum(self._image_counts.values())

        # Create default mask of appropriate length
        self.add_mask('default', np.ones(self._total_objects, dtype=bool))
        self.use_mask('default')

    def add_mask(self, name, mask):
        """Add a new mask, ensuring it matches the data length."""
        if len(mask) != self._total_objects:
            raise ValueError(f"Mask length ({len(mask)}) does not match data length ({self._total_objects})")
        self._masks[name] = mask

    def use_mask(self, name):
        if name not in self._masks:
            raise ValueError(f"Mask '{name}' does not exist.")
        self._current_mask = name

    def get_current_mask(self):
        if self._current_mask is None:
            raise ValueError("No mask is currently active.")
        return self._masks[self._current_mask]

    def compute_colors_and_apply_limits(self, phschema, options):
        """
        Compute colors based on the selected photometric system and apply color limits.

        Args:
        photometric_system (str): Either 'Johnson' or 'AB' to indicate the photometric system.
        options (argparse.Namespace): Command line options containing redlim and bluelim.

        Returns:
        None. Updates the object in-place.
        """

        # Compute colors
        print(options.filter_schemas[phschema])
        schema = options.filter_schemas[phschema]
        mags = [self._data[f] for f in options.filter_schemas[phschema] ]
        self._data['color1'] = mags[0%len(schema)] - mags[1%len(schema)]
        self._data['color2'] = mags[1%len(schema)] - mags[2%len(schema)]
        self._data['color3'] = mags[2%len(schema)] - mags[3%len(schema)]
        self._data['color4'] = mags[3%len(schema)] - mags[4%len(schema)]

        #color_mask = self._data['color3'] > 5
        # Apply color limits
        #if options.redlim is not None:
        #    color_mask &= (self._data['color1'] + self._data['color2'])/2 <= options.redlim
        #if options.bluelim is not None:
        #    color_mask &=  (self._data['color1'] + self._data['color2'])/2 >= options.bluelim
        # Color filtering is now handled at catalog input stage in make_pairs_to_fit()
        pass

    def get_arrays(self, *names):
        """Get arrays applying the current mask."""
        self.check_required_columns()

        arrays = []
        current_mask = self._masks[self._current_mask]

        for name in names:
            if name == 'x':
                if not self._current_filter:
                    raise ValueError("No filter is currently set. Use set_current_filter() first.")
                # Ensure the filter data matches the mask length
                if len(self._data[self._current_filter]) != len(current_mask):
                    raise ValueError(f"Filter data length ({len(self._data[self._current_filter])}) "
                                   f"does not match mask length ({len(current_mask)})")
                arrays.append(self._data[self._current_filter][current_mask])
            elif name in self._data:
                # Ensure data array matches mask length
                if len(self._data[name]) != len(current_mask):
                    raise ValueError(f"Data array '{name}' length ({len(self._data[name])}) "
                                   f"does not match mask length ({len(current_mask)})")
                arrays.append(self._data[name][current_mask])
            else:
                raise KeyError(f"Column '{name}' not found in data. Available columns: {', '.join(self._data.keys())}")

        return tuple(arrays)

    def get_fitdata(self, *names):
        """
        Create an immutable FitData snapshot of the requested data columns.

        This method creates a FitData object containing the current masked state
        of the requested columns. The resulting FitData is independent of this
        PhotometryData object and will not change if masks are switched later.

        Returns:
            FitData: Immutable snapshot with attribute access and tuple compatibility
        """
        arrays = self.get_arrays(*names)
        return FitData(names, *arrays)

    def apply_mask(self, mask, name=None):
        """Apply a new mask, ensuring proper length."""
        if len(mask) != self._total_objects:
            raise ValueError(f"New mask length ({len(mask)}) does not match data length ({self._total_objects})")

        if name is None:
            self._masks[self._current_mask] &= mask
        else:
            self.add_mask(name, self._masks[self._current_mask] & mask)
            self.use_mask(name)

    def reset_mask(self, name=None):
        if name is None:
            name = self._current_mask
        self._masks[name] = np.ones(len(next(iter(self._data.values()))), dtype=bool)

    def add_filter_column(self, filter_name, data):
        if filter_name not in self._data:
            self._data[filter_name] = []
        self._data[filter_name].extend(data)
        if filter_name not in self._filter_columns:
            self._filter_columns.append(filter_name)
        if not self._current_filter:
            self.set_current_filter(filter_name)

    def get_filter_columns(self):
        return self._filter_columns

    def set_current_filter(self, filter_name):
        if filter_name not in self._filter_columns:
            raise ValueError(f"Filter '{filter_name}' not found in data.")
        self._current_filter = filter_name

    def get_current_filter(self):
        return self._current_filter

    def check_required_columns(self):
        missing_columns = [col for col in self._required_columns if col not in self._data]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def __len__(self):
        if self._data:
            return np.sum(self._masks[self._current_mask])
        return 0

    def __repr__(self):
        return f"PhotometryData with columns: {list(self._data.keys())}, current filter: {self._current_filter}, current mask: {self._current_mask}"

def make_pairs_to_fit(det, cat, nearest_ind, imgwcs, options, data, maglim=None, target_match=None):
    """
    Efficiently create pairs of data to be fitted.

    :param det: Detection table
    :param cat: Catalog table
    :param nearest_ind: Indices of nearest catalog stars for each detection
    :param imgwcs: WCS object for the image
    :param options: Command line options
    :param data: PhotometryData object to store results
    :return: Number of matched stars added to the data
    """
    try:
        valid_matches = np.array([len(inds) > 0 for inds in nearest_ind])
        det_data = np.array([det['X_IMAGE'], det['Y_IMAGE'], det['MAG_AUTO'], det['MAGERR_AUTO'], det['ERRX2_IMAGE'], det['ERRY2_IMAGE']]).T[valid_matches]
        cat_inds = np.array([inds[0] if len(inds) > 0 else -1 for inds in nearest_ind])[valid_matches]
        cat_data = cat[cat_inds]

        # RA+Dec of detections from their measured X&Y
        ra, dec = imgwcs.all_pix2world(det_data[:, 0], det_data[:, 1], 1)
        # X,Y for catalog for their catalog RA&Dec
        cat_x, cat_y = imgwcs.all_world2pix(cat_data['radeg'], cat_data['decdeg'], 1)

        # Calculate airmass if we have reliable time and location info
        try:
            # Check if we have the required information for airmass calculation
            required_keys = ['LATITUDE', 'LONGITUD', 'ALTITUDE', 'JD']
            missing_keys = [key for key in required_keys if key not in det.meta or det.meta[key] is None]

            if missing_keys:
                print(f"Missing airmass calculation info: {missing_keys} - using default airmass arrays")
                # Use default airmass: absolute=1.0, relative=0.0 for all objects
                airmass = np.ones_like(ra)  # Absolute airmass = 1.0 (zenith)
            else:
                # Calculate airmass using coordinate transformation
                # (slow astropy imports happen inside this function only when called)
                airmass = calculate_airmass_array(
                    ra, dec,
                    det.meta['LATITUDE'],
                    det.meta['LONGITUD'],
                    det.meta['ALTITUDE'],
                    det.meta['JD']
                )

                # Check for invalid airmass values (below horizon gives negative secz)
                invalid_mask = (airmass < 0) | (airmass > 10) | np.isnan(airmass)
                if np.any(invalid_mask):
                    print(f"Found {np.sum(invalid_mask)} invalid airmass values - setting to 1.0")
                    airmass[invalid_mask] = 1.0

        except Exception as e:
            print(f"Airmass calculation failed: {e} - using default airmass=1.0")
            airmass = np.ones_like(ra)

        # Ensure det.meta['AIRMASS'] is consistent with our airmass calculation approach
        # If we used default airmass=1.0, also set the reference airmass to 1.0
        if np.all(airmass == 1.0):
            det.meta['AIRMASS'] = 1.0  # This ensures adif = airmass - det.meta['AIRMASS'] = 0.0 everywhere

        coord_x = (det_data[:, 0] - det.meta['CTRX']) / 1024
        coord_y = (det_data[:, 1] - det.meta['CTRY']) / 1024

        filter_mags = {}

        magcat = cat_data[det.meta['PHFILTER']]
        maglim = options.maglim or det.meta['MAGLIM']
        mag_mask = magcat <= maglim
        if options.brightlim:
            mag_mask &= magcat >= options.brightlim

        # Apply color limits using max/min of first three colors from catalog
        # This targets problematic objects with extreme colors (e.g. i-z < -0.5)
        if options.redlim is not None or options.bluelim is not None:
            schema = options.filter_schemas[det.meta['PHSCHEMA']]

            # Compute colors from catalog data
            color1 = cat_data[schema[0]] - cat_data[schema[1]]  # e.g., g-r
            color2 = cat_data[schema[1]] - cat_data[schema[2]]  # e.g., r-i
            color3 = cat_data[schema[2]] - cat_data[schema[3]]  # e.g., i-z

            # Use max/min of first three colors for red/blue limits
            color_extreme = np.maximum(np.maximum(color1, color2), color3)
            color_blue = np.minimum(np.minimum(color1, color2), color3)

            # Start with current magnitude mask
            color_mask = np.ones(len(cat_data), dtype=bool)

            # Apply red limit if specified
            if options.redlim is not None:
                color_mask &= (color_extreme <= options.redlim)

            # Apply blue limit if specified
            if options.bluelim is not None:
                color_mask &= (color_blue >= options.bluelim)

            # Combine with magnitude mask
            total_before = np.sum(mag_mask)
            mag_mask &= color_mask
            total_after = np.sum(mag_mask)
            n_removed = total_before - total_after

            if n_removed > 0:
                print(f"Color filtering: {n_removed} objects removed ({n_removed/total_before*100:.1f}%)")
                if options.redlim is not None:
                    print(f"  Red limit: {options.redlim} (reddest color <= {options.redlim})")
                if options.bluelim is not None:
                    print(f"  Blue limit: {options.bluelim} (bluest color >= {options.bluelim})")

        # Exclude target object from calibration if requested
        logging.info(f"Target exclusion enabled: {options.exclude_target}, target found: {'Yes' if target_match is not None else 'No'}")
        if options.exclude_target and target_match is not None:
            idlimit = options.idlimit if options.idlimit else 2.0
            target_x, target_y = target_match['X_IMAGE'], target_match['Y_IMAGE']
            det_x, det_y = det_data[:, 0], det_data[:, 1]
            distances = np.sqrt((det_x - target_x)**2 + (det_y - target_y)**2)
            target_indices = distances < idlimit

            n_target_removed = np.sum(target_indices & mag_mask)
            if n_target_removed > 0:
                mag_mask &= ~target_indices
                print(f"Target exclusion: {n_target_removed} target object(s) excluded from calibration")
                print(f"  Target position: ({target_x:.1f}, {target_y:.1f}), search radius: {idlimit:.1f} px")

        n_matched_stars = np.sum(mag_mask)

        # Store filter data before applying magnitude mask
        for filter_name in cat.filters:
            if filter_name in options.filter_schemas[det.meta['PHSCHEMA']]:
                if filter_name in cat_data.columns:
                    # Apply magnitude mask to get filter data for matched stars only
                    filter_mags[filter_name] = cat_data[filter_name][mag_mask]

        temp_dy = det_data[:, 3][mag_mask]
        temp_dy_no_zero = np.sqrt(np.power(temp_dy,2)+0.0004)

        _dx = det_data[:, 4][mag_mask]
        _dy = det_data[:, 5][mag_mask]
        _image_dxy = np.sqrt(np.power(_dx,2) + np.power(_dy,2) + 0.000025)

        data.extend(
            y=det_data[:, 2][mag_mask],
            adif=airmass[mag_mask] - det.meta['AIRMASS'],  # Keep existing relative airmass
            airmass=airmass[mag_mask],  # Add new absolute airmass column
            coord_x=coord_x[mag_mask],
            coord_y=coord_y[mag_mask],
            img=np.full(n_matched_stars, det.meta['IMGNO']),
            dy=temp_dy_no_zero,
            image_x=det_data[:, 0][mag_mask],
            image_y=det_data[:, 1][mag_mask],
            image_dxy=_image_dxy,
            ra=cat_data['radeg'][mag_mask],
            dec=cat_data['decdeg'][mag_mask],
            cat_x=cat_x[mag_mask],  # transformed catalog X positions
            cat_y=cat_y[mag_mask],  # transformed catalog Y positions
        )

        for filter_name, mag_values in filter_mags.items():
            data.add_filter_column(filter_name, mag_values)

        if det.meta['PHFILTER'] in filter_mags:
            data.set_current_filter(det.meta['PHFILTER'])
        else:
            raise ValueError(f"Filter '{det.meta['PHFILTER']}' not found in catalog data")

        return n_matched_stars

    except KeyError as e:
        logging.error(f"Error in make_pairs_to_fit: Missing key in detection or catalog data: {e}")
    except ValueError as e:
        logging.error(f"Error in make_pairs_to_fit: Invalid value encountered: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in make_pairs_to_fit: {e}")

    return 0

def compute_initial_zeropoints(data, metadata):
    """
    Compute initial zeropoints for each image based on the selected best filter.

    Args:
    data (PhotometryData): Object containing all photometry data.
    metadata (list): List of metadata for each image.

    Returns:
    list: Initial zeropoints for each image.
    """
    zeropoints = []
    x, y, img = data.get_arrays('x', 'y', 'img')

    for img_meta in metadata:
        img_mask = img == img_meta['IMGNO']
        img_x = x[img_mask]
        img_y = y[img_mask]

        if len(img_x) > 0:
            zeropoint = np.median(img_x - img_y)  # x (catalog mag) - y (observed mag)
        else:
            logging.warning(f"No data for image {img_meta['IMGNO']}, using default zeropoint of 0")
            zeropoint = 0

        zeropoints.append(zeropoint)

    return zeropoints


def compute_zeropoints_all_filters(data, metadata, options):
    """
    Compute zeropoints for all available filters with quality metrics.
    Unified function that handles both zeropoint computation and filter discovery/validation.

    Args:
    data (PhotometryData): Object containing all photometry data.
    metadata (list): List of metadata for each image.
    options: Command line options (for filter_check mode)

    Returns:
    tuple: (zeropoints_for_final_filter, final_filter_name, filter_results_dict)
        - zeropoints_for_final_filter: list of zeropoints for the chosen filter
        - final_filter_name: name of the filter that should be used
        - filter_results_dict: detailed results for all tested filters
    """
    import logging

    # Get all available catalog filters for comprehensive testing
    catalog_name = 'makak' if getattr(options, 'makak', False) else getattr(options, 'catalog', None)
    try:
        from filter_matching import get_catalog_filters
        all_catalog_filters = list(get_catalog_filters(catalog_name).keys())
    except:
        # Fallback to loaded filters if catalog lookup fails
        all_catalog_filters = data.get_filter_columns()

    loaded_filters = data.get_filter_columns()
    filter_check_mode = getattr(options, 'filter_check', 'none')
    original_filter = data._current_filter
    results = {}

    # Determine which filters to test
    if filter_check_mode in ['n', 'none']:
        # Only test current filter for efficiency
        filters_to_test = [original_filter] if original_filter else loaded_filters[:1]
    else:
        # Test all catalog filters for discovery/validation, but only those we can actually use
        filters_to_test = [f for f in all_catalog_filters if f in loaded_filters]

        # If we don't have all filters loaded, warn the user
        missing_filters = set(all_catalog_filters) - set(loaded_filters)
        if missing_filters:
            print(f"Note: {len(missing_filters)} catalog filters not loaded during schema processing: {sorted(missing_filters)}")
            print(f"Testing {len(filters_to_test)} available filters: {sorted(filters_to_test)}")

    # Computing zeropoints for multiple filters

    for filter_name in filters_to_test:
        data.set_current_filter(filter_name)
        x, y, img = data.get_arrays('x', 'y', 'img')

        zeropoints = []
        correlations = []
        n_stars_per_image = []

        for img_meta in metadata:
            img_mask = img == img_meta['IMGNO']
            img_x = x[img_mask]
            img_y = y[img_mask]

            n_stars = len(img_x)
            n_stars_per_image.append(n_stars)

            if n_stars > 0:
                zeropoint = np.median(img_x - img_y)  # x (catalog mag) - y (observed mag)

                # Compute correlation if we have enough stars
                if n_stars >= 5:
                    # Remove outliers for better correlation
                    residuals = img_x - img_y
                    median_res = np.median(residuals)
                    mad = np.median(np.abs(residuals - median_res))
                    good_mask = np.abs(residuals - median_res) < 3 * mad

                    if np.sum(good_mask) >= 5:
                        correlation = np.corrcoef(img_x[good_mask], img_y[good_mask])[0, 1]
                        if np.isnan(correlation):
                            correlation = 0
                    else:
                        correlation = 0
                else:
                    correlation = 0
            else:
                logging.warning(f"No stars for image {img_meta['IMGNO']}, filter {filter_name}")
                zeropoint = 0
                correlation = 0

            zeropoints.append(zeropoint)
            correlations.append(correlation)

        # Compute overall quality metrics
        valid_correlations = [c for c in correlations if c > 0]
        mean_correlation = np.mean(valid_correlations) if valid_correlations else 0
        total_stars = sum(n_stars_per_image)

        results[filter_name] = {
            'zeropoints': zeropoints,
            'correlations': correlations,
            'mean_correlation': mean_correlation,
            'total_stars': total_stars,
            'n_images': len([c for c in correlations if c > 0])
        }

        print(f"Filter {filter_name}: correlation={mean_correlation:.3f}, stars={total_stars}, images={results[filter_name]['n_images']}")

    # Determine final filter based on mode
    header_filter = original_filter

    if filter_check_mode in ['n', 'none']:
        final_filter = header_filter
        logging.info(f"Using header filter {final_filter} (no validation)")

    elif filter_check_mode in ['d', 'discover']:
        # Pick filter with best correlation
        best_filter = max(results.keys(), key=lambda f: results[f]['mean_correlation'])
        final_filter = best_filter

        if best_filter != header_filter:
            logging.info(f"Filter discovery: using {best_filter} (correlation={results[best_filter]['mean_correlation']:.3f}) instead of header {header_filter}")
            print(f"Filter discovery: using {best_filter} (header was {header_filter})")
        else:
            logging.info(f"Filter discovery confirmed header filter: {header_filter}")

    else:
        # Validation modes (warn/strict)
        best_filter = max(results.keys(), key=lambda f: results[f]['mean_correlation'])

        if best_filter == header_filter:
            logging.info(f"Filter validation passed: {header_filter}")
            final_filter = header_filter
        else:
            message = f"Filter mismatch: header={header_filter} (corr={results[header_filter]['mean_correlation']:.3f}), best={best_filter} (corr={results[best_filter]['mean_correlation']:.3f})"

            if filter_check_mode in ['w', 'warn']:
                logging.warning(message)
                print(f"WARNING: {message}")
                final_filter = header_filter  # Continue with header
            elif filter_check_mode in ['s', 'strict']:
                logging.error(message)
                print(f"ERROR: {message}")
                print("Use --filter-check=none to override or fix the filter in FITS header")
                import sys
                sys.exit(1)

    # Restore final filter and return its zeropoints
    data.set_current_filter(final_filter)
    final_zeropoints = results[final_filter]['zeropoints']

    return final_zeropoints, final_filter, results

