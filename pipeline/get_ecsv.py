#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from datetime import datetime

from astropy.io import fits
from proc_images import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PhotometryPipeline:
    """Complete photometry pipeline from raw images to ECSV catalogs."""

    def __init__(self,
                 phdb_root: str = "~/phdb",
                 png_root: str = "~/png",
                 dophot_path: str = "pyrt-dophot",
                 calib_dir_template: str = "/home/mates/flat{year}/",
                 verbose: bool = False):

        self.phdb_root = Path(phdb_root).expanduser()
        self.png_root = Path(png_root).expanduser()
        self.dophot_path = Path(dophot_path).expanduser()
        self.calib_dir_template = calib_dir_template
        self.verbose = verbose

        # Initialize image processor
        self.image_processor = ImageProcessor(calib_dir_template=calib_dir_template)

    def _get_header_value(self, filepath: str, keyword: str):
        """Extract header value using fitsheader equivalent."""
        try:
            with fits.open(filepath) as hdul:
                return hdul[0].header[keyword]
        except Exception as e:
            logger.error(f"Error reading {keyword} from {filepath}: {e}")
            return None

    def _get_year_month_code(self, ctime: float) -> str:
        """Convert CTIME to YYMM format."""
        dt = datetime.fromtimestamp(ctime)
        return dt.strftime("%y%m")

    def _check_existing_ecsv(self, image_path: str) -> Optional[str]:
        """Check if ECSV already exists for this image."""
        basename = Path(image_path).name

        # Extract CTIME to determine year-month directory
        ctime = self._get_header_value(image_path, 'CTIME')
        if ctime is None:
            return None

        ym = self._get_year_month_code(ctime)
        ecsv_dir = self.phdb_root / ym

        # Look for existing ECSV files (remove -RA.fits suffix if present)
        base_name = basename.replace('-RA.fits', '').replace('.fits', '')

        if ecsv_dir.exists():
            pattern = f"{base_name}*.ecsv"
            matching_files = list(ecsv_dir.glob(pattern))
            if matching_files:
                return str(matching_files[0])

        return None

    def _run_step(self, cmd: List[str], temp_dir: Path, step_name: str) -> bool:
        """Run a pipeline step, streaming output when verbose, capturing otherwise."""
        logger.info(f"--- {step_name} ---")
        if self.verbose:
            result = subprocess.run(cmd, cwd=temp_dir, text=True)
        else:
            result = subprocess.run(cmd, cwd=temp_dir, text=True,
                                    capture_output=True)
        if result.returncode != 0:
            logger.error(f"{step_name} failed (exit {result.returncode})")
            if not self.verbose and result.stderr:
                logger.error(result.stderr.rstrip())
            return False
        return True

    def _run_dophot(self, fits_file: str, temp_dir: Path,
                    iterations: int = 1) -> Optional[str]:
        """Run complete photometry pipeline on calibrated image.

        Steps:
          1. pyrt-phcat       <file.fits>
          2. pyrt-field-solve <file.fits>
          3. pyrt-cat2det     <file.fits>
          4. pyrt-dophot      <file.det>   (initial solution)
          5+. pyrt-dophot     <file.ecsv>  (repeated `iterations` times)
        """
        stem = fits_file.replace('.fits', '')
        det_file  = stem + '.det'
        ecsv_file = stem + '.ecsv'

        # Step 1: phcat
        if not self._run_step(["pyrt-phcat", fits_file], temp_dir, "pyrt-phcat"):
            return None

        # Step 2: field-solve
        if not self._run_step(["pyrt-field-solve", fits_file], temp_dir, "pyrt-field-solve"):
            return None

        # Step 3: cat2det
        if not self._run_step(["pyrt-cat2det", fits_file], temp_dir, "pyrt-cat2det"):
            return None

        # Step 4: initial dophot on .det  (-i2: identification limit, safer than -i5)
        if not self._run_step(
            [str(self.dophot_path), "-azS2", "-U", ".r5,.p5,.l", "--max-stars", "500", "-i2", det_file],
            temp_dir, "pyrt-dophot (initial)"
        ):
            return None

        # Step 5+: refinement iterations on .ecsv
        for i in range(iterations):
            if not self._run_step(
                [str(self.dophot_path), "-azS2", "-U", ".r3,.p5,.l", "--max-stars", "500", ecsv_file],
                temp_dir, f"pyrt-dophot (iter {i+1}/{iterations})"
            ):
                return None

        # Verify output
        if not (temp_dir / ecsv_file).exists():
            logger.error(f"ECSV file {ecsv_file} was not created")
            return None

        return ecsv_file

    def _check_astsigma(self, ecsv_path: Path, max_astsigma: float) -> bool:
        """Return True if ASTSIGMA < max_astsigma arcsec (read from ECSV metadata)."""
        try:
            from astropy.table import Table
            meta = Table.read(str(ecsv_path), format='ascii.ecsv').meta
            astsigma = meta.get('ASTSIGMA')
        except Exception as e:
            logger.warning(f"Could not read ASTSIGMA from {ecsv_path.name}: {e}")
            return False
        if astsigma is None:
            logger.warning(f"ASTSIGMA not found in {ecsv_path.name}")
            return False
        if astsigma >= max_astsigma:
            logger.warning(f"ASTSIGMA={astsigma:.3f}\" >= {max_astsigma}\" — frame rejected")
            return False
        logger.info(f"ASTSIGMA={astsigma:.3f}\" OK")
        return True

    def process_image(self, image_path: str, force: bool = False,
                      keep_image: bool = False,
                      iterations: int = 1,
                      min_astsigma: Optional[float] = None
                      ) -> Optional[Tuple[str, Optional[str]]]:
        """Process a single image through the complete pipeline.

        Args:
            image_path:    Path to input FITS image
            force:         Redo processing even if ECSV exists
            keep_image:    Keep calibrated FITS file after processing
            iterations:    Number of pyrt-dophot refinement passes on the ECSV
            min_astsigma:  Reject frames with ASTSIGMA >= this value (arcsec).
                           None disables the check.

        Returns:
            (ecsv_path, fits_path) on success; fits_path is None when not kept.
            None if the frame failed or was rejected.
        """
        image_path = Path(image_path)
        logger.debug(f"××××× {image_path.name} ×××××")

        # Check if result already exists
        if not force:
            existing_ecsv = self._check_existing_ecsv(str(image_path))
            if existing_ecsv:
                logger.info(f"Result already exists for {image_path.name}")
                logger.info(f"ECSV={existing_ecsv}")
                return existing_ecsv, None

        # Get CTIME for output directory naming
        ctime = self._get_header_value(str(image_path), 'CTIME')
        if ctime is None:
            logger.error(f"Cannot read CTIME from {image_path}")
            return None

        ym = self._get_year_month_code(ctime)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Calibrate image
            image_processor = ImageProcessor(calib_dir_template=self.image_processor.calib_dir_template)
            image_processor.load_calibration_frames([str(image_path)])

            processed_files = image_processor.process_all_objects(
                output_dir=str(temp_path), overwrite=True
            )

            if not processed_files:
                logger.error("No output from image processing")
                return None

            processed_file = Path(processed_files[0]).name
            logger.info(f"Running photometry for: {processed_file}")

            # Remove any stale ECSV files in temp directory
            for f in temp_path.glob("*.ecsv"):
                f.unlink()

            # Run the full pipeline
            ecsv_filename = self._run_dophot(processed_file, temp_path, iterations)
            if ecsv_filename is None:
                return None

            # Quality gate: check astrometric scatter from ECSV metadata
            if min_astsigma is not None:
                if not self._check_astsigma(temp_path / ecsv_filename, min_astsigma):
                    logger.error(
                        f"Sorry, {image_path.name} failed astrometry check "
                        f"(ASTSIGMA >= {min_astsigma}\") — discarding frame"
                    )
                    return None

            # Create output directories
            phdb_dir = self.phdb_root / ym
            png_dir   = self.png_root  / ym
            phdb_dir.mkdir(parents=True, exist_ok=True)
            png_dir.mkdir(parents=True,  exist_ok=True)

            # Move ECSV to phdb directory
            src_ecsv = temp_path / ecsv_filename
            dst_ecsv = phdb_dir  / ecsv_filename
            shutil.move(str(src_ecsv), str(dst_ecsv))
            logger.debug(f"  ECSV: {dst_ecsv}")

            # Move PNG files
            for png_file in temp_path.glob("*.png"):
                shutil.move(str(png_file), str(png_dir / png_file.name))

            dst_fits = None
            if keep_image:
                dft_file = processed_file.replace('-df.fits', '-dft.fits')
                dft_path = temp_path / dft_file
                if dft_path.exists():
                    dst_fits = Path.cwd() / dft_file
                    shutil.copy2(str(dft_path), str(dst_fits))
                    logger.info(f"Kept calibrated image: {dst_fits}")
                else:
                    logger.warning(f"Expected -dft.fits not found: {dft_file}")

            return str(dst_ecsv), str(dst_fits) if dst_fits else None

    def process_images(self, image_paths: List[str], force: bool = False,
                       keep_image: bool = False,
                       iterations: int = 1,
                       min_astsigma: Optional[float] = None) -> List[Tuple]:
        """Process multiple images through the pipeline.

        Returns:
            List of (ecsv_path, fits_path) tuples for successfully processed images.
        """
        successful = []

        for image_path in image_paths:
            try:
                result = self.process_image(image_path, force, keep_image,
                                            iterations, min_astsigma)
                if result:
                    successful.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

        logger.info(f"Successfully processed {len(successful)}/{len(image_paths)} images")
        return successful


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Complete photometry pipeline: calibration + dophot3"
    )
    parser.add_argument('images', nargs='+', help='Input FITS images')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Redo the ECSV output even if it exists')
    parser.add_argument('-i', '--keep-image', action='store_true',
                        help='Keep calibrated FITS file after processing')
    parser.add_argument('-n', '--iterations', type=int, default=1,
                        metavar='N',
                        help='Number of pyrt-dophot refinement passes on the ECSV (default: 1)')
    parser.add_argument('--min-astsigma', type=float, default=None,
                        metavar='ARCSEC',
                        help='Reject frames with ASTSIGMA >= this value in arcsec '
                             '(e.g. 0.2); disabled by default')
    parser.add_argument('--phdb-root', default='~/phdb',
                        help='Root directory for photometry database (default: ~/phdb)')
    parser.add_argument('--png-root', default='~/png',
                        help='Root directory for PNG files (default: ~/png)')
    parser.add_argument('--dophot-path', default='pyrt-dophot',
                        help='Path to pyrt-dophot (default: pyrt-dophot from PATH)')
    parser.add_argument('--calib-dir', default='/home/mates/flat{year}/',
                        help='Calibration directory template (default: /home/mates/flat{year}/)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    pipeline = PhotometryPipeline(
        phdb_root=args.phdb_root,
        png_root=args.png_root,
        dophot_path=args.dophot_path,
        calib_dir_template=args.calib_dir,
        verbose=args.verbose,
    )

    pipeline.process_images(
        args.images,
        force=args.force,
        keep_image=args.keep_image,
        iterations=args.iterations,
        min_astsigma=args.min_astsigma,
    )


if __name__ == "__main__":
    main()
