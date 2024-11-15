#!/usr/bin/python3

import argparse
import configparser
import os
from collections import OrderedDict
import logging

# Default configuration including filter schemas
DEFAULT_CONFIG = """
[DEFAULT]
astrometry = False
basemag = Sloan_r
verbose = False
autoupdate = False
fsr = False
guessbase = False
johnson = False
makak = False
select_best = False
stars = False
gain = 2.3

[FILTER_SCHEMAS]
JohnsonU = Johnson_U, Johnson_B, Johnson_V, Johnson_R, Johnson_I
JohnsonJ = Johnson_B, Johnson_V, Johnson_R, Johnson_I, J
SloanU = Sloan_u, Sloan_g, Sloan_r, Sloan_i, Sloan_z
SloanJ = Sloan_g, Sloan_r, Sloan_i, Sloan_z, J
Gaia = BP, G, RP, BP, RP
PS = g, r, i, z, y
USNO = B2, R2, I, R1, B1
"""

DEFAULT_CONFIG_FILE = '~/.config/dophot3/config'

def load_config(config_file: str = None):
    """
    Load configuration, properly merging [DEFAULT] sections from both built-in defaults
    and the config file to preserve all keywords.

    :param config_file: Optional path to user config file
    :return: Dictionary containing configuration options including all DEFAULT keywords
    """
    # Load built-in defaults
    default_config = configparser.ConfigParser(default_section=None)
    default_config.read_string(DEFAULT_CONFIG)

    # Create a new config parser for the file config
    file_config = configparser.ConfigParser(default_section=None)

    # Load either the default config file or user-specified file
    if config_file:
        config_path = os.path.expanduser(config_file)
    else:
        config_path = os.path.expanduser(DEFAULT_CONFIG_FILE)

    if os.path.exists(config_path):
        file_config.read(config_path)

    # Start with all keywords from the built-in DEFAULT section
    result = dict(default_config['DEFAULT'])

    # Add all keywords from the file's DEFAULT section, overriding the defaults if they exist
    if 'DEFAULT' in file_config:
        for key, value in file_config['DEFAULT'].items():
            result[key] = value

    # Handle FILTER_SCHEMAS separately
    result['filter_schemas'] = OrderedDict()

    # First add schemas from default config
    if 'FILTER_SCHEMAS' in default_config:
        for schema_name, filters in default_config['FILTER_SCHEMAS'].items():
            result['filter_schemas'][schema_name] = [f.strip() for f in filters.split(',')]

    # Then override/add schemas from file config
    if 'FILTER_SCHEMAS' in file_config:
        for schema_name, filters in file_config['FILTER_SCHEMAS'].items():
            result['filter_schemas'][schema_name] = [f.strip() for f in filters.split(',')]

    return result

def parse_arguments(args=None):
    """
    Parse command-line arguments, integrating with config file options.

    :param args: Command line arguments (if None, sys.argv is used)
    :return: Namespace object containing all configuration options
    """
    # First, we'll create a parser just for the config file argument
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_FILE,
                             help="Specify config file", metavar="FILE")
    conf_args, remaining_argv = conf_parser.parse_known_args(args)

    # Now we can load the config file
    config = load_config(conf_args.config)

    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Compute photometric calibration for FITS images",
        # Inherit options from config_parser
        parents=[conf_parser]
    )

    # Add arguments, using config values as defaults
    parser.add_argument("-a", "--astrometry", action="store_true", default=config.get('astrometry', 'False'), \
                        help="Refit astrometric solution using photometry-selected stars")
    parser.add_argument("-A", "--aterms", default=config.get('aterms'), help="Terms to fit for astrometry")
    parser.add_argument("--usewcs", default=config.get('usewcs'), help="Use this astrometric solution (file with header)")
    parser.add_argument("-b", "--basemag", default=config.get('basemag', None),
                        help="ID of the base filter to be used while fitting (def=\"Sloan_r\"/\"Johnson_V\")")
    parser.add_argument("-C", "--catalog", default=config.get('catalog'), help="Use this catalog as a reference")
    parser.add_argument("-d", "--date", action='store', help="what to put into the third column (char,mid,bjd), default=mid")
    parser.add_argument("-e", "--enlarge", type=float, default=config.get('enlarge'), help="Enlarge catalog search region")
    parser.add_argument("-f", "--filter", default=config.get('filter'), help="Override filter info from fits")
    parser.add_argument("--fsr", help="Use forward stepwise regression", default=config.get('fsr', 'False') )
    parser.add_argument("--fsr-terms", help="Terms to be used to do forward stepwise regression", default=config.get('fsr_terms', None) )
    parser.add_argument("-F", "--flat", help="Produce flats", action='store_true')
    parser.add_argument("-g", "--guessbase", action="store_true", default=config.get('guessbase', 'False'),
                        help="Try and set base filter from fits header (implies -j if Bessel filter is found)")
    parser.add_argument("-j", "--johnson", action="store_true", default=config.get('johnson', 'False'),
                        help="Use Stetson Johnson/Cousins filters and not SDSS")
    parser.add_argument("-X", "--tryflt", action='store_true', help="Try different filters (broken)")
    parser.add_argument("-G", "--gain", action='store', help="Provide camera gain", type=float, default=config.get('gain', 2.3))
    parser.add_argument("-i", "--idlimit", help="Set a custom idlimit", type=float, default=config.get('idlimit'))
    parser.add_argument("-k", "--makak", help="Makak tweaks", action='store_true', default=config.get('makak','False'))
    parser.add_argument("-R", "--redlim", help="Do not get stars redder than this g-r", type=float, default=config.get("redlim"))
    parser.add_argument("-B", "--bluelim", help="Do not get stars bler than this g-r", type=float, default=config.get("bluelim"))
    parser.add_argument("-l", "--maglim", help="Do not get stars fainter than this limit", type=float, default=config.get("maglim"))
    parser.add_argument("-L", "--brightlim", help="Do not get any less than this mag from the catalog to compare", type=float)
    parser.add_argument("-m", "--margin", help="Modify the magnitude margin for star selection (default=2.0)", type=float, default=config.get("margin",2.0))
    parser.add_argument("-M", "--model", help="Read model from a file", type=str)
    parser.add_argument("-n", "--nonlin", help="CCD is not linear, apply linear correction on mag", action='store_true')
    parser.add_argument("-p", "--plot", help="Produce plots", action='store_true')
    parser.add_argument("-r", "--reject", help="No outputs for Reduced Chi^2 > value", type=float)
    parser.add_argument("--select-best", action='store_true', default=config.get('select_best', None), help="Try to select the best filter for photometric fitting")
    parser.add_argument("-s", "--stars", action='store_true', default=config.get('stars', 'False'), help="Output fitted numbers to a file")
    parser.add_argument("-S", "--sip", help="Order of SIP refinement for the astrometric solution (0=disable)", type=int)
    parser.add_argument("-t", "--fit-terms", help="Comma separated list of terms to fit", type=str)
    parser.add_argument("-T", "--trypar", type=str, help="Terms to examine to see if necessary (and include in the fit if they are)")
    parser.add_argument("-u", "--autoupdate", action='store_true', help="Update .det if .fits is newer", default=config.get('autoupdate', 'False'))
    parser.add_argument("-U", "--terms", help="Terms to fit", type=str, default=config.get('terms', ''))
    parser.add_argument("-w", "--weight", action='store_true', help="Produce weight image")
    parser.add_argument("-W", "--save-model", help="Write model into a file", type=str)
    parser.add_argument("-x", "--fix-terms", help="Comma separated list of terms to keep fixed", type=str)
    parser.add_argument("-y", "--fit-xy", action='store_true', help="Fit xy tilt for each image separately (i.e. terms PX/PY)")
    parser.add_argument("-z", "--refit-zpn", action='store_true', help="Refit the ZPN radial terms")
    parser.add_argument("-Z", "--szp", action='store_true', help="use SZP while fitting astrometry")

    parser.add_argument("-v", "--verbose", action="store_true",
                        default=config.get('verbose', 'False'), help="Print debugging info")
#   parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
    parser.add_argument("files", nargs='+', help="Frames to process")

    # Parse remaining arguments
    args = parser.parse_args(remaining_argv)

    # Add the filter schemas to the parsed arguments
    args.filter_schemas = config['filter_schemas']

    # Convert string 'True'/'False' to boolean for action="store_true" arguments
    for arg in ['astrometry', 'guessbase', 'johnson', 'verbose', 'makak', 'fsr', 'select_best']:
        setattr(args, arg, str(getattr(args, arg)).lower() == 'true')

    return args

# Example usage
if __name__ == "__main__":
    options = parse_arguments()
    print(options)
