#!/bin/bash

cat > default-$$.sex <<EOF
CATALOG_TYPE     ASCII_HEAD
VERBOSE_TYPE     QUIET
PARAMETERS_NAME  default-$$.param
FILTER_NAME      default-$$.conv
EOF

cat > default-$$.param <<EOF
NUMBER
ALPHA_J2000
DELTA_J2000
MAG_AUTO
MAGERR_AUTO
X_IMAGE
Y_IMAGE
ERRX2_IMAGE
ERRY2_IMAGE
FWHM_IMAGE
ELLIPTICITY
FLAGS
EOF

cat > default-$$.conv <<EOF
CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
EOF

mypath=`dirname $0`

for file in $*
do

        i=${file%.fits}

        if [ -e $mypath/sex ]; then SEX=$mypath/sex; else SEX=sex; fi

        $SEX $file -CATALOG_NAME $i.cat -c default-$$.sex

done

rm default-$$.*
