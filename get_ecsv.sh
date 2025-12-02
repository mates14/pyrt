#!/bin/bash

TEMP=/tmp/get_ecsv_$$

for i in $*; do

        j=`basename $i`
        k=`echo $j|cut -c 1-6`
        echo $k/${j%.fits}.ecsv
        if [ -e $k/${j%-RA.fits}* ]; then
                continue
                fi

        eval `fitsheader -r CCD_NAME $i`
        eval `fitsheader -r CTIME $i`
        year=`date +%Y -d@$CTIME`

        mkdir $TEMP
        pushd $TEMP
        proc_images.py /home/mates/flat$year/$CCD_NAME/*.fits $i
        df=`ls -t *.fits | head -n1`
        base=${df%.fits}
        
        mv $df ${base}c.fits
        case $CCD_NAME in
            C[12])
            imcopy ${base}c.fits[30:4127,11:4108] $df
            ;;
            C3)
            imcopy ${base}c.fits[35:2218,5:1476] $df
            ;;
            *)
            imcopy ${base}c.fits $df
            ;;
        esac
        rm ${base}c.fits
        ls -l $df
       
        solve-field -pT $df
        if [ -e ${base}.new ]; then
            mv ${base}.new $df
        else
            echo Astrometry redo failed, no guarantee of success
        fi
 
        sscat-noradec $df
        pyrt-cat2det $df

#        rm $df.xat
        
        eval `fitsheader -r CTIME $df`
        year=`date +%Y%m -d@$CTIME`

        pyrt-dophot -av -i10 -U PR,PX,PY,P2R,P2Y,PXY,P2XY,PX2Y,P3X,P3Y -M sbt $df -l12 -L11
        ln -s $df.xat ${base}t.fits.xat
# sscat-noradec ${base}t.fits
        pyrt-cat2det ${base}t.fits
        pyrt-dophot -sv -U PR,PX,PY,P2R,P2Y,PXY,P2XY,PX2Y,P3X,P3Y -M sbt ${base}t.fits -l12 -L11

#       rm $df ${df%fits}det ${df%.fits}t.fits ${df%.fits}t.det ${df%.fits}tt.fits ${df%.fits}t.fits.xat *.reg astmodel.ecsv
        rm $df ${base}.det *.reg ${base}.axy ${base}.corr ${base}.fits.xat ${base}-indx.png ${base}-indx.xyls ${base}.match ${base}-ngc.png ${base}-objs.png ${base}.rdls ${base}.solved ${base}t.det ${base}t.fits ${base}t.fits.xat ${base}.wcs $base.ecsv astmodel.ecsv


        popd
        mkdir -p $year
        mv $TEMP/${base}t.ecsv ./$year
        cat $TEMP/stars >> stars
        rm $TEMP/stars
        rmdir $TEMP

        done
