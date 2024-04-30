#!/bin/bash

MAG_LIMIT=""
SET_APERT=""
SKIP="no"

while getopts "l:a:s" optn; do
case $optn in
    l)
        MAG_LIMIT="-l $OPTARG"; 
        ;;
    a)
        SET_APERT="-a $OPTARG"; 
        ;;
    s)
        SKIP="yes"; 
        ;;
esac
done
shift $(($OPTIND - 1))

if [ $SKIP = "no" ]; then
        ls $* |xargs -n1 -P8 sscat $SET_APERT
        ls $* |xargs -n1 -P8 ~/pyrt/cat2det.py

        for x in $*; do
            test -e $x.xat && rm $x.xat
        done
fi

> .filter$$.tmp
> .allfiles$$.tmp
for arg in $*; do
        case $arg in
                *.fits)
                        g=${arg%.fits}.det
                        ;;
                *)
                        g=$arg
                        ;;
        esac
        eval `fitsheader -r FILTER $g[1]`
        echo $FILTER >> .filter$$.tmp
        echo $g >> .allfiles$$.tmp
done

for filtsel in `cat .filter$$.tmp | sort | uniq`; do
    > .files$$.tmp
    for g in `cat .allfiles$$.tmp`; do

        eval `fitsheader -r FILTER $g[1]`

        if [ $filtsel == $FILTER ]; then
                echo $g >> .files$$.tmp
                fi
        done
    cat .files$$.tmp | xargs ~/pyrt/dophot3.py -vs $MAG_LIMIT |tee $filtsel.out
    mv stars stars.$f
    done

rm .filter$$.tmp
rm .files$$.tmp
rm .allfiles$$.tmp

