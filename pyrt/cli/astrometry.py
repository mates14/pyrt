#!/usr/bin/python3
# (C) 2010, Markus Wildi, markus.wildi@one-arcsec.org
# (C) 2011-2012, Petr Kubanek, Institute of Physics <kubanek@fzu.cz>
#  usage 
#  astrometry.py fits_filename
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2, or (at your option)
#  any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#  Or visit http://www.gnu.org/licenses/gpl.html.
#

__author__ = 'kubanek@fzu.cz'

import os
import shutil
import string
import subprocess
import sys
import re
import time
import astropy.io.fits as pyfits
import tempfile
import numpy
import math

import dms
#from kapteyn import wcs

#ast_scales=[4,5,6,8,10,12,14]
ast_scales=[1,2,3,4] # ,5,6,8,10,12,14]

class WCSAxisProjection:
    def __init__(self,fkey):
        self.wcs_axis = None
        self.projection_type = None
        self.sip = False

        for x in fkey.split('-'):
            if x == 'RA' or x == 'DEC':
                self.wcs_axis = x
            elif x == 'TAN':
                self.projection_type = x
            elif x == 'SIP':
                self.sip = True
        if self.wcs_axis is None or self.projection_type is None:
            raise Exception('uknown projection type {0}'.format(fkey))

def xy2wcs(x, y, fitsh):
    """Transform XY pixel coordinates to WCS coordinates"""

    proj = wcs.Projection(fitsh)
    (ra, dec) = proj.toworld((x, y))
    return [ra, dec]

#    wcs1 = WCSAxisProjection(fitsh['CTYPE1'])
#    wcs2 = WCSAxisProjection(fitsh['CTYPE2'])
#    # retrieve CD matrix
#    cd = numpy.array([[fitsh['CD1_1'],fitsh['CD1_2']],[fitsh['CD2_1'],fitsh['CD2_2']]])
#    # subtract reference pixel
#    xy = numpy.array([x,y]) - numpy.array([fitsh['CRPIX1'],fitsh['CRPIX2']])
#    xy = numpy.dot(cd,xy)
#
#    if wcs1.wcs_axis == 'RA' and wcs2.wcs_axis == 'DEC':
#        dec = xy[1] + fitsh['CRVAL2']
#        if wcs1.projection_type == 'TAN':
#            if abs(dec) != 90:
#                xy[0] /= math.cos(math.radians(dec))
#        return [xy[0] + fitsh['CRVAL1'],dec]
#
#    if wcs1.wcs_axis == 'DEC' and wcs2.wcs_axis == 'RA':
#        dec = xy[0] + fitsh['CRVAL1']
#        if wcs2.projection_type == 'TAN':
#            if abs(dec) != 90:
#                xy[1] /= math.cos(math.radians(dec))
#        return [xy[1] + fitsh['CRVAL2'],dec]
#    raise Exception('unsuported axis combination {0} {1}'.format(wcs1.wcs_axis,wcs2.wcs_axis))

class AstrometryScript:
    def __init__(self, odir=None, scale_relative_error=0.25, astrometry_bin='/usr/bin', zoom=1.0, poly=2, time=30):
        """initialize the registration pipeline"""
        self.scale_relative_error = scale_relative_error
        self.astrometry_bin = astrometry_bin
        self.zoom = zoom
        self.poly = poly
        self.time = time
        self.odir = odir

    def run(self, fits_file, scale=None, ra=None, dec=None, replace=False, naxis1=None, naxis2=None, zoom=None):
        """Run the registration pipeline"""
        
        # Use context manager for temporary directory
        with tempfile.TemporaryDirectory(suffix='', prefix='astrometry.') as odir:

            infpath = os.path.join(odir, 'input.fits')
            shutil.copy(fits_file, infpath)

            #solve_field=[self.astrometry_bin + '/solve-field', '-D', odir,'--no-plots', '--no-fits2fits']
            solve_field=[self.astrometry_bin + '/solve-field', '-D', odir,'--no-plots']


            if zoom is None:
                zoom = self.zoom
            
            if scale is not None:
                scale_low=scale*(1-self.scale_relative_error)
                scale_high=scale*(1+self.scale_relative_error)
                solve_field.append('-u')
                solve_field.append('app')
                solve_field.append('-L')
                solve_field.append(str(scale_low))
                solve_field.append('-H')
                solve_field.append(str(scale_high))
            
            # stupen polynomu pro fit, standardni 2 pro WF snimky nekdy nestacil
            # sergej: --order 4 --radius 1 -z 4 --scale-error 1
    #        solve_field.append('--scale-error 1')
    #        solve_field.append('--radius 1')
    #        solve_field.append('-t%d'%self.poly)

            solve_field.append('-T')
            solve_field.append('-l%d'%self.time)
            solve_field.append('-z%d'%zoom)
            solve_field.append('-y')
            solve_field.append('--uniformize')
            solve_field.append('10')

    # This produces problematic results (bad precision):
    #        if naxis1 is not None and naxis2 is not None:
    #            solve_field.append('--crpix-x')
    #            solve_field.append(str(naxis1/2))
    #            solve_field.append('--crpix-y')
    #            solve_field.append(str(naxis2/2))

    # This is virtually useless apart of speed (if it does not work when it should, it helps not):
    #        if ra is not None and dec is not None:
    #            solve_field.append('--ra')
    #            solve_field.append(str(ra))
    #            solve_field.append('--dec')
    #            solve_field.append(str(dec))
    #            solve_field.append('--radius')
    #            solve_field.append('15')

            solve_field.append(infpath)

            #print(solve_field)
          
            print(solve_field)

            proc = subprocess.Popen(solve_field, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            radecline=re.compile(r'Field center: \(RA H:M:S, Dec D:M:S\) = \(([^,]*),(.*)\).')

            ret = None

            while True:
                a=proc.stdout.readline().decode("utf-8")
                if a == '':
                    break
            #    print(a)
                match=radecline.match(a)
                if match:
                    ret=[dms.parseDMS(match.group(1)), dms.parseDMS(match.group(2))]
            
            # Always overwrite the original file if .new file exists and solution was found
            new_file = os.path.join(odir, 'input.new')
            if ret is not None and os.path.exists(new_file):
                shutil.move(new_file, fits_file)
            
            # Temporary directory cleanup happens automatically via context manager
            
        return ret

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: %s <fits filename>' % (sys.argv[0]))
        sys.exit(1)

    a=AstrometryScript(time=15)

    for zoom in ast_scales:
        ret=a.run(sys.argv[1],zoom=zoom,replace=True)
        if ret is not None:
            break

    print("astrometry:",ret,zoom)
    exit(0)

    ra = dec = None

    ff=pyfits.open(sys.argv[1],'readonly')
    naxis1=ff[0].header['NAXIS1']
    naxis2=ff[0].header['NAXIS2']

    try:
        ra=ff[0].header['OBJRA']
        dec=ff[0].header['OBJDEC']
        obj=ff[0].header['OBJECT']
    except:
        ra=None
        dec=None
        obj=""

    try:
        num=ff[0].header['IMGID']
    except:
        num=0

    try:
        scale=ff[0].header['CDELT1']*3600.0
    except:
        scale=1.18

    ff.close()

#      ret=a.run(scale=1.18,ra=ra,dec=dec,replace=True)
# Kdyz se pouzije natvrdo, nefunguje binning, kdyz se da none, na pokazenych snimkach zatuhne na dlouho... Resenim je vypocitat scale z hlavicky z CDELT1:
    ret=a.run(sys.argv[1],scale=scale,ra=ra,dec=dec,replace=True,naxis1=naxis1,naxis2=naxis2)

    if ret:
        raorig=ra
        decorig=dec

        ff=pyfits.open(sys.argv[1],'readonly')

        # spectroscopic centering
        try:
            xpos_slit=ff[0].header['slitposx']
        except:
            xpos_slit=ff[0].header['naxis1']/2
        
        if xpos_slit<0:
            xpos_slit=ff[0].header['naxis1']/2

        try:
            ypos_slit=ff[0].header['slitposy']
        except:
            ypos_slit=ff[0].header['naxis2']/2
        
        if ypos_slit<0:
            ypos_slit=ff[0].header['naxis2']/2
        
        #c.log('I',"astrometry: centering to {0}x{1}".format(xpos_slit,ypos_slit))

        try:
          from kapteyn import wcs
          import pynova

          rastrxy = xy2wcs(xpos_slit, ypos_slit, ff[0].header)
          err = pynova.angularSeparation(raorig,decorig,rastrxy[0],rastrxy[1])
        except:
          c=0
        
        ff.close()


        try:
            import rts2comm
            c = rts2comm.Rts2Comm()

            c.log('I',"corrwerr 1 {0:.10f} {1:.10f} {2:.10f} {3:.10f} {4:.10f}".format(rastrxy[0], rastrxy[1], raorig-rastrxy[0], decorig-rastrxy[1], err))
            print("corrwerr 1 {0:.10f} {1:.10f} {2:.10f} {3:.10f} {4:.10f}".format(rastrxy[0], rastrxy[1], raorig-rastrxy[0], decorig-rastrxy[1], err))

            c.doubleValue('real_ra','[hours] image ra as calculated from astrometry',rastrxy[0])
            c.doubleValue('real_dec','[deg] image dec as calculated from astrometry',rastrxy[1])

            c.doubleValue('tra','[hours] telescope ra',raorig)
            c.doubleValue('tdec','[deg] telescope dec',decorig)

            c.doubleValue('ora','[arcdeg] offsets ra ac calculated from astrometry',raorig-rastrxy[0])
            c.doubleValue('odec','[arcdeg] offsets dec as calculated from astrometry',decorig-rastrxy[1])
            
            c.doubleValue('slitposx','[pixel] x position of target on the CCD for centering',xpos_slit)
            c.doubleValue('slitposy','[pixel] y position of target on the CCD for centering',ypos_slit)

    #        c.stringValue('object','astrometry object',obj)
            c.integerValue('img_num','last astrometry number',num)

        except:
            print("%s done"%(sys.argv[1]))
