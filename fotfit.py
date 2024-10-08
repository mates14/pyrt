#!/usr/bin/python3
"""
fotfit.py
Photometric response fitter
(c) Martin Jelinek, ASU AV CR, 2023
"""

import numpy as np
import termfit

# wishlist :)
#from scipy.interpolate import RegularGridInterpolator

rad = np.pi/180.0

class fotfit(termfit.termfit):
    """Photometric response fitter
    """

    modelname = "FotFit photometric model"
    fit_xy = False
    zero = []
    px = []
    py = []

    def __init__(self, proj=None, file=None, fit_xy=False):
        """Start myself up"""

        if file is not None:
            self.readmodel(file)
            self.fixall()

        self.fit_xy = fit_xy

    def __str__(self):
        output = " img --zeropoint--- ------px------ ------py------ \n"

        Nt = len(self.fitterms)
        
        if self.fit_xy: # xcoord/ycoord/zeropoint for each image
            N = np.int64((len(self.fitvalues) - len(self.fitterms)) / 3 )
        else:
            N = np.int64(len(self.fitvalues) - len(self.fitterms))

        for i in range(0,N):
            try: error = self.fiterrors[ Nt + i ]
            except IndexError: error = np.nan
            output += " %03d %6.3f ± %-5.3f "%(i, self.fitvalues[ Nt + i ], error)  # zero
            if self.fit_xy:
                try: error = self.fiterrors[ Nt + N + i ]
                except IndexError: error = np.nan
                output += "%6.3f ± %-5.3f "%(self.fitvalues[ Nt + N + i ], error)  # px
                try: error = self.fiterrors[ Nt + 2*N + i ]
                except IndexError: error = np.nan
                output += "%6.3f ± %-5.3f "%(self.fitvalues[ Nt + 2*N + i ], error)  # py
            output += "\n"

        output += termfit.termfit.__str__(self)
        
        return output;

    def fit(self, data):

        self.fitvalues = self.fitvalues[0:(len(self.fitterms))] + self.zero  
        if self.fit_xy:
            self.fitvalues = np.append(self.fitvalues, np.zeros(2*len(self.zero)))
        ret = termfit.termfit.fit(self, data)        
        return ret

    def zero_val(self):
        p = len(self.fitterms)
        return self.fitvalues[p:], self.fiterrors[p:]

    def mesh_flat(self, x, y, ctrx=512, ctry=512, img=0):
        """
        Generate flat field data efficiently for a given image using the existing model function.

        :param x: 2D array of x-coordinates
        :param y: 2D array of y-coordinates
        :param ctrx: x-coordinate of the image center
        :param ctry: y-coordinate of the image center
        :param img: image number
        :return: 2D array of flat field data
        """
        # Normalize coordinates
        coord_x = (x - ctrx) / 1024.0
        coord_y = (y - ctry) / 1024.0

        # Prepare the data array for the model function
        shape = x.shape
        data = (
            np.zeros(shape),  # mc (magnitude)
            np.ones(shape),   # airmass (set to 1 for flat field)
            coord_x,
            coord_y,
            np.zeros(shape),  # color1
            np.zeros(shape),  # color2
            np.zeros(shape),  # color3
            np.zeros(shape),  # color4
            np.full(shape, img),
            np.zeros(shape),  # y (not used in flat field calculation)
            np.ones(shape)    # err (set to 1, not used in model calculation)
        )

        # Use the existing model function to calculate the flat field
        return self.model(self.fitvalues, data)

    def old_flat(self, x, y, ctrx=512, ctry=512, img=0):
        return self.model(np.array(self.fitvalues), (0, 1, (x-ctrx)/1024., (y-ctry)/1024., 0, 0, 0, 0, img, 0, 0))

    def hyperbola(self, x, y):
        return (y*np.sqrt(1.0*x*x/y/y)+x)/2.0

    def model(self, values, data):
        """Optimized photometric response model"""
        mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err = data
        values = np.asarray(values)
        img = np.int64(img)
        
        model = np.zeros_like(mc)
        radius2 = coord_x**2 + coord_y**2

        val2 = np.concatenate((values[0:len(self.fitterms)], np.array(self.fixvalues)))

        for term, value in zip(self.fitterms + self.fixterms, val2):
            if term == 'N1': model += (1 + value) * mc
            elif term == 'N2': model += value * mc**2
            elif term == 'N3': model += value * mc**3
            elif term[0] == 'P':
                components = {'A': airmass, 'C': color1, 'D': color2, 'E': color3, 'F': color4, 
                              'R': radius2, 'X': coord_x, 'Y': coord_y}
                pterm = value
                n = 1
                for a in term[1:]:
                    if a.isdigit():
                        n = int(a)
                    elif a in components:
                        pterm *= components[a]**n
                        n = 1
                model += pterm
            elif term == 'GA': ga = value
            elif term == 'GW': gw = 10**value
            elif term == 'EA': ea = value
            elif term == 'EW': ew = value
            elif term == 'EQ': eq = value
            elif term == 'HA': ha = value
            elif term == 'HM': hm = value
            elif term == 'HN': hn = value
            elif term == 'XC':
                bval = np.where(value < 0, value * color1,
                                np.where(value <= 1, value * color2,
                                         (value - 1) * color3 + color2))
                model += bval
            elif term == 'SC':
                s = np.sin(value * np.pi)
                w1 = 0.5 * (-s + 1)
                w2 = 0.5 * (s + 1)
                bval = np.where(value < -0.5, value * color1,
                                np.where(value < 0.5, w1 * value * color1 + w2 * value * color2,
                                         np.where(value <= 1.5, color2 + w2 * (value - 1) * color2 + w1 * (value - 1) * color3,
                                                  (value - 1) * color3 + color2)))
                model += bval

        if 'ga' in locals() and 'gw' in locals():
            model += ga * np.exp(-radius2 * 25 / gw)

        if 'ea' in locals() and 'ew' in locals():
            foo = np.sqrt(radius2 / ew)
            model += ea * (np.exp(foo) / (foo + 1) - 1)


        if self.fit_xy:
            N = (len(values) - len(self.fitterms)) // 3
            zeros = values[len(self.fitterms):len(self.fitterms) + N]
            px = values[len(self.fitterms) + N:len(self.fitterms) + 2 * N]
            py = values[len(self.fitterms) + 2 * N:len(self.fitterms) + 3 * N]
            
            model += px[img] * coord_x + py[img] * coord_y + zeros[img]
        else:
            zeros = values[len(self.fitterms):]
            model += zeros[img]

        return model


    def old_model(self, values, data):
        """photometric response model"""

        mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err = data
        radius2 = None

        model = 0 # mc
        gw=0
        ga=0.05
        ew=1
        ea=0.05
        eq=1
        gauss=False
        expo=False
        hyp=False
        hm=6
        hn=0.5
        ha=1

        val2 = np.concatenate((values[0:len(self.fitterms)], np.array(self.fixvalues)))

        for term,value in zip(self.fitterms+self.fixterms, val2):

            # nonlinearity requires a special treatment
            if term == 'N1':
                #model += (value+1) * mc # value * mc
                model += (1+value) * mc # value * mc
            if term == 'N2':
                #model += (value+1) * mc # value * mc
                model += value * mc * mc
            if term == 'N3':
                #model += (value+1) * mc # value * mc
                model += value * mc * mc * mc # value * mc

            # Polynomial term in form PnXnY
            if term[0] == 'P':
                pterm = value; n=1;
                for a in term[1:]:
                    if self.isnumber(a): n = int(a)
                    if a == 'A': pterm *= np.power(airmass, n); n=1;
                    if a == 'C': pterm *= np.power(color1, n); n=1;
                    if a == 'D': pterm *= np.power(color2, n); n=1;
                    if a == 'E': pterm *= np.power(color3, n); n=1;
                    if a == 'F': pterm *= np.power(color4, n); n=1;
                    if a == 'R':
                        if radius2 is None: radius2=coord_x*coord_x+coord_y*coord_y
                        pterm *= np.power(radius2, n); n=1;
                    if a == 'X': pterm *= np.power(coord_x, n); n=1;
                    if a == 'Y': pterm *= np.power(coord_y, n); n=1;
                model += pterm

            if term == 'GA': ga = value; gauss=True;
            if term == 'GW': gw = np.power(10,value); gauss=True;

            if term == 'EA': ea = value; expo=True;
            if term == 'EW': ew = value; expo=True;
            if term == 'EQ': eq = value; expo=True;

            if term == 'HA': ha = value; hyp=True;
            if term == 'HM': hm = value; hyp=True;
            if term == 'HN': hn = value; hyp=True;

            if term == 'XC': 
                if value < 0: bval = value * color1; 
                if value > 0 and value <= 1: bval = value * color2; 
                if value > 1: bval = (value-1) * color3 + color2; 
                model += bval;

            if term == 'SC': 
                if value < -0.5: bval = value * color1;
                if value < 0.5 and value > -0.5: 
                    bval = \
                        (0.5 * np.sin(-value*np.pi) + 0.5) * value * color1 + \
                        (0.5 * np.sin( value*np.pi) + 0.5) * value * color2; 
                if value > 0.5 and value <= 1.5: 
                    bval = color2 + \
                        (0.5 * np.sin( value*np.pi) + 0.5) * (value-1) * color2 + \
                        (0.5 * np.sin(-value*np.pi) + 0.5) * (value-1) * color3; 
                if value > 1.5: 
                    bval = (value-1) * color3 + color2; 
                model += bval;

        if gauss:
            if radius2 is None: radius2=coord_x*coord_x+coord_y*coord_y
            model += ga*np.exp(-radius2*25/gw)
    #        print("Fu! "+str(ga*np.exp(-radius2/gw)))

#        if hyp:
#            model += ha * self.hyperbola(hm - x, hn)

        if expo:
            if radius2 is None: radius2=coord_x*coord_x+coord_y*coord_y
            # gw is a fwhm of the gaussian, ga is its amplitude
            foo = np.sqrt(radius2 / ew)
            model += ea * ( np.exp(foo)/(foo + 1) - 1 )
            #model += ea*np.power(np.exp(foo)/(foo+1),eq)

        if self.fit_xy: # xcoord/ycoord/zeropoint for each image
            N = np.int64(len(values[len(self.fitterms):]) / 3)
            zeros = np.array(values[len(self.fitterms):len(self.fitterms)+N])
            px = np.array(values[len(self.fitterms)+N:len(self.fitterms)+2*N])
            py = np.array(values[len(self.fitterms)+2*N:len(self.fitterms)+3*N])
            model += px[np.int64(img)] * coord_x
            model += py[np.int64(img)] * coord_y
            model += zeros[np.int64(img)]
        else:
            zeros = np.array(values[len(self.fitterms):])
            model += zeros[np.int64(img)]

        return model

    def isnumber(self, a):
        try:
            k=int(a)
            return True
        except:
            return False

    def residuals0(self, values, data):
        """pure residuals to compute sigma and similar things"""
        mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err = data
        return np.abs(y - self.model(values, data))

    def residuals(self, values, data):
        """residuals for fitting with error weighting and delinearization"""
        mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err = data
        dist = np.abs((y - self.model(values, data))/err)
        if self.delin:
            return self.cauchy_delin(dist)
        else:
            return dist
