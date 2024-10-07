import os
import astropy.table
import astropy.io.ascii

def get_atlas_dir(rasc, decl, width, height, directory, mlim):
    """get contents of one split of Atlas catalog (it is split into directories by magnitude)"""
    atlas_ecsv_tmp = f"atlas{os.getpid()}.ecsv"
    cmd = f"atlas {rasc} {decl} -rect {width},{height} -dir {directory} -mlim {mlim:.2f} -ecsv > {atlas_ecsv_tmp}"
    print(cmd)
    os.system(cmd)
    new = astropy.io.ascii.read(atlas_ecsv_tmp, format='ecsv')
    print(len(new))
    os.system("rm " + atlas_ecsv_tmp)
    return new

# filter transforms by Lupton (2005) (made for SDSS DR4)
# B: PC=-1.313 Bz=-0.2271 # With PanSTARRS has a serious trend and scatter
# V: PC=-0.4216 Vz=0.0038 # With PanSTARRS has a clear trend
# R: PD=0.2936 Rz=0.1439 # With PanSTARRS is ~ok (stars with r-i>2 tend off the line)
# I: PD=1.2444 Iz=0.3820 # With PanSTARRS is ok (more spread than PDPE, but no trends)

# filter transforms by mates (2024) PanSTARRS -> Stetson
# B: PC=-1.490989±0.008892 P2C=-0.125787±0.010429 P3C=0.022359±0.003590 Bz=-0.186304+/-0.003624
# V: PC=-0.510236±0.001648 Vz=0.0337082+/-0.004563
# R: PD=0.197420±0.005212 P2D=0.083113±0.004458 Rz=0.179943+/-0.002598
# I: PD=0.897087±0.012819 PE=0.575316±0.020487  Iz=0.423971+/-0.003196

def get_atlas(rasc, decl, width=0.25, height=0.25, mlim=17):
    """Load Atlas catalog from disk, (atlas command needs to be in path and working)"""
    cat = get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/00_m_16/', mlim)
    if mlim > 16: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/16_m_17/', mlim)])
    if mlim > 17: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/17_m_18/', mlim)])
    if mlim > 18: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/18_m_19/', mlim)])
    if mlim > 19: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/19_m_20/', mlim)])

    # Fill in the Stetson/Johnson filter transformations
    gr = cat['Sloan_g'] - cat['Sloan_r']
    ri = cat['Sloan_r'] - cat['Sloan_i']
    iz = cat['Sloan_i'] - cat['Sloan_z']
    cat['Johnson_B'] = cat['Sloan_r'] + 1.490989 * gr + 0.125787 * gr * gr - 0.022359 * gr*gr*gr + 0.186304
    cat['Johnson_V'] = cat['Sloan_r'] + 0.510236 * gr - 0.0337082
    cat['Johnson_R'] = cat['Sloan_r'] - 0.197420 * ri - 0.083113 * ri * ri - 0.179943
    cat['Johnson_I'] = cat['Sloan_r'] - 0.897087 * ri - 0.575316 * iz - 0.423971

    cat.meta['astepoch']=2015.5
    return cat

# get another catalog from a file
def get_catalog(filename, mlim=17):
    """Expected use: read output of another image as a calibration source for this frame"""
    cat = astropy.table.Table()
    try:
        catalog = astropy.io.ascii.read(filename, format='ecsv')
        cat.add_column(astropy.table.Column(name='radeg', dtype=np.float64, data=catalog['ALPHA_J2000']))
        cat.add_column(astropy.table.Column(name='decdeg', dtype=np.float64, data=catalog['DELTA_J2000']))
        cat.add_column(astropy.table.Column(name='pmra', dtype=np.float64, data=catalog['_RAJ2000']*0))
        cat.add_column(astropy.table.Column(name='pmdec', dtype=np.float64, data=catalog['_DEJ2000']*0))
        # obviously, a single image output is single filter only
        fltnames=['Sloan_r','Sloan_i','Sloan_g','Sloan_z', 'Johnson_B','Johnson_V','Johnson_R','Johnson_I','J','c','o']
        for fltname in fltnames:
            cat.add_column(astropy.table.Column(name=fltname, dtype=np.float64, data=catalog['MAG_CALIB']))
        print(f"Succesfully read the ecsv catalog {filename} (mlim={mlim})")
        return cat
    except astropy.io.ascii.core.InconsistentTableError:
        pass
#        print("Hm... the catalog is not an ecsv!")
    try:
        catalog = astropy.io.ascii.read(filename, format='tab')
#        cat.add_column(catalog['_RAJ2000'])
        cat.add_column(astropy.table.Column(name='radeg', dtype=np.float64, data=catalog['_RAJ2000']))
        cat.add_column(astropy.table.Column(name='decdeg', dtype=np.float64, data=catalog['_DEJ2000']))
        cat.add_column(astropy.table.MaskedColumn(name='pmra', dtype=np.float64, data=catalog['pmRA']/1e6, fill_value=0))
        cat.add_column(astropy.table.MaskedColumn(name='pmdec', dtype=np.float64, data=catalog['pmDE']/1e6, fill_value=0))

        # we may do different arrangements of the primary/secondary filters
        # for BV@Tautenburg the logical choice is BP primary and BP-RP to correct. 
        # PC = (R-B), PD = (B-G), PE = (G-R), with -B Sloan_i, Gmag as a bas may be chosen, default is BPmag
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_g', dtype=np.float64, data=catalog['RPmag'],fill_value=99.9))
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_r', dtype=np.float64, data=catalog['BPmag'],fill_value=99.9))
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_i', dtype=np.float64, data=catalog['Gmag'],fill_value=99.9))
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_z', dtype=np.float64, data=catalog['RPmag'],fill_value=99.9))
        # BPmag values into the other filter columns 
        fltnames=['Johnson_B','Johnson_V','Johnson_R','Johnson_I','J','c','o']
        for fltname in fltnames:
            cat.add_column(astropy.table.MaskedColumn(name=fltname, dtype=np.float64, data=catalog['BPmag'], fill_value=99.9))
        print(f"Succesfully read the tab-sep-list catalog {filename} (mlim={mlim})")
        cat.meta['astepoch']=2015.5
        return cat[np.logical_not(np.any([
            np.ma.getmaskarray(cat['Sloan_g']),
            np.ma.getmaskarray(cat['Sloan_r']),
            np.ma.getmaskarray(cat['Sloan_i']),
            np.ma.getmaskarray(cat['pmra']),
            np.ma.getmaskarray(cat['pmdec']),
            cat['Sloan_r']>mlim,
            cat['Sloan_i']>mlim,
            cat['Sloan_g']>mlim,
            cat['Sloan_g']-cat['Sloan_r']<-1.5
            ] ,axis=0))]
    except astropy.io.ascii.core.InconsistentTableError:
        print("Catalog cannot be read as an ECSV or TAB-SEP-LIST file!")
    return None


