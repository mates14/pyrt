/* Extract stars from refcat files */
/* Syntax: refcat ra[deg] dec[deg] [options] */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <fcntl.h>


#define ABS(a) (((a) > 0) ? (a) : -(a))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define NINT(a)  (((a) > 0) ? (int)((a)+0.5) : (int)((a)-0.5))

#define MAXSTAR 50000	/* number of stars to alloc each time */

/* A 3-vector */
typedef double REAL;	/* 8 byte floating point */
typedef struct {
   REAL x;
   REAL y;
   REAL z;
} VEC;

/* Dot product of two vectors: A.B */
#define DOT(A,B) ( (A).x*(B).x+(A).y*(B).y +(A).z*(B).z )

/* Test for lowendian byte order */
static int _ENDIAN_TEST = 1;
#define LOWENDIAN() (*(char*)&_ENDIAN_TEST)


/* STAR structure has units of [deg] [deg/yr] [mag] */
typedef struct {
   double ra;
   double dec;
   double plx;
   double dplx;
   double pmra;
   double dpmra;
   double pmdec;
   double dpmdec;
   double G;
   double dG;
   double B;
   double dB;
   double R;
   double dR;
   double Teff;
   double AG;
   int dupvar;
   double Ag;
   double rp1;
   double r1;
   double r10;
   double g;
   double dg;
   double gchi;
   int gcontrib;
   double r;
   double dr;
   double rchi;
   int rcontrib;
   double i;
   double di;
   double ichi;
   int icontrib;
   double z;
   double dz;
   double zchi;
   int zcontrib;
   int nstat;
   double J;
   double dJ;
   double H;
   double dH;
   double K;
   double dK;
} STAR;

/* STARDAT structure for binary storage */
typedef struct {
   unsigned int ra;		// 1e-7 deg
   unsigned int cdec;		// 1e-7 deg: south colatitude: 90+Dec
   int plx;			// 1e-5 arcsec
   int pmra;			// 1e-5 arcsec/yr
   int pmdec;			// 1e-5 arcsec/yr    5*4=20
   short int Teff;		// K
   unsigned short int AG;	// mmag
   unsigned short int Ag;	// mmag
   short int G;			// mmag
   short int B;			// mmag
   short int R;			// mmag
   short int g;			// mmag
   short int r;			// mmag
   short int i;			// mmag
   short int z;			// mmag
   short int J;			// mmag
   short int H;			// mmag
   short int K;			// mmag             13*2=26
   unsigned char rp1;		// 0.2 arcsec
   unsigned char r1;		// 0.2 arcsec
   unsigned char r10;		// 0.2 arcsec
   unsigned char dplx;		// 1e-5 arcsec
   unsigned char dpmra;		// 1e-5 arcsec/yr
   unsigned char dpmdec;	// 1e-5 arcsec/yr
   unsigned char dG;		// 2 mmag
   unsigned char dB;		// 2 mmag
   unsigned char dR;		// 2 mmag
   unsigned char dg;		// 2 mmag
   unsigned char dr;		// 2 mmag
   unsigned char di;		// 2 mmag
   unsigned char dz;		// 2 mmag
   unsigned char dJ;		// 2 mmag
   unsigned char dH;		// 2 mmag
   unsigned char dK;		// 2 mmag
   unsigned char nstat;
   unsigned char dupvar;
   unsigned char gchi;		// 0.1
   unsigned char gcontrib;
   unsigned char rchi;		// 0.1
   unsigned char rcontrib;
   unsigned char ichi;		// 0.1
   unsigned char icontrib;
   unsigned char zchi;		// 0.1
   unsigned char zcontrib;	//                26*1=26
} STARDAT;


/* Structure describing variable names and output format */
typedef struct {
   char *name;			// variable name
   size_t offset;		// offset from start of STAR structure
   char *fmt;			// output format
   double scale;		// rescale internal variable for output
} VARFMT;

VARFMT varname[44] = {
   { "RA",      offsetof(STAR, ra),       "%11.7f", 1.0}, 
   { "Dec",     offsetof(STAR, dec),      "%11.7f", 1.0}, 
   { "plx",     offsetof(STAR, plx),      "%6.2f",  3.6e6},   // mas
   { "dplx",    offsetof(STAR, dplx),     "%4.2f",  3.6e6},   // mas
   { "pmra",    offsetof(STAR, pmra),     "%8.2f",  3.6e6},   // mas
   { "dpmra",   offsetof(STAR, dpmra),    "%4.2f",  3.6e6},   // mas
   { "pmdec",   offsetof(STAR, pmdec),    "%8.2f",  3.6e6},   // mas
   { "dpmdec",  offsetof(STAR, dpmdec),   "%4.2f",  3.6e6},   // mas
   { "Gaia",    offsetof(STAR, G),        "%6.3f",  1.0},
   { "dGaia",   offsetof(STAR, dG),       "%5.3f",  1.0},
   { "BP",      offsetof(STAR, B),        "%6.3f",  1.0},
   { "dBP",     offsetof(STAR, dB),       "%5.3f",  1.0},
   { "RP",      offsetof(STAR, R),        "%6.3f",  1.0},
   { "dRP",     offsetof(STAR, dR),       "%5.3f",  1.0},
   { "Teff",    offsetof(STAR, Teff),     "%5.0f",  1.0},
   { "AGaia",   offsetof(STAR, AG),       "%5.3f",  1.0},
   { "dupvar",  offsetof(STAR, dupvar),   "%1d",    1.0},
   { "Ag",      offsetof(STAR, Ag),       "%5.3f",  1.0},
   { "rp1",     offsetof(STAR, rp1),      "%4.1f",  3.6e3},  // arcsec
   { "r1",      offsetof(STAR, r1),       "%4.1f",  3.6e3},  // arcsec
   { "r10",     offsetof(STAR, r10),      "%4.1f",  3.6e3},  // arcsec
   { "g",       offsetof(STAR, g),        "%6.3f",  1.0},
   { "dg",      offsetof(STAR, dg),       "%5.3f",  1.0},
   { "gchi",    offsetof(STAR, gchi),     "%5.2f",  1.0},
   { "gcontrib",offsetof(STAR, gcontrib), "%02x",   1.0},
   { "r",       offsetof(STAR, r),        "%6.3f",  1.0},
   { "dr",      offsetof(STAR, dr),       "%5.3f",  1.0},
   { "rchi",    offsetof(STAR, rchi),     "%5.2f",  1.0},
   { "rcontrib",offsetof(STAR, rcontrib), "%02x",   1.0},
   { "i",       offsetof(STAR, i),        "%6.3f",  1.0},
   { "di",      offsetof(STAR, di),       "%5.3f",  1.0},
   { "ichi",    offsetof(STAR, ichi),     "%5.2f",  1.0},
   { "icontrib",offsetof(STAR, icontrib), "%02x",   1.0},
   { "z",       offsetof(STAR, z),        "%6.3f",  1.0},
   { "dz",      offsetof(STAR, dz),       "%5.3f",  1.0},
   { "zchi",    offsetof(STAR, zchi),     "%5.2f",  1.0},
   { "zcontrib",offsetof(STAR, zcontrib), "%02x",   1.0},
   { "nstat",   offsetof(STAR, nstat),    "%3d",    1.0},
   { "J",       offsetof(STAR, J),        "%6.3f",  1.0},
   { "dJ",      offsetof(STAR, dJ),       "%5.3f",  1.0},
   { "H",       offsetof(STAR, H),        "%6.3f",  1.0},
   { "dH",      offsetof(STAR, dH),       "%5.3f",  1.0},
   { "K",       offsetof(STAR, K),        "%6.3f",  1.0},
   { "dK",      offsetof(STAR, dK),       "%5.3f",  1.0}
};



/* Input formats */
#define IN_NONE   0	/* test for input format */
#define IN_CSV    1	/* all 44 fields from CSV refcat2 */
#define IN_BIN    2	/* 44 field binary format created by refcat.c */
#define IN_GRI    3	/* just ra,dec,pmra,pmdec,rp1,r1,g,r,i,z,J */

/* Output formats */
#define OUT_ALL   1	/* all 44 fields from refcat2 */
#define OUT_ATLAS 2	/* just ra,dec,g,r,i,z,J,c,o */
#define OUT_VAR   3	/* custom list of variables */
#define OUT_ECSV  4	/* astropy ecsv with grizJHK */

/* Prototypes */
void syntax(char *prog);

/* Sky coord defined by offset of da,dd[rad] from the circles through a,d */
void adoffset(VEC a, VEC d, double da, double dd, VEC *p);

/* Read all 44 columns from CSV file */
int read_csv(char *fname, double mlim, double rlim,
	     int rect, VEC p1, double t1, VEC p2, double t2,
	     int *nalloc, int *nstar, STAR **star);

/* Read binary data file */
int read_bin(char *fname, double mlim, double rlim,
	     int rect, VEC p1, double t1, VEC p2, double t2,
	     int *nalloc, int *nstar, STAR **star);

/* Test whether a file is refcat binary CSV, return number of stars */
int test_bin(char *fname);

/* Read CSV and re-write as a binary data file */
int write_bin(int ra, int dec, int ndir, char **root, char *bin, char *exten);

/* Write a single variable from a STAR structure */
int write_var(int var, STAR *star);


/* Global variables */
int VERBOSE=0;

char *rootspec = "/home/mates/atlas/00_m_16";	// Root directory

int main(int argc, char **argv)
{
   int i, j, k, m, nstar, nalloc, rect, infmt, outfmt, hdr;
   char *exten, *bindir, *degin, fname[1024];
   // char *rootspec, 
   char *rootdir[44], *varlist, *vptr, var[1024];
   int ndir, nvar, varidx[44];
   double mlim, rlim, dr=atan(1.0)/45, pi=4*atan(1.0);
   double ra0, dec0, dra, ddec, decmin, decmax, ra, dec, sina, sind, cosa;
   double cyan, orange, clr;
   VEC rapole, decpole, pointing, corner[4], P;
   struct stat statbuf;
   STAR *star;

/* Mandatory arguments */
   if(argc < 3) {
      syntax(argv[0]);
      exit(1);
   }

   i  = sscanf(argv[1], "%lf", &ra0);
   i += sscanf(argv[2], "%lf", &dec0);
   if(i != 2) {
      fprintf(stderr, "Error parsing ra dec from %s %s\n",
	      argv[2], argv[3]);
      exit(1);
   }

/* Defaults */
   //rootspec = "/atlas/cal/RC2/m17";	// Root directory
   mlim = 18.0;		// Limiting magnitude for m<mlim
   rlim = 0.0;		// Limiting radius for rp1>rlim
   VERBOSE = 0;		// Verbosity level
   infmt = IN_NONE;	// Test for format
   outfmt = OUT_ATLAS;	// Output format
   exten = "rc2";	// Input file extension
   rect = 1;		// 0/1 for rectangle or radius around pointing
   dra = 0;		// RA size of rectangle [angular deg] or radius
   ddec = 0;		// Dec size of rectangle [angular deg]
   bindir = NULL;	// Hijack: read all CSV, write all binary
   hdr = 0;		// header line?
   varlist = NULL;	// list of variables to print

/* Parse options */
   for(j=3; j<argc; j++) {
      if(strcmp(argv[j], "-dir") == 0) {
	 rootspec = argv[++j];

      } else if(strcmp(argv[j], "-exten") == 0) {
	 exten = argv[++j];

      } else if(strcmp(argv[j], "-gri") == 0) {
	 infmt = IN_GRI;

      } else if(strcmp(argv[j], "-csv") == 0) {
	 infmt = IN_CSV;

      } else if(strcmp(argv[j], "-bin") == 0) {
	 infmt = IN_BIN;

      } else if(strcmp(argv[j], "-hdr") == 0) {
	 hdr = 1;

      } else if(strcmp(argv[j], "-nohdr") == 0) {
	 hdr = 0;

      } else if(strcmp(argv[j], "-all") == 0) {
	 outfmt = OUT_ALL;
	 hdr = 1;

      } else if(strcmp(argv[j], "-ecsv") == 0) {
	 outfmt = OUT_ECSV;

      } else if(strcmp(argv[j], "-var") == 0) {
	 varlist = argv[++j];
	 outfmt = OUT_VAR;
	 hdr = 1;

      } else if(strcmp(argv[j], "-rect") == 0) {
	 if(sscanf(argv[++j], "%lf,%lf", &dra, &ddec) != 2) {
	    fprintf(stderr, "Cannot parse dra,ddec from %s\n", argv[j]);
	    exit(1);
	 }
	 rect = 1;

      } else if(strcmp(argv[j], "-rad") == 0) {
	 if(sscanf(argv[++j], "%lf", &dra) != 1) {
	    fprintf(stderr, "Cannot parse dra from %s\n", argv[j]);
	    exit(1);
	 }
	 rect = 0;

      } else if(strcmp(argv[j], "-mlim") == 0) {
	 if(sscanf(argv[++j], "%lf", &mlim) != 1) {
	    fprintf(stderr, "Cannot parse mlim from %s\n", argv[j]);
	    exit(1);
	 }

      } else if(strcmp(argv[j], "-rlim") == 0) {
	 if(sscanf(argv[++j], "%lf", &rlim) != 1) {
	    fprintf(stderr, "Cannot parse rlim from %s\n", argv[j]);
	    exit(1);
	 }

      } else if(strcmp(argv[j], "-CSV_to_binary") == 0) {
	 bindir = argv[++j];
	 VERBOSE = 1;		// Default is verbose, quiet with -silent

      } else if(strcmp(argv[j], "-silent") == 0) {
	 VERBOSE = 0;

      } else if(strcmp(argv[j], "-verb") == 0) {
	 VERBOSE = 1;

      } else if(strcmp(argv[j], "-VERB") == 0) {
	 VERBOSE = 2;

      } else {
	 syntax(argv[0]);
	 exit(0);
      }
   }

/* Pick apart the rootspec into directories */
   vptr = rootspec;
   for(j=ndir=0; j<44; j++) {
      if(sscanf(vptr, "%[^,]", var) != 1) break;
/* Sanity check that input directory exists */
      if( (i = stat(var, &statbuf)) ) {
	 perror("stat");
	 fprintf(stderr,"Cannot access root directory %s\n", var);
	 exit(1);
      }
      rootdir[ndir] = malloc(strlen(var)+1);
      strcpy(rootdir[ndir++], var);
      vptr = index(vptr, ',');
      if(vptr++ == NULL) break;
   }

   if(VERBOSE > 0) {
      printf("Searching directories:\n");
      for(m=0; m<ndir; m++) {
	 printf("%d %s\n", m, rootdir[m]);
      }
   }

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
/* Hijack everything and convert all CSV files to binary? */
   if(bindir != NULL) {

/* Sanity check that output directory exists */
      if( (i = stat(bindir, &statbuf)) ) {
	 perror("stat");
	 fprintf(stderr,"Cannot access binary output directory %s\n", bindir);
	 exit(1);
      }
/* Make a binary file for every CSV file */
      for(j=-90; j<90; j++) {
	 for(i=0; i<360; i++) {
	    k = write_bin(i, j, ndir, rootdir, bindir, exten);
	 }
      }
      exit(0);
   }
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


/* More sanity checks */
   if(dra == 0 || (rect && ddec == 0) ) {
      fprintf(stderr, "Require a radius or rectangle dimension\n");
      syntax(argv[0]);
      exit(1);
   }

/* Convert coordinate degrees to radians */
   ra0 *= dr;
   dec0 *= dr;
   dra *= dr;
   ddec *= dr;

/* Identify all sqdeg overlapped by this rectangle or circle */
/* "Within rectangle" means closer in angle to the N-S, E-W great
   circles defined by RA,Dec than the specified offsets. */
   
/* The pointing */
   pointing.x = cos(dec0) * cos(ra0);
   pointing.y = cos(dec0) * sin(ra0);
   pointing.z = sin(dec0);


/* Poles of the central great circles defining the rectangle */
   rapole.x = cos(ra0+pi/2);
   rapole.y = sin(ra0+pi/2);
   rapole.z = 0;

   decpole.x = -sin(dec0) * cos(ra0);
   decpole.y = -sin(dec0) * sin(ra0);
   decpole.z =  cos(dec0);

   if(VERBOSE > 1) {
      fprintf(stderr, "Poles for %.1f,%.1f  %6.3f,%6.3f,%6.3f are %6.3f,%6.3f,%6.3f   %6.3f,%6.3f,%6.3f\n",
	     ra0/dr, dec0/dr, pointing.x, pointing.y, pointing.z,
	     rapole.x, rapole.y, rapole.z, decpole.x, decpole.y, decpole.z);
   }

   if(rect) {
/* Get the corners of the rectangle on the sky */
      adoffset(rapole, decpole, +dra, +ddec, &corner[0]);
      adoffset(rapole, decpole, -dra, +ddec, &corner[1]);
      adoffset(rapole, decpole, -dra, -ddec, &corner[2]);
      adoffset(rapole, decpole, +dra, -ddec, &corner[3]);

      if(VERBOSE > 1) {
	 for(i=0; i<4; i++) {
	    fprintf(stderr, "Corner %d  %6.3f,%6.3f,%6.3f  %7.1f %7.1f\n",
		   i, corner[i].x, corner[i].y, corner[i].z,
		   atan2(corner[i].y, corner[i].x)/dr, asin(corner[i].z)/dr);
	 }
      }

/* Dec range to consider */
      decmin = MIN(corner[0].z, corner[1].z);
      decmin = MIN(decmin, corner[2].z);
      decmin = MIN(decmin, corner[3].z);
      decmax = MAX(corner[0].z, corner[1].z);
      decmax = MAX(decmax, corner[2].z);
      decmax = MAX(decmax, corner[3].z);
      decmin = asin(decmin);
      decmax = asin(decmax);
   } else {
      ddec = dra;
      decmin = dec0 - ddec;
      decmax = dec0 + ddec;
   }
   if(dec0+ddec >= pi/2) decmax = pi/2;
   if(dec0-ddec <= -pi/2) decmin = -pi/2;

   if(VERBOSE > 1) {
      fprintf(stderr, "Dec range %.1f %.1f\n", decmin/dr, decmax/dr);
   }

/* degin[] = 0/1 if it is inside the rectangle */
   degin = (char *)calloc(360*180, sizeof(char));
   
/* Each corner of the rectangle lies in a sqdeg */
   if(rect) {
      for(k=0; k<4; k++) {
	 ra = atan2(corner[k].y, corner[k].x)/dr;
	 dec = asin(corner[k].z)/dr;
	 i = (int)floor(fmod(ra+360,360)+1e-8);
	 j = (int)floor(dec+1e-8) + 90;
	 degin[i+j*360] = 1;
      }

      if(VERBOSE > 1) {
	 fprintf(stderr, "rectangle corners lie in:  ");
	 for(k=0; k<360*180; k++) {
	    if(degin[k]) {
	       i = k % 360;
	       j = k / 360 - 90;
	       fprintf(stderr, " %03d%+03d", i, j);
	    }
	 }
	 fprintf(stderr, "\n");
      }

/* Center the circle lies in a sqdeg */
   } else {
      i = (int)floor(fmod(ra0/dr+360,360)+1e-8);
      j = (int)floor(dec0/dr+1e-8) + 90;
      degin[i+j*360] = 1;
/* (Ignore the possibility of a chord getting into a sqdeg
   without touching a corner) */
   }

/* dot product with rapole and decpole should be 0+/-sin{da,dd} */
   sina = sin(dra);
   sind = sin(ddec);
/* For circle dot product with pointing should be >cosda */
   cosa = cos(dra);

/* Mark each sqdeg that has a corner inside the rectangle or circle */
   for(j=(int)floor(decmin/dr+1e-8)+90; j<=(int)floor(decmax/dr-1e-8)+90; j++) {
      dec = (j-90) * dr;
      for(i=0; i<360; i++) {
	 ra = i * dr;
	 for(k=0; k<4; k++) {
	    P.x = cos(dec+(k/2)*dr) * cos(ra+(k%2)*dr);
	    P.y = cos(dec+(k/2)*dr) * sin(ra+(k%2)*dr);
	    P.z = sin(dec+(k/2)*dr);
	    if(rect) {
	       if(DOT(P,pointing) < 0 ||
		  DOT(P,rapole) > sina  || DOT(P,rapole) < -sina ||
		  DOT(P,decpole) > sind || DOT(P,decpole) < -sind) {
		  continue;
	       }
	    } else {
	       if(DOT(P,pointing) < cosa) continue;
	    }
//	    printf("%5d %5d %8.3f %8.3f %8.3f %8.3f\n", i, j,
//		   DOT(P,rapole), sina, DOT(P,decpole), sind);
	    degin[i+j*360] = 1;
	    break;
	 }
      }
   }
   
   if(VERBOSE > 1) {
      fprintf(stderr, "Sqdeg with corners inside area:  ");
      for(k=0; k<360*180; k++) {
	 if(degin[k]) {
	    i = k % 360;
	    j = k / 360 - 90;
	    fprintf(stderr, " %03d%+03d", i, j);
	 }
      }
      fprintf(stderr, "\n");
   }

/* Read each refcat file, keep desired stars */
   star = NULL;
   nalloc = 0;
   nstar = 0;
   for(m=0; m<ndir; m++) {
      for(k=0; k<360*180; k++) {
	 if(!degin[k]) continue;	// Skip unused degree files
	 i = k % 360;
	 j = k / 360 - 90;
	 snprintf(fname, 1024, "%s/%03d%+03d.%s", rootdir[m], i, j, exten);

/* Unspecified input format?  Test the first file. */
	 if(infmt == IN_NONE) {
	    if(test_bin(fname) > 0) {
	       infmt = IN_BIN;
	    } else {
	       infmt = IN_CSV;
	    }
	 }

/* Read the data file */
	 if(infmt == IN_CSV) {
	    read_csv(fname, mlim, rlim, rect,
		     rect?rapole:pointing, rect?sina:cosa, decpole, sind,
		     &nalloc, &nstar, &star);

	 } else if(infmt == IN_BIN) {
	    read_bin(fname, mlim, rlim, rect,
		     rect?rapole:pointing, rect?sina:cosa, decpole, sind,
		     &nalloc, &nstar, &star);

	 }
	 if(VERBOSE > 0) {
	    fprintf(stderr, "Read file %s with format %d total number of stars %d\n",
		    fname, infmt, nstar);
	 }
      }
   }

   if(outfmt == OUT_ATLAS) {

/* Synthesize (181023) ATLAS cyan and orange from a couple of observations
   02a58400o0400c and 02a58406o0400o.  There's a bit of curvature:
     (g - c_inst) = 25.77 + 0.467 (g-r) + 0.048 (g-r)^2^
     (r - o_inst) = 25.59 + 0.443 (r-i) + 0.090 (r-i)^2^
 */

/* Write ATLAS-specific results to stdout */
      if(hdr) {
	 printf("#  RA         Dec       g      r      i      z      J      c      o\n");
      }
      
      for(i=0; i<nstar; i++) {
	 clr = star[i].g - star[i].r;
	 cyan = star[i].g - 0.467*clr - 0.048*clr*clr;
	 clr = star[i].r - star[i].i;
	 orange = star[i].r - 0.443*clr - 0.090*clr*clr;
	 printf("%10.6f %10.6f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n",
		star[i].ra, star[i].dec, 
		star[i].g, star[i].r, star[i].i, star[i].z, star[i].J,
		cyan, orange);
      }

/* Dump the entire star record */
   } else if(outfmt == OUT_ALL) {
      if(hdr) {
	 printf("#   RA             Dec      plx  dplx    pmra dpmra   pmdec dpmdec Gaia  dGaia   BP   dBP     RP    dRP   Teff AGaia dupvar Ag rp1     r1    r10    g   dg  gchi gcontrib r   dr  rchi rcontrib i   di  ichi icontrib z   dz  zchi zcontrib nstat J   dJ     H     dH     K     dK\n");
      }
      for(i=0; i<nstar; i++) {
	 printf("%11.7f %11.7f %6.2f %4.2f %8.2f %4.2f %8.2f %4.2f %6.3f %5.3f %6.3f %5.3f %6.3f %5.3f %5.0f %5.3f %1d %5.3f %6.1f %6.1f %6.1f %6.3f %5.3f %5.2f %02x %6.3f %5.3f %5.2f %02x %6.3f %5.3f %5.2f %02x %6.3f %5.3f %5.2f %02x %3d %6.3f %5.3f %6.3f %5.3f %6.3f %5.3f\n",
		star[i].ra, star[i].dec, 3.6e6*star[i].plx, 3.6e6*star[i].dplx,
		3.6e6*star[i].pmra, 3.6e6*star[i].dpmra,
		3.6e6*star[i].pmdec, 3.6e6*star[i].dpmdec,
		star[i].G, star[i].dG, star[i].B, star[i].dB, star[i].R, star[i].dR,
		star[i].Teff, star[i].AG, star[i].dupvar, star[i].Ag,
		3.6e3*star[i].rp1, 3.6e3*star[i].r1, 3.6e3*star[i].r10,
		star[i].g, star[i].dg, star[i].gchi, star[i].gcontrib,
		star[i].r, star[i].dr, star[i].rchi, star[i].rcontrib,
		star[i].i, star[i].di, star[i].ichi, star[i].icontrib,
		star[i].z, star[i].dz, star[i].zchi, star[i].zcontrib, star[i].nstat,
		star[i].J, star[i].dJ, star[i].H, star[i].dH, star[i].K, star[i].dK);
      }

/* Dump a custom list of variables */
   } else if(outfmt == OUT_VAR) {

/* Decipher the indices for each named variable */
      vptr = varlist;
      for(j=nvar=0; j<44; j++) {
	 if(sscanf(vptr, "%[^,]", var) != 1) break;

	 for(i=0; strcasecmp(varname[i].name, var) && i<44; i++);
	 if(i == 44) {
	    fprintf(stderr, "Variable name %s not known\n", var);
	    return(-1);
	 }
	 varidx[nvar++] = i;

	 vptr = index(vptr, ',');
	 if(vptr++ == NULL) break;
      }

/* Write a header? */
      if(hdr) {
	 printf("#");
	 for(j=0; j<nvar ; j++) printf(" %s", varname[varidx[j]].name);
	 printf("\n");
      }

/* Write each star */
      for(i=0; i<nstar; i++) {
	 for(j=0; j<nvar ; j++) write_var(varidx[j], star+i);
	 printf("\n");
      }

   } else if(outfmt == OUT_ECSV) {
       printf("# %%ECSV 1.0\n");
       printf("# ---\n");
       printf("# datatype:\n");
       printf("# - {name: radeg, unit: deg, datatype: float64, description: Right ascension (J2000)}\n");
       printf("# - {name: decdeg, unit: deg, datatype: float64, description: Declination (J2000)}\n");
       printf("# - {name: Sloan_g, unit: mag, datatype: float64, description: SDSS g-band magnitude}\n");
       printf("# - {name: Sloan_r, unit: mag, datatype: float64, description: SDSS r-band magnitude}\n");
       printf("# - {name: Sloan_i, unit: mag, datatype: float64, description: SDSS i-band magnitude}\n");
       printf("# - {name: Sloan_z, unit: mag, datatype: float64, description: SDSS z-band magnitude}\n");
       printf("# - {name: J, unit: mag, datatype: float64, description: J magnitude}\n");
       printf("# - {name: H, unit: mag, datatype: float64, description: H magnitude}\n");
       printf("# - {name: K, unit: mag, datatype: float64, description: K magnitude}\n");
       printf("# - {name: pmra, unit: deg/yr, datatype: float64, description: proper motion in RA}\n");
       printf("# - {name: pmdec, unit: deg/yr, datatype: float64, description: proper motion in DEC}\n");
       printf("# schema: astropy-2.0\n");
       printf("radeg decdeg Sloan_g Sloan_r Sloan_i Sloan_z J H K pmra pmdec\n");

       for(i=0; i<nstar; i++) {
           printf("%0.6f %0.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.9f %.9f\n",
                   star[i].ra, star[i].dec, 
                   star[i].g, star[i].r, star[i].i, star[i].z, 
                   star[i].J, star[i].H, star[i].K,
                   star[i].pmra, star[i].pmdec
                    );
       }

   }
   
   exit(0);
}


/* Sky coord defined by offset of da,dd[rad] from the circles through a,d */
/* p.a=sin(da) and a.z=0   p.d=sin(dd)   p.p=1 */
void adoffset(VEC a, VEC d, double da, double dd, VEC *p)
{
   double tmp, ca, sa, cd, sd, pi=4*atan(1.0);

/* Impossible to find this offset? */
   if(ABS(da) + ABS(dd) >= pi/2) {
      p->x = p->y = p->z = 0;
      return;
   }

/* Solve for point when RA=Dec=0, i.e. a=y and d=z */
   p->y = sin(da);
   p->z = sin(dd);
   p->x = sqrt(1 - p->y*p->y - p->z*p->z);
   
/* Rotate by -Dec around y and by RA around z */
/* Note sin(RA)=-a.x, cos(RA)=a.y; sin(Dec)=-d.x/a.y, cos(Dec)=d.z */
   sa = -a.x;
   ca =  a.y;
   sd = ABS(ca)>0.7 ? -d.x/ca : -d.y/sa;
   cd = d.z;

   tmp = p->x;
   p->x = p->x * cd - p->z * sd;
   p->z =  tmp * sd + p->z * cd;

   tmp = p->x;
   p->x = p->x * ca - p->y * sa;
   p->y =  tmp * sa + p->y * ca;

//   printf("%8.3f %8.3f %8.3f  %8.3f %8.3f\n",
//	  p->x, p->y, p->z, atan2(p->y,p->x)*180/pi, asin(p->z)*180/pi);

   return;
}


/* Read all 44 columns from CSV file */
int read_csv(char *fname, double mlim, double rlim, int rect,
	     VEC p1, double t1, VEC p2, double t2,
	     int *nalloc, int *nstar, STAR **star)
{
   int i, n;
   char line[1024];
   FILE *fp;
   double m, dr=atan(1.0)/45;
   VEC P;

   long int inra, indec;
   int inplx, indplx, inpmra, indpmra, inpmdec, indpmdec;
   int inG, indG, inB, indB, inR, indR;
   int inTeff, inAG, indupvar, inAg, inrp1, inr1, inr10;
   int ing, indg, ingchi, ingcontrib, inr, indr, inrchi, inrcontrib;
   int ini, indi, inichi, inicontrib, inz, indz, inzchi, inzcontrib;
   int innstat, inJ, indJ, inH, indH, inK, indK;

   if( (fp=fopen(fname, "r")) == NULL) {
      fprintf(stderr, "Cannot open %s\n", fname);
      return(-1);
   }
   for(i=0;  ; i++) {
      if(fgets(line, 1024, fp) == NULL) break;

      n = sscanf(line, "%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%x,%d,%d,%d,%x,%d,%d,%d,%x,%d,%d,%d,%x,%d,%d,%d,%d,%d,%d,%d",
		 &inra, &indec,
		 &inplx, &indplx, &inpmra, &indpmra, &inpmdec, &indpmdec,
		 &inG, &indG, &inB, &indB, &inR, &indR,
		 &inTeff, &inAG, &indupvar, &inAg, &inrp1, &inr1, &inr10,
		 &ing, &indg, &ingchi, &ingcontrib,
		 &inr, &indr, &inrchi, &inrcontrib,
		 &ini, &indi, &inichi, &inicontrib,
		 &inz, &indz, &inzchi, &inzcontrib,
		 &innstat, &inJ, &indJ, &inH, &indH, &inK, &indK);
      if(n != 44) {
	 if(VERBOSE > 0) {
	    fprintf(stderr, "Error: only read %d items at line %d from file %s\n", n, i+1, fname);
	 }
	 fclose(fp);
	 return(-2);
      }
/* Verify there is enough space */
      if(*nstar > *nalloc-1) {
	 (*star) = (STAR *)realloc((*star), sizeof(STAR)*(size_t)(*nalloc+MAXSTAR));
	 *nalloc += MAXSTAR;
      }

/* Rescale integers to units of [deg] [deg/yr] and [mag] */
      (*star)[*nstar].ra = 1e-8 * inra;
      (*star)[*nstar].dec = 1e-8 * indec;
      (*star)[*nstar].plx = 1e-5 * inplx / 3600.0;
      (*star)[*nstar].dplx = 1e-5 * indplx / 3600.0;
      (*star)[*nstar].pmra = 1e-5 * inpmra / 3600.0;
      (*star)[*nstar].dpmra = 1e-5 * indpmra / 3600.0;
      (*star)[*nstar].pmdec = 1e-5 * inpmdec / 3600.0;
      (*star)[*nstar].dpmdec = 1e-5 * indpmdec / 3600.0;
      (*star)[*nstar].G = 1e-3 * inG;
      (*star)[*nstar].dG = 1e-3 * indG;
      (*star)[*nstar].B = 1e-3 * inB;
      (*star)[*nstar].dB = 1e-3 * indB;
      (*star)[*nstar].R = 1e-3 * inR;
      (*star)[*nstar].dR = 1e-3 * indR;
      (*star)[*nstar].Teff = inTeff;
      (*star)[*nstar].AG = 1e-3 * inAG;
      (*star)[*nstar].dupvar = indupvar;
      (*star)[*nstar].Ag = 1e-3 * inAg;
      (*star)[*nstar].rp1 = 1e-1 * inrp1 / 3600.0;
      (*star)[*nstar].r1 = 1e-1 * inr1 / 3600.0;
      (*star)[*nstar].r10 = 1e-1 * inr10 / 3600.0;
      (*star)[*nstar].g = 1e-3 * ing;
      (*star)[*nstar].dg = 1e-3 * indg;
      (*star)[*nstar].gchi = 1e-2 * ingchi;
      (*star)[*nstar].gcontrib = ingcontrib;
      (*star)[*nstar].r = 1e-3 * inr;
      (*star)[*nstar].dr = 1e-3 * indr;
      (*star)[*nstar].rchi = 1e-2 * inrchi;
      (*star)[*nstar].rcontrib = inrcontrib;
      (*star)[*nstar].i = 1e-3 * ini;
      (*star)[*nstar].di = 1e-3 * indi;
      (*star)[*nstar].ichi = 1e-2 * inichi;
      (*star)[*nstar].icontrib = inicontrib;
      (*star)[*nstar].z = 1e-3 * inz;
      (*star)[*nstar].dz = 1e-3 * indz;
      (*star)[*nstar].zchi = 1e-2 * inzchi;
      (*star)[*nstar].zcontrib = inzcontrib;
      (*star)[*nstar].nstat = innstat;
      (*star)[*nstar].J = 1e-3 * inJ;
      (*star)[*nstar].dJ = 1e-3 * indJ;
      (*star)[*nstar].H = 1e-3 * inH;
      (*star)[*nstar].dH = 1e-3 * indH;
      (*star)[*nstar].K = 1e-3 * inK;
      (*star)[*nstar].dK = 1e-3 * indK;

/* Is it bright enough? */
      m = MIN((*star)[*nstar].g, (*star)[*nstar].r);
      m = MIN((*star)[*nstar].i, m);
      if(m > mlim) continue;
/* Is it isolated enough? */
      if(1e-1*inrp1 < rlim) continue;
	    
/* Is it inside the rectangle or radius?  Skip if not. */
      P.x = cos((*star)[*nstar].dec*dr) * cos((*star)[*nstar].ra*dr);
      P.y = cos((*star)[*nstar].dec*dr) * sin((*star)[*nstar].ra*dr);
      P.z = sin((*star)[*nstar].dec*dr);
      if(rect) {
	 if(DOT(P,p1) > t1  || DOT(P,p1) < -t1 ||
	    DOT(P,p2) > t2 || DOT(P,p2) < -t2) {
	    continue;
	 }
      } else {
	 if(DOT(P,p1) < t1) continue;
      }

/* Keep the star */
      *nstar += 1;
   }
   fclose(fp);
   return(0);
}


#define BOMB(n) {fprintf(stderr, "failed to read/write %d bytes", n); return(-1);}


void byterev(int n, char *data)
{
   char tmp;
   if(n == 2) {
      tmp = *data;
      *data = *(data+1);
      *(data+1) = tmp;
      return;
   }
/* Swap words to complete the 32 bit byte swap */
   if(n == 4) {
      tmp = *data;
      *data = *(data+3);
      *(data+3) = tmp;
      tmp = *(data+1);
      *(data+1) = *(data+2);
      *(data+2) = tmp;
      return;
   }
   fprintf(stderr, "byterev: illegal n %d\n", n);
   exit(1);
}


/* Test whether a file is refcat binary CSV, return number of stars */
int test_bin(char *fname)
{
   int fd, n, lowendian;
   unsigned int ipi=3141592653, magic;

   if( (fd = open(fname, O_RDONLY)) < 0) {
      fprintf(stderr, "Cannot open %s for reading\n", fname);
      return(-1);
   }

   if( (read(fd, &n, 4) != 4) || (read(fd, &magic, 4) != 4) ||
       (read(fd, &lowendian, 4) != 4) ) {
      close(fd);
      return(-2);
   }
   close(fd);

   if( lowendian ^ LOWENDIAN() ) byterev(4, (char *)&n);
   if( lowendian ^ LOWENDIAN() ) byterev(4, (char *)&magic);

   if(magic != ipi) return(-3);
   return(n);
}

/* Read a binary CSV file */
int read_bin(char *fname, double mlim, double rlim, int rect,
	     VEC p1, double t1, VEC p2, double t2,
	     int *nalloc, int *nstar, STAR **star)
{
   int i, fd, nrc, lowendian;
   unsigned int ipi=3141592653, magic;
   double m, dr=atan(1.0)/45;
   VEC P;
   STARDAT dat;

   if( (fd = open(fname, O_RDONLY)) < 0) {
      fprintf(stderr, "Cannot open %s for reading\n", fname);
      return(-1);
   }

   if( (read(fd, &nrc, 4) != 4) ||
       (read(fd, &magic, 4) != 4) ||
       (read(fd, &lowendian, 4) != 4) ) {
      fprintf(stderr, "Cannot read from file %s\n", fname);
      close(fd);
      return(-1);
   }

   if( lowendian ^ LOWENDIAN() ) byterev(4, (char *)&nrc);
   if( lowendian ^ LOWENDIAN() ) byterev(4, (char *)&magic);

   if(magic != ipi) {
      fprintf(stderr, "Binary file %s has ID %d instead of %d\n", fname, magic, ipi);
      return(-1);
   }

   for(i=0; i<nrc ; i++) {
      if(read(fd, &dat, sizeof(STARDAT)) != sizeof(STARDAT))
	 BOMB((int)sizeof(STARDAT));

/* Verify there is enough space */
      if(*nstar > *nalloc-1) {
	 (*star) = (STAR *)realloc((*star), sizeof(STAR)*(size_t)(*nalloc+MAXSTAR));
	 *nalloc += MAXSTAR;
      }

/* Flip bytes around if required */
      if( lowendian ^ LOWENDIAN() ) {
	 byterev(4, (char *)&dat.ra);
	 byterev(4, (char *)&dat.cdec);
	 byterev(4, (char *)&dat.plx);
	 byterev(4, (char *)&dat.pmra);
	 byterev(4, (char *)&dat.pmdec);
	 byterev(2, (char *)&dat.Teff);
	 byterev(2, (char *)&dat.AG);
	 byterev(2, (char *)&dat.Ag);
	 byterev(2, (char *)&dat.G);
	 byterev(2, (char *)&dat.B);
	 byterev(2, (char *)&dat.R);
	 byterev(2, (char *)&dat.g);
	 byterev(2, (char *)&dat.r);
	 byterev(2, (char *)&dat.i);
	 byterev(2, (char *)&dat.z);
	 byterev(2, (char *)&dat.J);
	 byterev(2, (char *)&dat.H);
	 byterev(2, (char *)&dat.K);
      }

/* Rescale integers to units of [deg] [deg/yr] and [mag] */
      (*star)[*nstar].ra = 1e-7 * dat.ra;
      (*star)[*nstar].dec = 1e-7 * dat.cdec - 90.0;
      (*star)[*nstar].plx = 1e-5 * dat.plx / 3600.0;
      (*star)[*nstar].dplx = 1e-5 * dat.dplx / 3600.0;
      (*star)[*nstar].pmra = 1e-5 * dat.pmra / 3600.0;
      (*star)[*nstar].dpmra = 1e-5 * dat.dpmra / 3600.0;
      (*star)[*nstar].pmdec = 1e-5 * dat.pmdec / 3600.0;
      (*star)[*nstar].dpmdec = 1e-5 * dat.dpmdec / 3600.0;
      (*star)[*nstar].G = 1e-3 * dat.G;
      (*star)[*nstar].dG = 2e-3 * dat.dG;
      (*star)[*nstar].B = 1e-3 * dat.B;
      (*star)[*nstar].dB = 2e-3 * dat.dB;
      (*star)[*nstar].R = 1e-3 * dat.R;
      (*star)[*nstar].dR = 2e-3 * dat.dR;
      (*star)[*nstar].Teff = dat.Teff;
      (*star)[*nstar].AG = 1e-3 * dat.AG;
      (*star)[*nstar].dupvar = dat.dupvar;
      (*star)[*nstar].Ag = 1e-3 * dat.Ag;
      (*star)[*nstar].rp1 = 2e-1 * dat.rp1 / 3600.0;
      (*star)[*nstar].r1 = 2e-1 * dat.r1 / 3600.0;
      (*star)[*nstar].r10 = 2e-1 * dat.r10 / 3600.0;
      (*star)[*nstar].g = 1e-3 * dat.g;
      (*star)[*nstar].dg = 2e-3 * dat.dg;
      (*star)[*nstar].gchi = 1e-1 * dat.gchi;
      (*star)[*nstar].gcontrib = dat.gcontrib;
      (*star)[*nstar].r = 1e-3 * dat.r;
      (*star)[*nstar].dr = 2e-3 * dat.dr;
      (*star)[*nstar].rchi = 1e-1 * dat.rchi;
      (*star)[*nstar].rcontrib = dat.rcontrib;
      (*star)[*nstar].i = 1e-3 * dat.i;
      (*star)[*nstar].di = 2e-3 * dat.di;
      (*star)[*nstar].ichi = 1e-1 * dat.ichi;
      (*star)[*nstar].icontrib = dat.icontrib;
      (*star)[*nstar].z = 1e-3 * dat.z;
      (*star)[*nstar].dz = 2e-3 * dat.dz;
      (*star)[*nstar].zchi = 1e-1 * dat.zchi;
      (*star)[*nstar].zcontrib = dat.zcontrib;
      (*star)[*nstar].nstat = dat.nstat;
      (*star)[*nstar].J = 1e-3 * dat.J;
      (*star)[*nstar].dJ = 2e-3 * dat.dJ;
      (*star)[*nstar].H = 1e-3 * dat.H;
      (*star)[*nstar].dH = 2e-3 * dat.dH;
      (*star)[*nstar].K = 1e-3 * dat.K;
      (*star)[*nstar].dK = 2e-3 * dat.dK;

/* Is it bright enough? */
      m = MIN((*star)[*nstar].g, (*star)[*nstar].r);
      m = MIN((*star)[*nstar].i, m);
      if(m > mlim) continue;
/* Is it isolated enough? */
      if(2e-1*dat.rp1 < rlim) continue;
	    
/* Is it inside the rectangle or radius?  Skip if not. */
      P.x = cos((*star)[*nstar].dec*dr) * cos((*star)[*nstar].ra*dr);
      P.y = cos((*star)[*nstar].dec*dr) * sin((*star)[*nstar].ra*dr);
      P.z = sin((*star)[*nstar].dec*dr);
      if(rect) {
	 if(DOT(P,p1) > t1  || DOT(P,p1) < -t1 ||
	    DOT(P,p2) > t2 || DOT(P,p2) < -t2) {
	    continue;
	 }
      } else {
	 if(DOT(P,p1) < t1) continue;
      }

/* Keep the star */
      *nstar += 1;
   }
   close(fd);
   return(0);
}


/* Read a CSV file and re-write as a binary file */
int write_bin(int ra, int dec, int ndir, char **root, char *bin, char *exten)
{
   int i, n, m, nstar, fd;
   unsigned int ipi=3141592653;
   char line[1024], fin[1024], fout[1024];
   FILE *fp;
   STARDAT dat;

   long int inra, indec;
   int inplx, indplx, inpmra, indpmra, inpmdec, indpmdec;
   int inG, indG, inB, indB, inR, indR;
   int inTeff, inAG, indupvar, inAg, inrp1, inr1, inr10;
   int ing, indg, ingchi, ingcontrib, inr, indr, inrchi, inrcontrib;
   int ini, indi, inichi, inicontrib, inz, indz, inzchi, inzcontrib;
   int innstat, inJ, indJ, inH, indH, inK, indK;

/* Output file */
   snprintf(fout, 1024, "%s/%03d%+03d.%s", bin, ra, dec, exten);

/* Sanity check!  No input file matches output file! */
   for(m=0; m<ndir; m++) {
      snprintf(fin, 1024, "%s/%03d%+03d.%s", root[m], ra, dec, exten);
      if(strcmp(fin, fout) == 0) {
	 fprintf(stderr, "Input and output files are the same %s\n", fin);
	 return(-3);
      }
   }

/* Open the output file */
   if( (fd = creat(fout, 0644)) < 0) {
      fprintf(stderr, "Cannot open %s for writing\n", fout);
      return(-1);
   }

/* Write a preamble to the output: nstar, magic, lowendian */
   nstar = 0;
   if(write(fd, &nstar, 4) != 4) BOMB(4); // Update later when we know the count
   if(write(fd, &ipi, 4) != 4) BOMB(4);	  // magic marker
   i = LOWENDIAN() ? 1 : 0;
   if(write(fd, &i, 4) != 4) BOMB(4);	  // lowendian

/* Read all the data from each input file */
   for(m=0; m<ndir; m++) {

/* Open the input file */
      snprintf(fin, 1024, "%s/%03d%+03d.%s", root[m], ra, dec, exten);
      if( (fp=fopen(fin, "r")) == NULL) {
	 fprintf(stderr, "Cannot open %s for reading\n", fin);
	 return(-1);
      }

      if(VERBOSE > 0) printf("%s -> %s", fin, fout);

      for(i=0;  ; i++) {
	 if(fgets(line, 1024, fp) == NULL) break;

	 n = sscanf(line, "%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%x,%d,%d,%d,%x,%d,%d,%d,%x,%d,%d,%d,%x,%d,%d,%d,%d,%d,%d,%d",
		    &inra, &indec,
		    &inplx, &indplx, &inpmra, &indpmra, &inpmdec, &indpmdec,
		    &inG, &indG, &inB, &indB, &inR, &indR,
		    &inTeff, &inAG, &indupvar, &inAg, &inrp1, &inr1, &inr10,
		    &ing, &indg, &ingchi, &ingcontrib,
		    &inr, &indr, &inrchi, &inrcontrib,
		    &ini, &indi, &inichi, &inicontrib,
		    &inz, &indz, &inzchi, &inzcontrib,
		    &innstat, &inJ, &indJ, &inH, &indH, &inK, &indK);
	 if(n != 44) {
	    fprintf(stderr, "Only read %d items at line %d from file %s\n", n, i+1, fin);
	    fclose(fp);
	    return(-2);
	 }

/* Fill in the STARDAT structure */
	 dat.ra = (inra + 5) / 10;
	 dat.cdec = (indec + 9000000005l) / 10;
	 dat.plx = inplx;
	 dat.pmra = inpmra;
	 dat.pmdec = inpmdec;
	 dat.Teff = inTeff;
	 dat.AG = MIN(65535, inAG);
	 dat.Ag = MIN(65535, inAg);

	 dat.G = inG;
	 dat.B = inB;
	 dat.R = inR;
	 dat.g = ing;
	 dat.r = inr;
	 dat.i = ini;
	 dat.z = inz;
	 dat.J = inJ;
	 dat.H = inH;
	 dat.K = inK;

	 dat.rp1 = MIN(255, (inrp1+1)/2);
	 dat.r1 = MIN(255, (inr1+1)/2);
	 dat.r10 = MIN(255, (inr10+1)/2);

	 dat.dplx = MIN(255, indplx);
	 dat.dpmra = MIN(255, indpmra);
	 dat.dpmdec = MIN(255, indpmdec);

	 dat.dG = MIN(255, (indG+1)/2);	// Preserve 1mmag -> 1mmag
	 dat.dB = MIN(255, (indB+1)/2);
	 dat.dR = MIN(255, (indR+1)/2);
	 dat.dg = MIN(255, (indg+1)/2);
	 dat.dr = MIN(255, (indr+1)/2);
	 dat.di = MIN(255, (indi+1)/2);
	 dat.dz = MIN(255, (indz+1)/2);
	 dat.dJ = MIN(255, (indJ+1)/2);
	 dat.dH = MIN(255, (indH+1)/2);
	 dat.dK = MIN(255, (indK+1)/2);

	 dat.nstat = MIN(255, innstat);
	 dat.dupvar = indupvar;

	 dat.gchi = MIN(255, (ingchi+5)/10);
	 dat.gcontrib = ingcontrib;
	 dat.rchi = MIN(255, (inrchi+5)/10);
	 dat.rcontrib = inrcontrib;
	 dat.ichi = MIN(255, (inichi+5)/10);
	 dat.icontrib = inicontrib;
	 dat.zchi = MIN(255, (inzchi+5)/10);
	 dat.zcontrib = inzcontrib;

/* Write the star to the output */
	 if(write(fd, &dat, sizeof(STARDAT)) != sizeof(STARDAT))
	    BOMB((int)sizeof(STARDAT));

	 nstar += 1;
      }
      if(VERBOSE > 0) printf("%8d stars\n", nstar);
      fclose(fp);
   }

/* Rewind the binary file */
   lseek(fd, 0l, SEEK_SET);

/* Rewrite the number of stars */
   if(write(fd, &nstar, 4) != 4) BOMB(4);
   close(fd);

   return(nstar);
}


/* Write a single variable from a STAR structure */
int write_var(int i, STAR *star)
{
   printf(" ");
   if(strstr(varname[i].fmt, "f") != NULL) {
      printf(varname[i].fmt, varname[i].scale * 
	     *((double *)(((char *)star)+varname[i].offset)));
   } else {
      printf(varname[i].fmt, *((int *)(((char *)star)+varname[i].offset)));
   }
   return(0);
}




void syntax(char *prog)	 
{
   printf("Syntax: %s ra[deg] dec[deg] [options]\n", prog);
   printf("  returns stars close to ra, dec from ATLAS Refcat2 sqdeg files\n");
   printf("\nOptions include:\n");

   printf("   -dir P       [default P=%s]\n", rootspec);
   printf("     Read the data files from directory P\n");
   printf("   -exten X     [default X=rc2]\n");
   printf("     Data files have file names P/rrr+dd.X\n");
   printf("   -csv\n");
   printf("   -bin\n");
   printf("     Request refcat to read from CSV text or binary files of the\n");
   printf("     form P/rrr+dd.X.  The default is to attempt to auto-detect\n");
   printf("     the file type.\n");
   printf("   -mlim m      [default 17]\n");
   printf("     Return only stars with the smallest of g,r,i less than or\n");
   printf("     equal to m.\n");
   printf("   -all\n");
   printf("     Request refcat to return all 44 fields from Refcat2 according\n");
   printf("     to the units given in the Value column of the table in the man page.\n");
   printf("     A header gives the name of each of the 44 variables.  The\n");
   printf("     default is to return ATLAS minimal results consisting of\n");
   printf("     RA, Dec, g, r, i, z, J, c, and o.\n");
   printf("   -rect dR,dD   [default 0.1,0.1]\n");
   printf("   -rad R\n");
   printf("     Return stars within a circle of radius R deg or a rectangle\n");
   printf("     of size +/-dR, +/-dD deg from the N-S, E-W great circles that\n");
   printf("     pass through the central point RA, Dec.\n");
   printf("   -silent      [default]\n");
   printf("   -verb\n");
   printf("   -VERB\n");
   printf("     Request increasing levels of verbosity\n");
   printf("   -CSV_to_binary B   [default NULL]\n");
   printf("     Hijack mode!  Read all 64800 square degree files in CSV format\n");
   printf("     from directory P rewrite them in binary format in directory B.\n");
   printf("     Use carefully - apart from checking that the input and output\n");
   printf("     files are not the same, refcat is reading and writing files of\n");
   printf("     the same name in different directories and will overwrite existing\n");
   printf("     output files.\n");

   return;
}


/* Format of Refcat2 CSV file */
#if 0
////////////////////////////////////////////////////////////////
Col Varname  Entry    Units         Value        Description
 1  RA   28000001672 [10ndeg]  280.00001672~deg  RA from Gaia DR2, J2000, epoch 2015.5
 2  Dec  -1967818581 [10ndeg]  -19.67818581~deg  Dec from Gaia DR2, J2000, epoch 2015.5
 3  plx        98    [10uas]        0.98~mas     Parallax from Gaia DR2
 4  dplx       10    [10uas]        0.10~mas     Parallax uncertainty
 5  pmra      114    [10uas/yr]     1.14~mas/yr  Proper motion in RA from Gaia DR2
 6  dpmra      16    [10uas/yr]     0.16~mas/yr  Proper motion uncertainty in RA
 7  pmdec   -1460    [10uas/yr]   -14.60~mas/yr  Proper motion in Dec from Gaia DR2
 8  dpmdec     15    [10uas/yr]     0.15~mas/yr  Proper motion uncertainty in Dec
 9  Gaia    15884    [mmag]        15.884        Gaia DR2 G magnitude
10  dGaia       1    [mmag]         0.001        Gaia DR2 G magnitude uncertainty
11  BP      16472    [mmag]        16.472        Gaia G_BP magnitude
12  dBP        10    [mmag]         0.010        Gaia G_BP magnitude uncertainty
13  RP      15137    [mmag]        15.137        Gaia G_RP magnitude
14  dRP         1    [mmag]         0.001        Gaia G_RP magnitude uncertainty
15  Teff     4729    [K]           4729~K        Gaia stellar effective temperature
16  AGaia     895    [mmag]         0.895        Gaia estimate of G-band extinction for this star
17  dupvar      2    [...]          2            Gaia flags coded as CONSTANT (0), VARIABLE (1), or NOT_AVAILABLE (2) + 4*DUPLICATE
18  Ag       1234    [mmag]         1.234        SFD estimate of total column g-band extinction
19  rp1        50    [0.1asec]      5.0~arcsec   Radius where cumulative G flux exceeds 0.1x this star
20  r1         50    [0.1asec]      5.0~arcsec   Radius where cumulative G flux exceeds 1x this star
21  r10       155    [0.1asec]     15.5~arcsec   Radius where cumulative G flux exceeds 10x this star
22  g       16657    [mmag]        16.657        Pan-STARRS g_P1 magnitude
23  dg         10    [mmag]         0.010        Pan-STARRS g_P1 magnitude uncertainty
24  gchi       23    [0.01]         0.23         chi^2/DOF for contributors to g
25  gcontrib   1f    [%02x]       00011111       Bitmap of contributing catalogs to g
26  r       15915    [mmag]        15.915        Pan-STARRS r_P1 magnitude
27  dr         12    [mmag]         0.012        Pan-STARRS r_P1 magnitude uncertainty
28  rchi       41    [0.01]         0.41         chi^2/DOF for contributors to r
29  rcontrib   3f    [%02x]       00111111       Bitmap of contributing catalogs to r
30  i       15578    [mmag]        15.578        Pan-STARRS i_P1 magnitude
31  di         10    [mmag]         0.010        Pan-STARRS i_P1 magnitude uncertainty
32  ichi       49    [0.01]         0.49         chi^2/DOF for contributors to i
33  icontrib   0f    [%02x]       00001111       Bitmap of contributing catalogs to i
34  z       15346    [mmag]        15.346        Pan-STARRS z_P1 magnitude
35  dz         12    [mmag]         0.012        Pan-STARRS z_P1 magnitude uncertainty
36  zchi        0    [0.01]         0.00         chi^2/DOF for contributors to z
37  zcontrib   06    [%02x]       00000110       Bitmap of contributing catalogs to z
38  nstat       0    [...]          0            Count of griz deweighted outliers
39  J       14105    [mmag]        14.105        2MASS J magnitude
40  dJ         36    [mmag]         0.036        2MASS J magnitude uncertainty
41  H       14105    [mmag]        14.105        2MASS H magnitude
42  dH         53    [mmag]         0.053        2MASS H magnitude uncertainty
43  K       13667    [mmag]        13.667        2MASS K magnitude
44  dK         44    [mmag]         0.044        2MASS K magnitude uncertainty
////////////////////////////////////////////////////////////////
#endif
