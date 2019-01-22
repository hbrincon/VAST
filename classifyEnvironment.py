'''Identify galaxies as being in a void or not.'''


################################################################################
#
#   IMPORT LIBRARIES
#
################################################################################


import numpy as np

from astropy.table import Table

from vflag import determine_vflag


################################################################################
#
#   CONSTANTS
#
################################################################################


c = 3e5 # km/s
h = 1
H = 100*h


################################################################################
#
#   IMPORT DATA
#
################################################################################


voids = Table.read('SDSSdr7/vollim_dr7_cbp_102709_holes.txt', format='ascii.commented_header')
'''
voids['x'] == x-coordinate of center of void (in h^-1 Mpc)
voids['y'] == y-coordinate of center of void (in h^-1 Mpc)
voids['z'] == z-coordinate of center of void (in h^-1 Mpc)
voids['R'] == radius of void (in h^-1 Mpc)
voids['voidID'] == index number identifying to which void the sphere belongs
'''

#galaxy_file = input('Galaxy data file (with extension): ')
galaxy_file = '/Users/kellydouglass/Documents/Research/Rotation_curves/vflag_not_found.txt'
galaxies = Table.read(galaxy_file, format='ascii.commented_header')


################################################################################
#
#   CONVERT GALAXY ra,dec,z TO x,y,z
#
################################################################################
'''Conversions are from http://www.physics.drexel.edu/~pan/VoidCatalog/README'''


# Convert redshift to distance
galaxies_r = c*galaxies['redshift']/H

# Calculate x-coordinates
galaxies_x = galaxies_r*np.cos(galaxies['dec'])*np.cos(galaxies['ra'])

# Calculate y-coordinates
galaxies_y = galaxies_r*np.cos(galaxies['dec'])*np.sin(galaxies['ra'])

# Calculate z-coordinates
galaxies_z = galaxies_r*np.sin(galaxies['dec'])


################################################################################
#
#   IDENTIFY AS IN VOID OR NO
#
################################################################################


for i in range(len(galaxies)):
    
    galaxies['vflag'][i] = determine_vflag(galaxies_x[i],galaxies_y[i],galaxies_z[i], voids)


################################################################################
#
#   SAVE RESULTS
#
################################################################################


# Output file name
outfile = galaxy_file[:-4] + '_vflag.txt'

galaxies.write(outfile, format=)