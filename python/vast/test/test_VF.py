# -*- coding: utf-8 -*-
import unittest

import os
import numpy as np
from sklearn import neighbors
from astropy.table import Table, setdiff, vstack

from vast.voidfinder.constants import c
from vast.voidfinder import find_voids, filter_galaxies
from vast.voidfinder.multizmask import generate_mask
from vast.voidfinder.preprocessing import file_preprocess

class TestVoidFinder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up some global variables available to all test cases.
        TestVoidFinder.wall = None
        TestVoidFinder.dist_limits = None
        TestVoidFinder.mask = None
        TestVoidFinder.grid_shape = None

    def setUp(self):
        # Set up a dummy survey that can be used to test I/O, preprocessing,
        # and void finding.
        self.ra_range = np.arange(10, 30, 0.5)
        self.dec_range = np.arange(-10, 10, 0.5)
        self.redshift_range = np.arange(0, 0.011, 0.0005)

        RA, DEC, REDSHIFT = np.meshgrid(self.ra_range, self.dec_range, self.redshift_range)

        self.galaxies_table = Table()
        self.galaxies_table['ra'] = np.ravel(RA)
        self.galaxies_table['dec'] = np.ravel(DEC)
        self.galaxies_table['redshift'] = np.ravel(REDSHIFT)

        # Shuffle the table (so that the KDtree does not die)
        rng = np.random.default_rng()
        self.galaxies_shuffled = Table(rng.permutation(self.galaxies_table))
        self.galaxies_shuffled['Rgal'] = c*self.galaxies_shuffled['redshift']/100.
        N_galaxies = len(self.galaxies_shuffled)

        # All galaxies will be brighter than the magnitude limit, so that none
        # of them are removed
        self.galaxies_shuffled['rabsmag'] = 5*np.random.rand(N_galaxies) - 25.1

        self.galaxies_filename = 'test_galaxies.txt'
        self.galaxies_shuffled.write(self.galaxies_filename,
                                     format='ascii.commented_header',
                                     overwrite=True)

        self.gal = np.zeros((N_galaxies,3))
        self.gal[:,0] = self.galaxies_shuffled['Rgal']*np.cos(self.galaxies_shuffled['ra']*np.pi/180.)*np.cos(self.galaxies_shuffled['dec']*np.pi/180.)
        self.gal[:,1] = self.galaxies_shuffled['Rgal']*np.sin(self.galaxies_shuffled['ra']*np.pi/180.)*np.cos(self.galaxies_shuffled['dec']*np.pi/180.)
        self.gal[:,2] = self.galaxies_shuffled['Rgal']*np.sin(self.galaxies_shuffled['dec']*np.pi/180.)

    def test_1_file_preprocess(self):
        """Take a galaxy data file and return a data table, compute the redshift range in comoving coordinates, and generate output filename.
        """
        f_galaxy_table, f_dist_limits, f_out1_filename, f_out2_filename = \
            file_preprocess(self.galaxies_filename, '', '', dist_metric='redshift')

        # Check the galaxy table
        self.assertEqual(len(setdiff(f_galaxy_table, self.galaxies_shuffled)), 0)

        # Check the distance limits
        TestVoidFinder.dist_limits = np.zeros(2)
        TestVoidFinder.dist_limits[1] = c*self.redshift_range[-1]/100.
        self.assertTrue(np.isclose(f_dist_limits, TestVoidFinder.dist_limits).all())

        # Check the first output file name
        self.assertEqual(f_out1_filename, 'test_galaxies_redshift_maximal.txt')

        # Check the second output file name
        self.assertEqual(f_out2_filename, 'test_galaxies_redshift_holes.txt')

   
