




import numpy as np
from vast.voidfinder import find_voids, calculate_grid
from vast.voidfinder.preprocessing import load_data_to_Table



wall_coords_xyz = load_data_to_Table("gadget_sim_100_256_wall.dat")

x = wall_coords_xyz['x']
y = wall_coords_xyz['y']
z = wall_coords_xyz['z']

num_gal = x.shape[0]

wall_coords_xyz = np.concatenate((x.reshape(num_gal,1),
                                  y.reshape(num_gal,1),
                                  z.reshape(num_gal,1)), axis=1)

hole_grid_edge_length = 5.0

hole_grid_shape, coords_min, coords_max = calculate_grid(wall_coords_xyz,
                                                         hole_grid_edge_length)


#xyz_limits = np.concatenate((coords_min.reshape(1,3),
#                             coords_max.reshape(1,3)), axis=0)
xyz_limits = np.array([[-50.,-50.,-50.],[50.,50.,50.]])


survey_name = "Gadget_100_256_"

out1_filename = survey_name+"maximals.txt"

out2_filename = survey_name+"voids.txt"


find_voids(wall_coords_xyz,
           #coords_min,
           #hole_grid_shape,
           survey_name,
           mask_type='periodic',
           mask=None, 
           mask_resolution=None,
           dist_limits=None,
           xyz_limits=xyz_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           hole_grid_edge_length=hole_grid_edge_length,
           galaxy_map_grid_edge_length=None,
           hole_center_iter_dist=1.0,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=survey_name+'potential_voids_list.txt',
           num_cpus=4,
           batch_size=10000,
           verbose=1,
           print_after=5.0)








