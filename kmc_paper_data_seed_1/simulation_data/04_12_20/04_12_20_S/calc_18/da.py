"""
Trajectory tools for use with the cubic lattice grain boundary models.
"""

import numpy as np
import os
import shutil
import json
import matplotlib.pyplot as plt
import time


def linear_cell_to_tuple(c1, repeat_units):
    """
    Convert a linear index into a tuple assuming lexicographic indexing.

    :arg c1: (int) Linear index.
    :arg repeat_units: (3 int indexable) Dimensions of grid.
    """

    c1 = int(c1)
    c1x = c1 % repeat_units[0]
    c1y = ((c1 - c1x) // repeat_units[0]) % repeat_units[1]
    c1z = (c1 - c1x - c1y * repeat_units[0]) // (repeat_units[0] * repeat_units[1])
    return c1x, c1y, c1z


def linear_sites_to_offset(site_1, site_2, N_sites_U, repeat_units, unit_cell_size, unit_cell):
    """
    Compute the offset vector for a hopping move trajectory record in the Bath
    KMC format. Ideally repeat units is at least 3 in each dimension. If a
    dimension has repeat units less than 3 then the minimum image convention
    is applied in this dimension.

    :arg site_1: (int) Linear index storing site index and repeat grid index for site 1.
    :arg site_2: (int) Linear index storing site index and repeat grid index for site 2.
    :arg repeat_units: (3 int numpy array) Number of repeat units in each dimension.
    :arg unit_cell_size: (3 float numpy array) Size of unit cell in each dimension.
    :arg unit_cell: (Nx3 float numpy array) Locations of sites in unit cell.
    :returns: (3 float numpy array) Offset vector for hop.
    """

    site_1 = int(site_1)
    site_2 = int(site_2)
    N_sites_U = int(N_sites_U)

    s1 = site_1 % N_sites_U
    c1 = (site_1 - s1) // N_sites_U
    c1x, c1y, c1z = linear_cell_to_tuple(c1, repeat_units)

    s2 = site_2 % N_sites_U
    c2 = (site_2 - s2) // N_sites_U
    c2x, c2y, c2z = linear_cell_to_tuple(c2, repeat_units)

    initial_offset = unit_cell[s2, :] - unit_cell[s1, :]

    xcell_shift = c2x - c1x
    # forwards around the pbc
    if xcell_shift < -1:
        xcell_shift = 1
    # backwards around the pbc
    elif xcell_shift > 1:
        xcell_shift = -1

    ycell_shift = c2y - c1y
    # forwards around the pbc
    if ycell_shift < -1:
        ycell_shift = 1
    # backwards around the pbc
    elif ycell_shift > 1:
        ycell_shift = -1

    zcell_shift = c2z - c1z
    # forwards around the pbc
    if zcell_shift < -1:
        zcell_shift = 1
    # backwards around the pbc
    elif zcell_shift > 1:
        zcell_shift = -1

    # add the cell offsets onto the initial offset
    initial_offset[0] += xcell_shift * unit_cell_size[0]
    initial_offset[1] += ycell_shift * unit_cell_size[1]
    initial_offset[2] += zcell_shift * unit_cell_size[2]

    # If repeat units is less than 3 in a dimension we apply the minimum image
    # convention in that dimension.

    sim_cell_size = np.multiply(repeat_units, unit_cell_size)

    for dimx in range(3):
        if repeat_units[dimx] < 3:
            if initial_offset[dimx] > 0.5 * sim_cell_size[dimx]:
                initial_offset[dimx] -= sim_cell_size[dimx]
            elif initial_offset[dimx] < -0.5 * sim_cell_size[dimx]:
                initial_offset[dimx] += sim_cell_size[dimx]

    return initial_offset


def get_xyz_index(repeat_units, chain_len, cell_id, site_id):
    """
    Convert a cell_id and site_id into a (x, y, z) int tuple that indexes into the lattice.

    :arg repeat_units: (3 int indexable) Dimensions of grid.
    :arg chain_len: (int) Side length of cubic lattice.
    :arg cell_id: (int) Linear cell id.
    :arg site_id: (int) Linear site id.
    """

    assert chain_len > 0
    assert site_id >= 0
    assert site_id < chain_len
    assert repeat_units[0] == chain_len
    assert repeat_units[1] == chain_len
    assert repeat_units[2] == 1
    assert cell_id >= 0
    assert cell_id < chain_len * chain_len

    cell_tuple = linear_cell_to_tuple(cell_id, repeat_units)
    xyz = (cell_tuple[0], cell_tuple[1], site_id)

    return xyz


def get_old_new_xyz(repeat_units, chain_len, event):
    """
    Returns (old_x, old_y, old_z), (new_x, new_y, new_z) as integer tuples that can be used to index
    into a cubic lattice.

    :arg repeat_units: (3 int indexable) Dimensions of grid.
    :arg chain_len: (int) Side length of cubic lattice.
    :arg event: Hop record from a FMMKMC simulation.
    """

    chain_len = int(chain_len)
    N_sites_U = 2 * chain_len

    old_site = event[5]
    new_site = event[6]

    old_site_index = old_site % N_sites_U
    assert old_site_index >= 0
    assert old_site_index < chain_len
    old_cell_index = (old_site - old_site_index) // N_sites_U
    assert old_cell_index >= 0
    assert old_cell_index < repeat_units[0] * repeat_units[1]
    xyz_old = get_xyz_index(repeat_units, chain_len, old_cell_index, old_site_index)

    new_site_index = new_site % N_sites_U
    assert new_site_index >= 0
    assert new_site_index < chain_len
    new_cell_index = (new_site - new_site_index) // N_sites_U
    assert new_cell_index >= 0
    assert new_cell_index < repeat_units[0] * repeat_units[1]
    xyz_new = get_xyz_index(repeat_units, chain_len, new_cell_index, new_site_index)

    return xyz_old, xyz_new


def print_average_plane_occupancy( simulation_paths,input_json, buffer=0, maximum = False):
    """
    Description: print a plot of the average plane occupancy

    Args:
    number_of_planes (int): the number of planes the simple cubic lattice has along the z axis
    paths (list of strings): a list of kmc output files in order of simulation. Ie the 0th
    item in the list is the first in the sequence of simulations
    occupation (float): the occupation of the simulation
    buffer (int): the number of kmc steps to omit in the averaging

    Returns:
    None

    """
    trajectory_data = []
    niter = 0
    with open(input_json) as json_file:
            input_parameters = json.load(json_file)

    number_of_planes = input_parameters["chain_length"]
    occupation = input_parameters["number_of_free_charges"] / input_parameters["chain_length"]**3

    for path in simulation_paths:
        with open(path) as json_file:
            data = json.load(json_file)
            trajectory_data += data['trajectory']
            if niter == 0:
                    initial_occupancies = data["initial_occupancy"]
            niter += 1

    occupied = np.zeros([number_of_planes, number_of_planes,number_of_planes])
    unoccupied = np.ones([number_of_planes, number_of_planes,number_of_planes])
    for i in initial_occupancies:
        occupied[tuple(i)] = 1
        unoccupied[tuple(i)] = 0

    #sanity check
    print("Number of occupied and unoccupied sites")
    print("Occupied:" ,np.sum(occupied))
    print("Unoccupied:", np.sum(unoccupied))

    times = np.zeros([number_of_planes, number_of_planes,number_of_planes])
    niter = 0
    total_time = 0
    start = time.time()
    if maximum == False:
        maxim = len(trajectory_data)
    else:
        maxim = maximum
    for move in trajectory_data:
        if maxim >niter > buffer:
            times += occupied * move[7]
            total_time += move[7]
        old_coords, new_coords = get_old_new_xyz([number_of_planes,number_of_planes,1], number_of_planes, move)
        # new_coords = get_old_new_xyz([number_of_planes,number_of_planes,1], number_of_planes, move)[1]
        occupied[old_coords] = 0
        occupied[new_coords] = 1
        unoccupied[old_coords] = 1
        unoccupied[new_coords] = 0
        niter += 1
        if niter % 10000 == 0:
            print(niter, time.time() - start)

    planes = [np.sum(times[:,:,i]) / total_time for i in range(number_of_planes)]
    plt.plot(planes)
    # plt.title("barrier depth = 0.1 eV, N=655,L = 32,steps = 100,000")
    plt.xlabel("z - axis")
    plt.ylabel(r"Average occupancy of plane")
    average_occupancy = number_of_planes**2*occupation
    plt.hlines(average_occupancy,0,number_of_planes-1,linestyles = 'dotted')
    plt.savefig("average_charge_distribution_by_plane.pdf")
#    plt.show()
    plt.clf()

    print("graph")
    av_occ = {"average_occupancy" : planes}
    print(input_json[:-10] +'average_occupancy.json')

    with open(input_json[:-10] +'average_occupancy.json', 'w') as outfile:
        json.dump(av_occ, outfile)
    print("done")
    return planes

simulation_paths = ["kmc_output.json"]
input_json = "input.json"
print_average_plane_occupancy( simulation_paths,input_json, buffer=500000)
