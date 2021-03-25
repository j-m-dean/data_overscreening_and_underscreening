"""
The output can be read from disk with
import pickle
with open('kmc_simulation.pickle', 'rb') as fh:
    output = pickle.load(fh)


`output` is a dictonary with keys 'configuration' and 'trajectory'. The configuration
is a dictonary of the parameters that the kmc-fmm instance used, i.e. the passed
parameters and computed defaults such as rng seeds. The trajectory is a list of hop
records. Each hop record is a list with the following:

    [
        1,                                      # Event type. 1 is a hop.
        particle_type,                          # First particle type in event.
        -1,                                     # Second particle type in event.
        particle_ID,                            # First particle ID in event.
        -1,                                     # Second particle ID in event.
        site_1_idx + (N_sites_U*old_cell_ID),   # First site of event (see below).
        site_2_idx + (N_sites_U*new_cell_ID),   # Second site of event (see below).
        time                                    # waiting time of event.
    ]


The sites are indexed by a single integer A,
To recover the site type S, S = A % N_sites_U

The linear (lexicographic) index of the cell containing the site is computed as,
    C = Cx + R[0]*Cy + R[0]*R[1]*Cz,
where R = repeat_units.

The linear index C is recovered by, C = (A - S) // N_sites_U
The values of Cx, Cy and Cz are recovered by:

Cx = C % R[0],
Cy = ((C - Cx) // R[0]) % R[1]
Cz = (C - Cx - Cy * R[0]) // (R[0] * R[1])
where // denotes integer division. Double check my maths here.
"""

import copy
import random
import numpy as np
from kmc_fmm import kmc_simulation, kernels
import ctypes
REAL = ctypes.c_double
INT64 = ctypes.c_int64
from itertools import product
import pickle
import json
import os.path
from kmc_fmm import kmc_lattice_simulation

from datetime import datetime

def main():

    start = datetime.now()

    start_time = start.strftime("%H:%M:%S")

    verbose = True

    with open('input.json') as json_file:
        parameters = json.load(json_file)

    rng = np.random.RandomState()

    # Number of free charges
    N_free = parameters['number_of_free_charges']
    # charge value of free charges
    Q_free = parameters['charge_of_free_charges']

    # number of sites in the 1D chain. Essentially defines the size of the simulation
    chain_len = parameters['chain_length']

    # Number of KMC steps
    N_steps = parameters['number_of_kmc_steps']

    # lattice constant
    a = parameters['lattice_constant']

    # applied voltage for the simulation
    #applied_voltage = parameters['applied_voltage']

    # Evaluation point is halfway along the hop
    hop_eval_scale = parameters['linear_hop_evaluation_scale']

    # number of site types in the unit cell
    # (+1 for the compensating charge site)
    N_sites_U = chain_len * 2

    # Each site has 6 neighbours, one for each Cartesian direction
    max_neighbours = parameters['max_number_of_neighbours']

    unit_cell_size = np.array((a, a, float(chain_len) * a))
    repeat_units  = np.array((chain_len, chain_len, 1))

    # simulation cell size
    sim_cell_size = np.multiply(unit_cell_size, repeat_units)


    # setup a generic rate of the form:
    # A * exp( B * (C*du + D*df + E)**F )
    #
    # A,B,C,D,E can be scalars or (N_sites_U x max_neighbours) numpy arrays
    # F is a non-zero positive integer


    # some kind of rate constant
    A = parameters['A']
    # should be something like -1/KbT
    B = parameters['B']
    # any coefficient that goes in front of the change in electrostatic energy
    C = parameters['C']
    # coefficient in front of change in applied field term
    D = parameters['D']
    # site-site terms are passed as a numpy matrix
    Esite_site = np.ones((N_sites_U, max_neighbours)) * parameters['E_hop']

    # find the "middle" site
    mid_point = int(chain_len // 2) - 1


    # want mid_point to mid_point+1 to have an energy barrier. i.e. E_ij should be
    # "large" (for i=mid_point and j=mid_point+1) and the reverse direction?
    # -> needs thought

    # Downwards z direction is offset number 2 (base 0 indexing)
    Esite_site[mid_point, 2] += parameters['barrier_depth']

    # Upwards z direction is offset number 5
    Esite_site[mid_point, 5] += parameters['barrier_depth']
    # Esite_site = parameters['Esite_site']

    # exponent
    F =  parameters['F']

    # create the object that generates the desired kernel of the form
    # A * exp( B * (C*du + D*df + E)**F )
    boltzmann_rate_generator = kernels.BoltzmannFactorHopping(
                                                              num_sites=N_sites_U,
                                                              max_num_neighbours=max_neighbours,
                                                              A=A,
                                                              B=B,
                                                              C=C,
                                                              D=D,
                                                              E=Esite_site,
                                                              F=F
                                                              )




    # --------------------------------------------------------------------------------
    # Below this line is Python that is less likely to be modified
    # --------------------------------------------------------------------------------



    # simulation cell is a cube with side length E
    E = sim_cell_size[0]

    # check simulation domain is cubic
    assert abs(sim_cell_size[1] - E) < 10.**-14
    assert abs(sim_cell_size[2] - E) < 10.**-14

    # Compute the number of fixed charges based on the lattice
    N_fixed = chain_len ** 3
    Q_fixed = -1.0 * Q_free * N_free / N_fixed

    # total number of charges
    N_part = N_fixed + N_free

    # arrays to populate with initial particle data
    particle_types      = np.zeros((N_part, 1), INT64)
    particle_global_ID  = np.arange(0, N_part).reshape((N_part, 1)) + 200
    particle_pos        = np.zeros((N_part, 3), REAL)
    part_sites          = np.zeros((N_part, 1), INT64)
    charges             = np.zeros((N_part, 1), REAL)


    # place the fixed charges in the holes between the lattice sites.
    tmp_fix_count = 0
    for pxi, px in enumerate(
            product(
                range(chain_len), range(chain_len), range(chain_len)
            )
        ):

        particle_pos[pxi, :] = np.array(px) * a - E * 0.5
        particle_types[pxi, 0] = 2
        charges[pxi, 0] = Q_fixed
        part_sites[pxi, 0] = chain_len + px[2]
        # these have site type 0
        tmp_fix_count += 1

    assert tmp_fix_count == N_fixed


    # Find a random lattice site to occupy with a mobile charge
    assert N_free < chain_len ** 3

    if os.path.exists("occupied_sites.json"):
        with open('occupied_sites.json', 'r') as fp:
            data = json.load(fp)

        occupied = data["occupied_coords"]

        for px in range(N_free):
            ind = occupied[px]
            part_sites[px + N_fixed, 0] = ind[2]

            particle_pos[px + N_fixed, :] = np.multiply(
                                                        ind,
                                                        np.array((a, a, a))
                                                        ) - E * 0.5 + 0.5 * a
            particle_types[px + N_fixed, 0] = 102
            charges[px + N_fixed, 0] = Q_free

    else:
        def get_ind(): return tuple(rng.randint(0, repeat_units[0], size=3))
        occupied = []
        sample = random.sample(range(0,chain_len**3), N_free)
        for px in range(N_free):
            z = sample[px] // (chain_len**2)
            y = (sample[px] % (chain_len**2)) // chain_len
            x = (sample[px] % (chain_len**2)) % chain_len
            # ind = get_ind()
            # while ind in occupied:
            #     ind = get_ind()
            ind = tuple([x,y,z])
            occupied.append(ind)

            part_sites[px + N_fixed, 0] = ind[2]

            particle_pos[px + N_fixed, :] = np.multiply(
                                                        ind,
                                                        np.array((a, a, a))
                                                        ) - E * 0.5 + 0.5 * a
            particle_types[px + N_fixed, 0] = 102
            charges[px + N_fixed, 0] = Q_free
    initial_occupancy = [ occ for occ in occupied ]
    initial_occupancy = [[int(num) for num in occ] for occ in initial_occupancy ]
    # print(initial_occupancy)
    # print(type(initial_occupancy))
    initial_occupancy_ID = [ int(pos[0] + chain_len * pos[1] + chain_len**2 * pos[2]) for pos in initial_occupancy]
    # print(initial_occupancy_ID)
    # print(type(initial_occupancy_ID))
    # check net neutrality
    total_charge = np.sum(charges)
    assert abs(total_charge) < 10.**-10




    # create the offset map from each site to a neighbour that can be hopped to
    # site 0 is the "null" site where the fixed charges are placed.
    site_to_real_space_offset_map = np.zeros((N_sites_U, max_neighbours, 3), REAL)
    for sx in range(chain_len):
        site_to_real_space_offset_map[sx, :, :] = (
            (-1 * a,  0,  0),
            ( 0, -1 * a,  0),
            ( 0,  0, -1 * a),
            ( 1 * a,  0,  0),
            ( 0,  1 * a,  0),
            ( 0,  0,  1 * a),
        )


    # set the number of neighbours of each site
    site_to_N_neighbours_map = np.zeros(N_sites_U, INT64)
    # 0 has no neighbours, the rest have max_neigbours
    site_to_N_neighbours_map[:chain_len] = max_neighbours

    # Define the topology for each site
    # Each site has 6 neighbours with the offset vectors defined above.
    # Here we list what the site ids are for each of the above offset vectors.
    site_to_neighbours_map = np.zeros((N_sites_U, max_neighbours), INT64)
    for sx in range(chain_len):
        site_to_neighbours_map[sx, :] = (
            sx,
            sx,
            ((sx - 1 + chain_len) % chain_len),
            sx,
            sx,
            ((sx + 1 + chain_len) % chain_len),
        )


    # define the unit positions and unit sites. These are not strictly needed but enable
    # additional self checking.

    # We set the first `chain_len` sites to be the hopping sites and the remaining `chain_len`
    # sites to be the static compenstating charge sites.


    # the unit lattice has both the hopping sites and the compenstating charge sites
    N_unit = chain_len * 2

    unit_positions = np.zeros((N_unit, 3))

    # define the hopping sites
    unit_positions[:chain_len, 0] = 0.5 * a
    unit_positions[:chain_len, 1] = 0.5 * a
    unit_positions[:chain_len, 2] = [(0.5 * a) + (a * sx) for sx in range(chain_len)]

    # define the fixed compenstating sites
    unit_positions[chain_len:, 0] = 0
    unit_positions[chain_len:, 1] = 0
    unit_positions[chain_len:, 2] = [(a * sx) for sx in range(chain_len)]

    # define the sites types in the unit cell
    unit_sites = np.zeros(N_unit)
    unit_sites[:chain_len] = range(0, chain_len)
    unit_sites[chain_len:] = range(chain_len, 2*chain_len)



    # Define the site-site constants (for Marcus rates) using matrices
    # We don't use these but I have yet to remove the assumption of marcus rates from kmc-fmm
    # Hence these can be ignored.
    transfer_integral_matrix = np.ones((N_sites_U, max_neighbours))
    pair_energies_matrix = np.zeros((N_sites_U, max_neighbours))



    # create the configuration dict that is used to create the KMC simulation
    config = {
        'applied_field'               : 0.0,
        'verbose_debug'                 : verbose,
        'N_part'                        : N_part,
        'N_steps'                       : N_steps,
        'max_neighbours'                : max_neighbours,
        'unit_cell_size'                : unit_cell_size,
        'repeat_units'                  : repeat_units,
        'N_sites_U'                     : N_sites_U,
        'E'                             : E,
        'particle_types'                : particle_types,
        'particle_global_ID'            : particle_global_ID,
        'particle_pos'                  : particle_pos,
        'part_sites'                    : part_sites,
        'charges'                       : charges,
        'transfer_integral_matrix'      : transfer_integral_matrix,
        'pair_energies_matrix'          : pair_energies_matrix,
        'site_to_real_space_offset_map' : site_to_real_space_offset_map,
        'site_to_N_neighbours_map'      : site_to_N_neighbours_map,
        'sim_cell_size'                 : sim_cell_size,
        'site_to_neighbours_map'        : site_to_neighbours_map,
        'write_result'                  : False,
        'hop_eval_scale'                : hop_eval_scale,
        'num_fmm_levels'                : None,
        'boltzmann_factor_hopping'      : boltzmann_rate_generator,
        'unit_positions'                : unit_positions,
        'unit_sites'                    : unit_sites
         }

    # create a KMC simulation
    sim = kmc_lattice_simulation.KMCSimulation(config)


    paraview_write = False
    if paraview_write:
        # this is my class that outputs each hop for use in Paraview.
        # requires "pyevtk" to be installed (e.g. "pip install pyevtk")
        # by default the first argument to the script is used as a directory to dump vtk files

        from kmc_fmm.kmc_tools import VTKWriter
        import sys
        KMCVTK = VTKWriter(sys.argv[1], 'a', sim.KMC)

        # func_handles is a list of function handles. Each function handle is called after a KMC step.
        sim.func_handles.append(KMCVTK)

    #def print_last_waiting_time():
     #   print(sim.traj_record[-1][-1])
    #sim.func_handles.append(print_last_waiting_time)

    # run the simulation
    sim.run()


    # save the config and trajectory to disk

    # bare minimum as json
    output_dict = dict()
    output_dict['trajectory'] = sim.traj_record
    output_dict['N_sites_U'] = int(N_sites_U)
    output_dict['repeat_units'] = [int(repeat_units[0]), int(repeat_units[1]), int(repeat_units[2])]
    output_dict['unit_cell_size'] = list(unit_cell_size.astype('float64'))
    output_dict['unit_cell'] = [list(rx) for rx in unit_positions]
    output_dict['unit_sites'] =[int(sx) for sx in unit_sites]
    output_dict['initial_occupancy'] = initial_occupancy
    output_dict['initial_occupancy_ID'] = initial_occupancy_ID

    with open('kmc_output.json', 'w') as fh:
        fh.write(json.dumps(output_dict, indent=2))


    # full dump of information as pickle
    output = {
        'configuration': sim.parameters,
        'trajectory': sim.traj_record
    }

    #with open('kmc_simulation.pickle', 'wb') as fh:
     #   pickle.dump(output, fh)


    finish = datetime.now()

    finish_time = finish.strftime("%H:%M:%S")
    print("Start Time =", start_time)
    print("Finish Time =", finish_time)
    print(finish-start)
if __name__ == '__main__':
    main()
