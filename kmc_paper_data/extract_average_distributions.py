# Import required packages

import json
import os
import numpy as np
import matplotlib.pyplot as plt

def format_distribution(path):

    """
    Format the average distribution from a kinetic Monte Carlo
    simulation such that the grain boundary is the centre
    of the distribution.

    Args:
    path (str): path to an 'average_occupancy.json' file

    Returns:
    distribution_data (numpy array): the average occupancy of
    each plane with the grain boundary centred. This average
    occupancy is given in number of charged.
    """

    # Read data
    with open(path) as json_file:
            data = json.load(json_file)

    # Get the average_occupancy data
    distribution_data = data['average_occupancy']

    # Centre the grain boundary
    distribution_data = np.roll(distribution_data,1)

    return(distribution_data)

def calculate_averaged_distributions(charges, permittivities):

    """
    Create a json file containing the distributions, standard deviations,
    and standard errors for each parameter pairing.

    Args:
    charges (int): the number of charges in the simulation
    permittivities (list of int): the permittivities considered for that number
    of charges

    Returns:
    Makes a file name "{X}_charges_distributions_errors" where X is the number
    of charges
    """

    all_distributions = np.zeros((37,len(permittivities)))

    all_errors_se = np.zeros((37,len(permittivities)))

    all_errors_sd = np.zeros((37,len(permittivities)))

    with open("./paths/{}_charges_paths.json".format(charges)) as data:
        paths = json.load(data)

    for idx, perm in enumerate(permittivities):

        relevant_paths = paths["permittivity_{}".format(perm)]

        number_of_simulations = len(relevant_paths)

        distributions = np.zeros((37,2 * number_of_simulations))

        for j, path in enumerate(relevant_paths):

            distribution_data = format_distribution(path)

            distributions[:,2 * j] = distribution_data[38:]
            distributions[:,2 * j + 1] = np.flip(distribution_data[:37])

        distribution = np.mean(distributions, axis = 1 )

        all_distributions[:,idx] = distribution

        err = np.zeros(37)

        for k in range(37):
            err[k] = np.std(distributions[k,:])


        all_errors_se[:,idx] = err / np.sqrt(2 * number_of_simulations)
        all_errors_sd[:,idx] = err

    # Create json containing distributions and errors

    data_dictionary = {}

    for i in range(len(permittivities)):
        data_dictionary["distribution_{}".format(permittivities[i])] = list(all_distributions[:,i])
        data_dictionary["standard_deviations_{}".format(permittivities[i])] = list(all_errors_sd[:,i])
        data_dictionary["standard_errors_{}".format(permittivities[i])] = list(all_errors_se[:,i])

    with open('./averaged_distributions/{}_charges_distributions_errors.json'.format(charges), 'w') as outfile:
        json.dump(data_dictionary, outfile)

if __name__ == "__main__":

    charges_105 = [105]
    permittivities_105 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 50, 75, 100, 2, 65, 85]

    for c in charges_105:
        calculate_averaged_distributions(c, permittivities_105)

    charges = [ 210, 421, 2109]
    permittivities = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 50, 75, 100, 2]

    for c in charges:
        calculate_averaged_distributions(c, permittivities)
