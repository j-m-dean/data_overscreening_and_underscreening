import json
import numpy as np
import os
home = os.getcwd() + "/"

with open(home + "averaged_distributions/105_charges_distributions_errors.json") as file:
  data = json.load(file)

x = list(np.array(range(4, 38)) * 2.5e-10)

permittivities = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 50, 75, 100, 2, 65, 85]

os.chdir(home + "charges_105")

for perm in permittivities:
    os.chdir(home + "charges_105/permittivity_{}".format(perm))

    y = data["distribution_{}".format(perm)][3:]
    yerr = data["standard_errors_{}".format(perm)][3:]

    input = {"x": x, "y" : y, "yerr":yerr}

    with open(home + "charges_105/permittivity_{}/inputs.json".format(perm), 'w') as d:
        json.dump(input,d)

    os.chdir(home + "charges_105")
