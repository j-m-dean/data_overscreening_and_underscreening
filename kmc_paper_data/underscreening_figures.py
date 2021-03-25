import json
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from uravu.distribution import Distribution
from uravu import plotting
from uravu.relationship import Relationship
from matplotlib import rc, rcParams
from collections import OrderedDict
from uravu.relationship import Relationship
import uravu.plotting as plotting

np.random.seed(seed = 1)

# ---------------------------------------------------
# Color sets
# ---------------------------------------------------
#Standard tableau 20 set
tableau = OrderedDict([
        ("blue", "#0173B2"),
        ("orange", "#DE8F05"),
        ("green", "#029E73"),
        ("red", "#D55E00"),
        ("purple", "#CC78BC"),
        ("brown", "#CA9161"),
        ("pink", "#FBAFE4"),
        ("grey", "#949494"),
        ("yellow", "#ECE133"),
        ("turquoise", "#56B4E9"),
])
fontsize=15
nearly_black = '#333333'
light_grey = '#EEEEEE'
lighter_grey = '#F5F5F5'
white = '#ffffff'
grey = '#7F7F7F'
master_formatting = {'axes.formatter.limits': (-5,5),
                     'axes.titlepad':10,
                          'xtick.major.pad': 7,
                          'ytick.major.pad': 7,
                          'ytick.color': nearly_black,
                          'xtick.color': nearly_black,
                          'axes.labelcolor': nearly_black,
                          'axes.linewidth': .5,
                          'axes.edgecolor' : nearly_black,
                          'axes.spines.bottom': True,
                          'axes.spines.left': True,
                          'axes.spines.right': True,
                          'axes.spines.top': True,
                          'axes.axisbelow': True,
                          'legend.frameon': False,
                          'lines.linewidth': 1.25,
                          'pdf.fonttype': 42,
                          'ps.fonttype': 42,
                          'font.size': fontsize,
                          'text.usetex': False,
                          'savefig.bbox':'tight',
                          'axes.facecolor': white,
                          'axes.labelpad': 10.0,
                          'axes.labelsize': fontsize,
                          'axes.titlesize': fontsize,
                          'axes.grid': False,
                          'lines.markersize': 7.0,
                          'lines.scale_dashes': False,
                          'xtick.labelsize': fontsize,
                          'ytick.labelsize': fontsize,
                          'legend.fontsize': fontsize,
                          'figure.figsize':[5.5,5.5]}
for k, v in master_formatting.items():
    rcParams[k] = v
color_cycle = tableau.values()
try:
    from matplotlib import cycler
    rcParams['axes.prop_cycle'] = cycler(color=color_cycle)
except Exception:
    raise

# Define useful properties as function


def debye(charges, permittivity):
    """
    Calculate the Debye length for a pair of parameters (charges, permittvity).

    Args:
    charges (int): the number of charges in the get_simulation
    permittivity (float): the relative permittivity

    Returns:
    The Debye length as a float.
    """

    num = 8.854e-12 * permittivity * 1.381e-23 * 300
    conc = charges / (75 * 2.5e-10)**3
    denom = 1.602e-19**2 * conc

    return( np.sqrt(num / denom))

def wigner_seitz_radius(charges):
    """
    Calculate the Wigner-Seitz radius for a given number of charges

    Args:
    charges (int): the number of charges in the get_simulation

    Returns:
    The Wigner-Seitz radius as a float.
    """

    n = charges / (2.5e-10 * 75 )**3
    denom = 4 * np.pi * n
    return((3 / denom)**(1/3))


def bjerrum(permittivity):
    """
    Calculates the Bjerrum length for a simulation at 300 K.

    Args:
    permittivity (float): the relative permittivity value

    Returns:
    The Bjerrum length as a simulation with a given permittivity.

    """
    num = 1.602e-19**2
    denom = 4 * np.pi * 8.854e-12 * permittivity * 1.381e-23 * 300
    return num / denom

def calculate_gamma(charges, permittivity):
    """
    Calculates Gamma = bjerrum length / Wigner-Seitz radius.

    Args:
    charges(int): the number of charges in the simulation
    permittivity (float): the relative permittivity of the simulation
    """

    bje = bjerrum(permittivity)
    a_ws = wigner_seitz_radius(charges)
    return bje / a_ws

def get_simulation_decays(charge, permittivity, home_path):

    """
    Determine which model (oscillatory or exponential) is favoured, and
    extract the necessary data

    Args:
    charges (int): the number of charges in the simulation
    permittivity (float): the relative permittivity of the simulation
    home_path (str): the reference path

    Returns:
    lambda_sys (float): the most probable simulation decay
    lambda_sys_err (list with two entries): the lower and upper errors respectively
    lambda_sys_distribution (uravu Distribution object): the distribution of lambda values
    debye_length (float): the Debye length for the simulation
    model (str): "oscillatory" or "monotonic", defined by the Bayes factor
    gamma (float): the Gamma value for the simulation
    """

    path = home_path + 'charges_{}/permittivity_{}/outputs.json'.format(charge, permittivity)

    with open(path) as json_file:
        data = json.load(json_file)

    lambda_sys = []
    lambda_sys_err = []

    if data["bayes_factor"] >= 10:

        lambda_sys = 1 / data["alpha_osc_max"][0]

        lambda_sys_err.append(( data["alpha_osc_err_lower"]) / data["alpha_osc_max"][0])
        lambda_sys_err.append((data["alpha_osc_err_upper"] ) / data["alpha_osc_max"][0])

        debye_length = debye(charge, permittivity)

        lambda_sys_distribution = Distribution((1/np.array(data["alpha_osc_distribution"]))  / debye(charge, permittivity))

        model = 'oscillatory'

    else:

        lambda_sys = 1 / data["alpha_dilute_max"][0]

        lambda_sys_err.append((data["alpha_dilute_err_lower"]) / data["alpha_dilute_max"][0])
        lambda_sys_err.append((data["alpha_dilute_err_upper"] ) / data["alpha_dilute_max"][0])

        debye_length = debye(charge, permittivity)

        lambda_sys_distribution = Distribution((1/np.array(data["alpha_dilute_distribution"]))  / debye(charge, permittivity))

        model = 'monotonic'

    gamma =calculate_gamma(charge, permittivity)

    return lambda_sys, lambda_sys_err, lambda_sys_distribution, debye_length, model, gamma

home = os.getcwd() + "/"

permittivities_105 = [1,2,4,7,10,13,16,19,22,25,28,50,65, 75, 85,100]
permittivities = [1,2,4,7,10,13,16,19,22,25,28,50,75,100]

# Set variables for plotting.

lambda_sys_105 = []
lambda_sys_105_err = [[],[]]
lambda_sys_distribution_105 = []
debye_105 = []
lambda_sys_105_dilute = []
lambda_sys_105_err_dilute = [[],[]]
lambda_sys_distribution_105_dilute = []
debye_105_dilute = []
gamma_105 = []
gamma_105_dilute = []

lambda_sys_210 = []
lambda_sys_210_err = [[],[]]
lambda_sys_distribution_210 = []
debye_210 = []
lambda_sys_210_dilute = []
lambda_sys_210_err_dilute = [[],[]]
lambda_sys_distribution_210_dilute = []
debye_210_dilute = []
gamma_210 = []
gamma_210_dilute = []


lambda_sys_421 = []
lambda_sys_421_err = [[],[]]
lambda_sys_distribution_421 = []
debye_421 = []
lambda_sys_421_dilute = []
lambda_sys_421_err_dilute = [[],[]]
lambda_sys_distribution_421_dilute = []
debye_421_dilute = []
gamma_421 = []
gamma_421_dilute = []

lambda_sys_2109 = []
lambda_sys_2109_err = [[],[]]
lambda_sys_distribution_2109 = []
debye_2109 = []
lambda_sys_2109_dilute = []
lambda_sys_2109_err_dilute = [[],[]]
lambda_sys_distribution_2109_dilute = []
debye_2109_dilute = []
gamma_2109 = []
gamma_2109_dilute = []

for p in permittivities_105:
    lam_105 , lam_105_err, lambda_105_dist, deb_105, model_105, gamma = get_simulation_decays(105, p, home)

    if model_105 == 'oscillatory':
        lambda_sys_105.append(lam_105)
        debye_105.append(deb_105)
        lambda_sys_105_err[0].append(lam_105_err[0])
        lambda_sys_105_err[1].append(lam_105_err[1])
        lambda_sys_distribution_105.append(lambda_105_dist)
        gamma_105.append(gamma)
    else:
        lambda_sys_105_dilute.append(lam_105)
        debye_105_dilute.append(deb_105)
        lambda_sys_105_err_dilute[0].append(lam_105_err[0])
        lambda_sys_105_err_dilute[1].append(lam_105_err[1])
        lambda_sys_distribution_105_dilute.append(lambda_105_dist)
        gamma_105_dilute.append(gamma)

for p in permittivities:

    lam_210 , lam_210_err, lambda_210_dist, deb_210, model_210, gamma = get_simulation_decays(210, p, home)

    if model_210 == 'oscillatory':
        lambda_sys_210.append(lam_210)
        debye_210.append(deb_210)
        lambda_sys_210_err[0].append(lam_210_err[0])
        lambda_sys_210_err[1].append(lam_210_err[1])
        lambda_sys_distribution_210.append(lambda_210_dist)
        gamma_210.append(gamma)
    else:
        lambda_sys_210_dilute.append(lam_210)
        debye_210_dilute.append(deb_210)
        lambda_sys_210_err_dilute[0].append(lam_210_err[0])
        lambda_sys_210_err_dilute[1].append(lam_210_err[1])
        lambda_sys_distribution_210_dilute.append(lambda_210_dist)
        gamma_210_dilute.append(gamma)

    lam_421 , lam_421_err, lambda_421_dist, deb_421, model_421, gamma = get_simulation_decays(421, p, home)

    if model_421 == 'oscillatory':
        lambda_sys_421.append(lam_421)
        debye_421.append(deb_421)
        lambda_sys_421_err[0].append(lam_421_err[0])
        lambda_sys_421_err[1].append(lam_421_err[1])
        lambda_sys_distribution_421.append(lambda_421_dist)
        gamma_421.append(gamma)
    else:
        lambda_sys_421_dilute.append(lam_421)
        debye_421_dilute.append(deb_421)
        lambda_sys_421_err_dilute[0].append(lam_421_err[0])
        lambda_sys_421_err_dilute[1].append(lam_421_err[1])
        lambda_sys_distribution_421_dilute.append(lambda_421_dist)
        gamma_421_dilute.append(gamma)

    lam_2109 , lam_2109_err, lambda_2109_dist, deb_2109, model_2109, gamma = get_simulation_decays(2109, p, home)

    if model_2109 == 'oscillatory':
        lambda_sys_2109.append(lam_2109)
        debye_2109.append(deb_2109)
        lambda_sys_2109_err[0].append(lam_2109_err[0])
        lambda_sys_2109_err[1].append(lam_2109_err[1])
        lambda_sys_distribution_2109.append(lambda_2109_dist)
        gamma_2109.append(gamma)
    else:
        lambda_sys_2109_dilute.append(lam_2109)
        debye_2109_dilute.append(deb_2109)
        lambda_sys_2109_err_dilute[0].append(lam_2109_err[0])
        lambda_sys_2109_err_dilute[1].append(lam_2109_err[1])
        lambda_sys_distribution_2109_dilute.append(lambda_2109_dist)
        gamma_2109_dilute.append(gamma)

# Set plotting variables

x_105 = np.log(wigner_seitz_radius(105) / np.array(debye_105))
y_105 = np.log(lambda_sys_105 / np.array(debye_105))
x_105_dilute = np.log(wigner_seitz_radius(105) / np.array(debye_105_dilute))
y_105_dilute = np.log(lambda_sys_105_dilute / np.array(debye_105_dilute))
x_105_gamma = np.log(gamma_105)
x_105_dilute_gamma = np.log(gamma_105_dilute)

x_210 = np.log(wigner_seitz_radius(210) / np.array(debye_210))
y_210 = np.log(lambda_sys_210 / np.array(debye_210))
x_210_dilute = np.log(wigner_seitz_radius(210) / np.array(debye_210_dilute))
y_210_dilute = np.log(lambda_sys_210_dilute / np.array(debye_210_dilute))
x_210_gamma = np.log(gamma_210)
x_210_dilute_gamma = np.log(gamma_210_dilute)

x_421 = np.log(wigner_seitz_radius(421) / np.array(debye_421))
y_421 = np.log(lambda_sys_421 / np.array(debye_421))
x_421_dilute = np.log(wigner_seitz_radius(421) / np.array(debye_421_dilute))
y_421_dilute = np.log(lambda_sys_421_dilute / np.array(debye_421_dilute))
x_421_gamma = np.log(gamma_421)
x_421_dilute_gamma = np.log(gamma_421_dilute)

x_2109 = np.log(wigner_seitz_radius(2109) / np.array(debye_2109))
y_2109 = np.log(lambda_sys_2109 / np.array(debye_2109))
x_2109_dilute = np.log(wigner_seitz_radius(2109) / np.array(debye_2109_dilute))
y_2109_dilute = np.log(lambda_sys_2109_dilute / np.array(debye_2109_dilute))
x_2109_gamma = np.log(gamma_2109)
x_2109_dilute_gamma = np.log(gamma_2109_dilute)

# Calculate line of best fits.
# calculation for a_WS / lambda_D

all_x = np.concatenate((x_105, x_105_dilute, x_210, x_210_dilute, x_421, x_421_dilute, x_2109, x_2109_dilute))
all_y = np.concatenate((y_105, y_105_dilute, y_210, y_210_dilute, y_421, y_421_dilute, y_2109, y_2109_dilute))
all_y_distributions = lambda_sys_distribution_105 + lambda_sys_distribution_105_dilute + lambda_sys_distribution_210 + lambda_sys_distribution_210_dilute + lambda_sys_distribution_421 + lambda_sys_distribution_421_dilute + lambda_sys_distribution_2109 + lambda_sys_distribution_2109_dilute

# Only want to fit the linear region
x_to_use = []
y_to_use = []
y_distributions_to_use = []

for i,j in enumerate(all_x):
    if j > 1:
        x_to_use.append(np.exp(j))
        y_to_use.append(np.exp(all_y[i]))
        y_distributions_to_use.append(all_y_distributions[i])

 # Define fitting model
def model(x, nu, a):
    """
    Model for fitting Figure 3.

    Args:
    x (numpy array): x coordinates
    nu (float): the exponent
    a (float): the constant of proportionality

    Returns:
    yy (numpy array): the y values for the model y = ax^{nu}
    """
    yy = a * x ** nu
    return yy

# Fit for a_ws / lambda_D
r = Relationship(model, np.array(x_to_use), np.array(y_distributions_to_use), bounds=((0, 5), (-1,1)))
r.max_likelihood('diff_evo')
r.nested_sampling(progress = False)
fig, ax = plotting.plot_corner(r)
ax[0,0].tick_params(axis='x', rotation=90)
ax[0,0].tick_params(axis='y', rotation=0)
ax[1,0].set_xlabel("Exponent")
ax[1,0].set_ylabel("Constant of proportionality")
ax[1,1].set_xlabel("Constant of proportionality")
plt.savefig(home + "Figures/corner_plot_for_a_WS_lambda_D.pdf")
plt.clf()

# Fit for gamma
all_x_gamma = np.concatenate((x_105_gamma, x_105_dilute_gamma, x_210_gamma, x_210_dilute_gamma, x_421_gamma, x_421_dilute_gamma, x_2109_gamma, x_2109_dilute_gamma))

x_to_use_gamma = []
y_to_use_gamma = []
y_distributions_to_use_gamma = []

for i,j in enumerate(all_x_gamma):
    if j > 1:
        x_to_use_gamma.append(np.exp(j))
        y_to_use_gamma.append(np.exp(all_y[i]))
        y_distributions_to_use_gamma.append(all_y_distributions[i])

r2 = Relationship(model, np.array(x_to_use_gamma), np.array(y_distributions_to_use_gamma), bounds=((0, 5), (-5,5)))
r2.max_likelihood('diff_evo')
r2.nested_sampling(progress = False)

fig, ax = plotting.plot_corner(r2)
ax[0,0].tick_params(axis='x', rotation=90)
ax[0,0].tick_params(axis='y', rotation=0)
ax[1,0].set_xlabel("Exponent")
ax[1,0].set_ylabel("Constant of proportionality")
ax[1,1].set_xlabel("Constant of proportionality")
plt.savefig(home + "Figures/corner_plot_for_Gamma.pdf")
plt.clf()

plt.errorbar(x_105,y_105, lambda_sys_105_err,fmt = 'o', color = "#0173B2", label = r"$n_{\infty} = 0.00025$")
plt.errorbar(x_105_dilute, y_105_dilute, lambda_sys_105_err_dilute, fmt = 'o', color = "#0173B2", mfc='none')
plt.errorbar(x_210,y_210, lambda_sys_210_err, fmt = 'o', color = "#DE8F05", label = r"$n_{\infty} = 0.0005$")
plt.errorbar(x_210_dilute, y_210_dilute, lambda_sys_210_err_dilute, fmt = 'o', color = "#DE8F05", mfc='none')
plt.errorbar(x_421,y_421, lambda_sys_421_err,fmt = 'o', color = "#029E73", label = r"$n_{\infty} = 0.001$")
plt.errorbar(x_421_dilute, y_421_dilute, lambda_sys_421_err_dilute,fmt = 'o', color = "#029E73", mfc='none')
plt.errorbar(x_2109,y_2109, lambda_sys_2109_err,fmt = 'o', color = "#D55E00", label = r"$n_{\infty} = 0.005$")
plt.errorbar(x_2109_dilute, y_2109_dilute, lambda_sys_2109_err_dilute,fmt = 'o', color = "#D55E00", mfc='none')
plt.legend(fontsize = "x-small")
plt.hlines(0, -0.25, 2.75, linestyle = "dotted")
plt.xlabel(r"$\ln(a_{\rm{WS}} / \lambda_\mathrm{D})$")
plt.ylabel(r"$\ln(\lambda_{\rm{sys}} / \lambda_\mathrm{D})$")

plt.plot(np.log(np.array(x_to_use)), np.log(model(np.array(x_to_use), r.variables[0].dist_max[0], r.variables[1].dist_max[0])) ,color = 'black', zorder = 4 )

plt.savefig(home + "Figures/Underscreening_Wigner_Seitz.pdf")
plt.clf()

plt.errorbar(x_105_gamma,y_105, lambda_sys_105_err,fmt = 'o', color = "#0173B2", label = r"$n_{\infty} = 0.00025$")
plt.errorbar(x_105_dilute_gamma, y_105_dilute, lambda_sys_105_err_dilute, fmt = 'o', color = "#0173B2", mfc='none')
plt.errorbar(x_210_gamma,y_210, lambda_sys_210_err, fmt = 'o', color = "#DE8F05", label = r"$n_{\infty} = 0.0005$")
plt.errorbar(x_210_dilute_gamma, y_210_dilute, lambda_sys_210_err_dilute, fmt = 'o', color = "#DE8F05", mfc='none')
plt.errorbar(x_421_gamma,y_421, lambda_sys_421_err,fmt = 'o', color = "#029E73", label = r"$n_{\infty} = 0.001$")
plt.errorbar(x_421_dilute_gamma, y_421_dilute, lambda_sys_421_err_dilute,fmt = 'o', color = "#029E73", mfc='none')
plt.errorbar(x_2109_gamma,y_2109, lambda_sys_2109_err,fmt = 'o', color = "#D55E00", label = r"$n_{\infty} = 0.005$")
plt.errorbar(x_2109_dilute_gamma, y_2109_dilute, lambda_sys_2109_err_dilute,fmt = 'o', color = "#D55E00", mfc='none')
plt.legend(fontsize = "x-small")
plt.hlines(0, -1.75, 4.25, linestyle = "dotted")
plt.xlabel(r"$\ln(\Gamma )$")
plt.ylabel(r"$\ln(\lambda_{\rm{sys}} / \lambda_\mathrm{D})$")

plt.plot(np.log(np.array(x_to_use_gamma)), np.log(model(np.array(x_to_use_gamma), r2.variables[0].dist_max[0], r2.variables[1].dist_max[0])) ,color = 'black', zorder = 4 )

plt.savefig(home + "Figures/Underscreening_Gamma.pdf")
plt.clf()

nu_analysis = {"nu_a_WS": float(r.nested_sampling_results.distributions[0].dist_max[0]),
                "nu_a_WS_lower":r.nested_sampling_results.distributions[0].con_int[0],
                "nu_a_WS_upper":r.nested_sampling_results.distributions[0].con_int[1],
                "constant_of_proportionality_a_WS": float(r.nested_sampling_results.distributions[1].dist_max[0]),
                "constant_of_proportionality_a_WS_lower":r.nested_sampling_results.distributions[1].con_int[0],
                "constant_of_proportionality_a_WS_upper":r.nested_sampling_results.distributions[1].con_int[1],
                "nu_Gamma":float(r2.nested_sampling_results.distributions[0].dist_max[0]),
                "nu_Gamma_lower":r2.nested_sampling_results.distributions[0].con_int[0],
                "nu_Gamma_upper":r2.nested_sampling_results.distributions[0].con_int[1],
                "constant_of_proportionality_Gamma": float(r2.nested_sampling_results.distributions[1].dist_max[0]),
                "constant_of_proportionality_Gamma_lower":r2.nested_sampling_results.distributions[1].con_int[0],
                "constant_of_proportionality_Gamma_upper":r2.nested_sampling_results.distributions[1].con_int[1]}

with open("nu_analysis_results.json", 'w') as outfile:
    json.dump(nu_analysis, outfile)
