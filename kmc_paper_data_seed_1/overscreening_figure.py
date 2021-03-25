# Produce the Overscreening Figure

import json
import numpy as np
import matplotlib.pyplot as plt

# Stylistic preferences

from matplotlib import rc, rcParams
from collections import OrderedDict
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

# Import data

home = os.getcwd() + "/"

with open(home + "averaged_distributions/2109_charges_distributions_errors.json") as outfile:
    sim_data = json.load(outfile)

with open(home + "charges_2109/permittivity_1/stern_2/outputs.json") as outfile:
    fitted_data_perm_1 = json.load(outfile)

with open(home + "charges_2109/permittivity_10/stern_2/outputs.json") as outfile:
    fitted_data_perm_10 = json.load(outfile)

with open(home + "charges_2109/permittivity_100/stern_2/outputs.json") as outfile:
    fitted_data_perm_100 = json.load(outfile)

# Define x 
x = np.array(range(1,38)) * 2.5e-10

# Get simulated data
# Divide by 75**2 to convert to mole fraction
# Multiply errors by 1.96 to get 95% confidence intervals

y_sim_1 = np.array(sim_data["distribution_1"]) / 75**2
yerr_sim_1 = 1.96 * np.array(sim_data["standard_errors_1"]) / 75**2

y_sim_10 = np.array(sim_data["distribution_10"]) / 75**2
yerr_sim_10 = 1.96 * np.array(sim_data["standard_errors_10"]) / 75**2

y_sim_100 = np.array(sim_data["distribution_100"]) / 75**2
yerr_sim_100 = 1.96 * np.array(sim_data["standard_errors_100"]) / 75**2

# Smooth x for the fitted curves
x_smooth = np.linspace(6.25e-10, 37*2.5e-10, 100000)

# Define fitting functions

def purely_exponential(x, alpha, A, n_infty):
    """
    Description: A purely exponential decay of charged
    species away from a grain boundary. This
    distribution is of the form:
    n(x) = n_infty - A * exp(- alpha * x)

    Args:
    x (numpy.array): distance from the grain boundary
    alpha (float): the recipricol of the decay length
    A (float): the amplitude of the decay
    n_infty (float): the asymptotic bulk mole fraction

    Return:
    n_x (numpy.array): the charged species distribution
    moving away from a grain boundary.
    """

    n_x = n_infty - A * np.exp( - alpha * x)

    return n_x


def oscillatory_exponential(x, alpha, A, n_infty, xi,theta):
    """
    Description: A oscillatory exponential decay of charged
    species away from a grain boundary. This distribution
    is of the form:
    n(x) = n_infty - A * exp(- alpha * x) * cos( xi * x + theta)

    Args:
    x (numpy.array): distance from the grain boundary
    alpha (float): the recipricol of the decay length
    A (float): the amplitude of the decay
    n_infty (float): the asymptotic bulk mole fraction
    xi (float): (2 * pi) / xi is  the period of oscillations
    theta (float): the phase shift

    Return:
    n_x (numpy.array): the charged species distribution
    moving away from a grain boundary.
    """

    n_x = n_infty - A * np.exp( - alpha * x) * np.cos( xi * x + theta)

    return n_x

# Linear Plot

y_fitted_purely_exp_perm_1 = purely_exponential(x_smooth, fitted_data_perm_1["alpha_dilute"], fitted_data_perm_1["A_dilute"], fitted_data_perm_1["n_dilute"]) / 75**2
y_fitted_purely_exp_perm_10 = purely_exponential(x_smooth, fitted_data_perm_10["alpha_dilute"], fitted_data_perm_10["A_dilute"], fitted_data_perm_1["n_dilute"]) / 75**2
y_fitted_purely_exp_perm_100 = purely_exponential(x_smooth, fitted_data_perm_100["alpha_dilute"], fitted_data_perm_100["A_dilute"], fitted_data_perm_1["n_dilute"]) / 75**2

y_fitted_osc_exp_perm_1 = oscillatory_exponential(x_smooth, fitted_data_perm_1["alpha_osc"], fitted_data_perm_1["A_osc"], fitted_data_perm_1["n_osc"], fitted_data_perm_1["xi_osc"], fitted_data_perm_1["theta_osc"]) / 75**2
y_fitted_osc_exp_perm_10 = oscillatory_exponential(x_smooth, fitted_data_perm_10["alpha_osc"], fitted_data_perm_10["A_osc"], fitted_data_perm_10["n_osc"], fitted_data_perm_10["xi_osc"], fitted_data_perm_10["theta_osc"]) / 75**2
y_fitted_osc_exp_perm_100 = oscillatory_exponential(x_smooth, fitted_data_perm_100["alpha_osc"], fitted_data_perm_100["A_osc"], fitted_data_perm_100["n_osc"], fitted_data_perm_100["xi_osc"], fitted_data_perm_100["theta_osc"]) / 75**2

MARKERSIZE = 4.0

# linear plots
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5), sharey='row')
plt.subplots_adjust(hspace = 0.35)

axs[0].errorbar(x * 1e9, y_sim_100,yerr_sim_100, fmt = "o", label = "simulated", markersize= MARKERSIZE, zorder = 0)
axs[0].plot(x_smooth * 1e9, y_fitted_purely_exp_perm_100,  label = "exponential", alpha = 0.7, linewidth = 1.5, zorder = 1 )
axs[0].plot(x_smooth * 1e9, y_fitted_osc_exp_perm_100,label = "oscillatory", alpha = 0.7, linewidth = 1.5, zorder = 2)

axs[1].errorbar(x * 1e9, y_sim_10, yerr_sim_10, fmt = "o", label = "simulated", markersize= MARKERSIZE, zorder = 0 )
axs[1].plot(x_smooth * 1e9,  y_fitted_purely_exp_perm_10, label = "exponential", alpha = 0.7, linewidth = 2, zorder = 1)
axs[1].plot(x_smooth * 1e9, y_fitted_osc_exp_perm_10,label = "oscillatory", alpha = 0.7, linewidth = 2, zorder = 2)

axs[2].errorbar(x * 1e9, y_sim_1, yerr_sim_1, fmt = "o", label = "simulated", markersize= MARKERSIZE, zorder = 0 )
axs[2].plot(x_smooth * 1e9,  y_fitted_purely_exp_perm_1, label = "exponential", alpha = 0.7, linewidth = 2, zorder = 1)
axs[2].plot(x_smooth * 1e9, y_fitted_osc_exp_perm_1,label = "oscillatory", alpha = 0.7, linewidth = 2, zorder = 2)

axs[2].legend(loc = 4, fontsize = "x-small")


axs[0].set_ylabel(r"$ \langle n(x) \rangle$")
axs[0].set_xlabel(r"$x$ / nm")
axs[1].set_xlabel(r"$x$ / nm")
axs[2].set_xlabel(r"$x$ / nm")
axs[0].set_title(r"$\varepsilon_{r}$ = 100")
axs[1].set_title(r"$\varepsilon_{r}$ = 10")
axs[2].set_title(r"$\varepsilon_{r}$ = 1")
axs[0].hlines( 2109 / 75**3,0, 10, color = nearly_black, linestyles = "dotted")
axs[1].hlines( 2109 / 75**3,0, 10, color = nearly_black, linestyles = "dotted")
axs[2].hlines( 2109 / 75**3,0, 10, color = nearly_black, linestyles = "dotted")
axs[0].set_xticks([0, 2.5, 5, 7.5, 10])
axs[1].set_xticks([0, 2.5, 5, 7.5, 10])
axs[2].set_xticks([0, 2.5, 5, 7.5, 10])

axs[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False, bottom = True, color = grey)        # ticks along the top edge are off
axs[0].tick_params(
    axis='y',          # changes apply to the x-axis
    left=True,  color = grey)        # ticks along the top edge are off
axs[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False, bottom = True, color = grey)        # ticks along the top edge are off
axs[1].tick_params(
    axis='y',          # changes apply to the x-axis
    left=True,  color = grey)        # ticks along the top edge are off
axs[2].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False, bottom = True, color = grey)        # ticks along the top edge are off
axs[2].tick_params(
    axis='y',          # changes apply to the x-axis
    left=True,  color = grey)        # ticks along the top edge are off

plt.tight_layout()
plt.savefig(home + "Figures/Overscreening.pdf")
plt.show()

# Log Plot

# Smooth x for the fitted curves
x_smooth = np.linspace(6.25e-10, 37*2.5e-10, 750)

y_fitted_purely_exp_perm_1 = purely_exponential(x_smooth, fitted_data_perm_1["alpha_dilute"], fitted_data_perm_1["A_dilute"], fitted_data_perm_1["f_dilute"]) / 75**2
y_fitted_purely_exp_perm_10 = purely_exponential(x_smooth, fitted_data_perm_10["alpha_dilute"], fitted_data_perm_10["A_dilute"], fitted_data_perm_10["f_dilute"]) / 75**2
y_fitted_purely_exp_perm_100 = purely_exponential(x_smooth, fitted_data_perm_100["alpha_dilute"], fitted_data_perm_100["A_dilute"], fitted_data_perm_100["f_dilute"]) / 75**2

y_fitted_osc_exp_perm_1 = oscillatory_exponential(x_smooth, fitted_data_perm_1["alpha_osc"], fitted_data_perm_1["A_osc"], fitted_data_perm_1["f_osc"], fitted_data_perm_1["b_osc"], fitted_data_perm_1["c_osc"]) / 75**2
y_fitted_osc_exp_perm_10 = oscillatory_exponential(x_smooth, fitted_data_perm_10["alpha_osc"], fitted_data_perm_10["A_osc"], fitted_data_perm_10["f_osc"], fitted_data_perm_10["b_osc"], fitted_data_perm_10["c_osc"]) / 75**2
y_fitted_osc_exp_perm_100 = oscillatory_exponential(x_smooth, fitted_data_perm_100["alpha_osc"], fitted_data_perm_100["A_osc"], fitted_data_perm_100["f_osc"], fitted_data_perm_100["b_osc"], fitted_data_perm_100["c_osc"]) / 75**2

MARKERSIZE = 4.0

# linear plots
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5), sharey='row')
plt.subplots_adjust(hspace = 0.35)

axs[0].plot(x * 1e9, np.log(np.abs(y_sim_100 * 75**2 - fitted_data_perm_100["n_osc"]) ), "o", label = "simulated", markersize= MARKERSIZE, zorder = 0)
axs[0].plot(x_smooth * 1e9, np.log(np.abs(y_fitted_purely_exp_perm_100  * 75**2 - fitted_data_perm_100["n_dilute"]) ),label = "exponential", alpha = 0.7, linewidth = 1.5, zorder = 1)
axs[0].plot(x_smooth * 1e9, np.log(np.abs(y_fitted_osc_exp_perm_100  * 75**2 - fitted_data_perm_100["n_osc"]) ),label = "oscillatory", alpha = 0.7, linewidth = 1.5, zorder = 2)

axs[1].plot(x * 1e9, np.log(np.abs(y_sim_10 * 75**2 - fitted_data_perm_10["n_osc"])), "o", label = "simulated", markersize= MARKERSIZE, zorder = 0)
axs[1].plot(x_smooth * 1e9, np.log(np.abs(y_fitted_purely_exp_perm_10  * 75**2 - fitted_data_perm_10["n_dilute"]) ),label = "exponential", alpha = 0.7, linewidth = 1.5, zorder = 1)
axs[1].plot(x_smooth * 1e9, np.log(np.abs(y_fitted_osc_exp_perm_10 * 75**2 - fitted_data_perm_10["n_osc"]) ),label = "oscillatory", alpha = 0.7, linewidth = 1.5, zorder = 2)

axs[2].plot(x * 1e9, np.log(np.abs(y_sim_1 * 75**2 - fitted_data_perm_1["n_osc"]) ), "o", label = "simulated", markersize= MARKERSIZE, zorder = 0)
axs[2].plot(x_smooth * 1e9, np.log(np.abs(y_fitted_purely_exp_perm_1  * 75**2 - fitted_data_perm_1["n_dilute"]) ),label = "exponential", alpha = 0.7, linewidth = 1.5, zorder = 1)
axs[2].plot(x_smooth * 1e9, np.log(np.abs(y_fitted_osc_exp_perm_1 * 75**2 - fitted_data_perm_1["n_osc"]) ),label = "oscillatory", alpha = 0.7, linewidth = 1.5, zorder = 2)

axs[2].legend(loc = 1, fontsize = "x-small")


axs[0].set_ylabel(r"$\ln(|\langle n(x) \rangle - n_{\infty}| $")
axs[0].set_xlabel(r"$x$ / nm")
axs[1].set_xlabel(r"$x$ / nm")
axs[2].set_xlabel(r"$x$ / nm")
axs[0].set_title(r"$\varepsilon_{r}$ = 100")
axs[1].set_title(r"$\varepsilon_{r}$ = 10")
axs[2].set_title(r"$\varepsilon_{r}$ = 1")

axs[0].set_xticks([0, 2.5, 5, 7.5, 10])
axs[1].set_xticks([0, 2.5, 5, 7.5, 10])
axs[2].set_xticks([0, 2.5, 5, 7.5, 10])

axs[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False, bottom = True, color = grey)        # ticks along the top edge are off
axs[0].tick_params(
    axis='y',          # changes apply to the x-axis
    left=True,  color = grey)        # ticks along the top edge are off
axs[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False, bottom = True, color = grey)        # ticks along the top edge are off
axs[1].tick_params(
    axis='y',          # changes apply to the x-axis
    left=True,  color = grey)        # ticks along the top edge are off
axs[2].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False, bottom = True, color = grey)        # ticks along the top edge are off
axs[2].tick_params(
    axis='y',          # changes apply to the x-axis
    left=True,  color = grey)        # ticks along the top edge are off


axs[0].set_ylim(-8, 4)
plt.tight_layout()
plt.savefig(home + "Overscreening_logarithmic.pdf")
plt.show()
