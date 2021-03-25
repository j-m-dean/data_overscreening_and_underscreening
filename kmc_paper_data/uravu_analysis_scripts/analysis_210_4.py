# Uravu analysis for simulations with 105 charges.

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from uravu.distribution import Distribution
from uravu import plotting
from uravu.relationship import Relationship
from uravu.utils import bayes_factor
import sys

np.random.seed(seed = 4)

class Simulation:

    """
    A class describing a simulation to be analysed.
    """
    def __init__(self, x, y, yerr, bounds, path_to_input):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.bounds = bounds
        self.path_to_input = path_to_input

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
        xi (float): (2 * pi) / xi is equivalent of the period of
        oscillations
        theta (float): the phase shift

        Return:
        n_x (numpy.array): the charged species distribution
        moving away from a grain boundary.
        """

    n_x = n_infty - A * np.exp( - alpha * x) * np.cos( xi * x + theta)

    return n_x

def worker(simulation):
    """thread worker function"""
    x = np.array(simulation.x)
    y = np.array(simulation.y)
    yerr = np.array(simulation.yerr)
    bounds = simulation.bounds
    path_to_input = simulation.path_to_input

    r = Relationship(oscillatory_exponential, x, y, ordinate_error=yerr, bounds=bounds)
    r2 = Relationship(purely_exponential, x, y, ordinate_error=yerr, bounds=bounds[:-2])

    r.max_likelihood('diff_evo')
    r2.max_likelihood('diff_evo')

    r.nested_sampling(progress = False)

    alpha_osc = r.nested_sampling_results.distributions[0].n
    alpha_osc_err_lower = float(r.nested_sampling_results.distributions[0].dist_max[0] - r.nested_sampling_results.distributions[0].con_int[0])
    alpha_osc_err_upper = float(-r.nested_sampling_results.distributions[0].dist_max[0] + r.nested_sampling_results.distributions[0].con_int[1])
    alpha_osc_sd = r.nested_sampling_results.distributions[0].s
    alpha_osc_max = list(r.nested_sampling_results.distributions[0].dist_max)
    A_osc = r.nested_sampling_results.distributions[1].n
    A_osc_err_lower = float(r.nested_sampling_results.distributions[1].dist_max[0] - r.nested_sampling_results.distributions[1].con_int[0])
    A_osc_err_upper = float(-r.nested_sampling_results.distributions[1].dist_max[0] + r.nested_sampling_results.distributions[1].con_int[1])
    A_osc_sd = r.nested_sampling_results.distributions[1].s
    A_osc_max = list(r.nested_sampling_results.distributions[1].dist_max)
    n_osc = r.nested_sampling_results.distributions[2].n
    n_osc_err_lower = float(r.nested_sampling_results.distributions[2].dist_max[0] - r.nested_sampling_results.distributions[2].con_int[0])
    n_osc_err_upper = float(-r.nested_sampling_results.distributions[2].dist_max[0] + r.nested_sampling_results.distributions[2].con_int[1])
    n_osc_sd= r.nested_sampling_results.distributions[2].s
    n_osc_max = list(r.nested_sampling_results.distributions[2].dist_max)
    xi_osc = r.nested_sampling_results.distributions[3].n
    xi_osc_err_lower = float(r.nested_sampling_results.distributions[3].dist_max[0] - r.nested_sampling_results.distributions[3].con_int[0])
    xi_osc_err_upper = float(-r.nested_sampling_results.distributions[3].dist_max[0] + r.nested_sampling_results.distributions[3].con_int[1])
    xi_osc_sd= r.nested_sampling_results.distributions[3].s
    xi_osc_max = list(r.nested_sampling_results.distributions[3].dist_max)
    theta_osc = r.nested_sampling_results.distributions[4].n
    theta_osc_err_lower = float(r.nested_sampling_results.distributions[4].dist_max[0] - r.nested_sampling_results.distributions[4].con_int[0])
    theta_osc_err_upper = float(-r.nested_sampling_results.distributions[4].dist_max[0] + r.nested_sampling_results.distributions[4].con_int[1])
    theta_osc_sd = r.nested_sampling_results.distributions[4].s
    theta_osc_max = list(r.nested_sampling_results.distributions[4].dist_max)

    r2.nested_sampling(progress = False)

    alpha_dilute = r2.nested_sampling_results.distributions[0].n
    alpha_dilute_err_lower = float(r2.nested_sampling_results.distributions[0].dist_max[0] - r2.nested_sampling_results.distributions[0].con_int[0])
    alpha_dilute_err_upper = float(-r2.nested_sampling_results.distributions[0].dist_max[0] + r2.nested_sampling_results.distributions[0].con_int[1])
    alpha_dilute_sd = r2.nested_sampling_results.distributions[0].s
    alpha_dilute_max = list(r2.nested_sampling_results.distributions[0].dist_max)
    A_dilute = r2.nested_sampling_results.distributions[1].n
    A_dilute_err_lower = float(r2.nested_sampling_results.distributions[1].dist_max[0] - r2.nested_sampling_results.distributions[1].con_int[0])
    A_dilute_err_upper = float(-r2.nested_sampling_results.distributions[1].dist_max[0] + r2.nested_sampling_results.distributions[1].con_int[1])
    A_dilute_sd = r2.nested_sampling_results.distributions[1].s
    A_dilute_max = list(r2.nested_sampling_results.distributions[1].dist_max)
    n_dilute = r2.nested_sampling_results.distributions[2].n
    n_dilute_err_lower = float(r2.nested_sampling_results.distributions[2].dist_max[0] - r2.nested_sampling_results.distributions[2].con_int[0])
    n_dilute_err_upper = float(-r2.nested_sampling_results.distributions[2].dist_max[0] + r2.nested_sampling_results.distributions[2].con_int[1])
    n_dilute_sd = r2.nested_sampling_results.distributions[2].s
    n_dilute_max = list(r2.nested_sampling_results.distributions[2].dist_max)

    a = bayes_factor(r.ln_evidence, r2.ln_evidence)

    if alpha_osc_max == None:
        alpha_osc_max = 0
    if A_osc_max == None:
        a_osc_max = 0
    if n_osc_max == None:
        n_osc_max = 0
    if xi_osc_max == None:
        xi_osc_max = 0
    if theta_osc_max == None:
        theta_osc_max = 0
    if alpha_dilute_max == None:
        alpha_dilute_max = 0
    if A_dilute_max == None:
        A_dilute_max = 0
    if n_dilute_max == None:
        n_dilute_max = 0
    if alpha_osc == None:
        alpha_osc = 0
    if  alpha_osc_err_lower== None:
        alpha_osc_err_lower = 0
    if  alpha_osc_err_upper== None:
        alpha_osc_err_upper = 0
    if alpha_osc_sd== None:
        alpha_osc_sd = 0
    if A_osc== None:
        A_osc = 0
    if A_osc_err_lower== None:
        A_osc_err_lower = 0
    if A_osc_err_upper== None:
        A_osc_err_upper = 0
    if A_osc_sd== None:
        A_osc_sd = 0
    if n_osc== None:
        n_osc = 0
    if n_osc_err_lower== None:
        n_osc_err_lower = 0
    if n_osc_err_upper== None:
        n_osc_err_upper = 0
    if n_osc_sd== None:
        n_osc_sd = 0
    if xi_osc== None:
        xi_osc = 0
    if xi_osc_err_lower== None:
        xi_osc_err_lower = 0
    if xi_osc_err_upper== None:
        xi_osc_err_upper = 0
    if xi_osc_sd== None:
        xi_osc_sd = 0
    if theta_osc== None:
        theta_osc = 0
    if theta_osc_err_lower== None:
        theta_osc_err_lower = 0
    if theta_osc_err_upper== None:
        theta_osc_err_upper = 0
    if theta_osc_sd== None:
        theta_osc_sd = 0
    if alpha_dilute== None:
        alpha_dilute = 0
    if alpha_dilute_err_lower== None:
        alpha_dilute_err_lower = 0
    if alpha_dilute_err_upper== None:
        alpha_dilute_err_upper = 0
    if alpha_dilute_sd== None:
        alpha_dilute_sd = 0
    if A_dilute== None:
        A_dilute = 0
    if A_dilute_err_lower== None:
        A_dilute_err_lower = 0
    if A_dilute_err_upper== None:
        A_dilute_err_upper = 0
    if A_dilute_sd== None:
        A_dilute_sd = 0
    if n_dilute== None:
        n_dilute = 0
    if n_dilute_err_lower== None:
        n_dilute_err_lower = 0
    if n_dilute_err_upper== None:
        n_dilute_err_upper = 0
    if n_dilute_sd== None:
        n_dilute_sd = 0
    if a == None:
        a = 0
    else: a = float(a.nominal_value)
    data = {
    "alpha_osc_max" : alpha_osc_max,
    "A_osc_max" : A_osc_max,
    "n_osc_max" : n_osc_max,
    "xi_osc_max" : xi_osc_max,
    "theta_osc_max" : theta_osc_max,
    "alpha_dilute_max" : alpha_dilute_max,
    "A_dilute_max" : A_dilute_max,
    "n_dilute_max" : n_dilute_max,
    "alpha_osc" : alpha_osc,
    "alpha_osc_err_lower" : alpha_osc_err_lower,
    "alpha_osc_err_upper" : alpha_osc_err_upper,
    "alpha_osc_sd" : alpha_osc_sd,
    "alpha_osc_distribution" : list(r.variables[0].samples),
    "A_osc" : A_osc,
    "A_osc_err_lower" : A_osc_err_lower,
    "A_osc_err_upper" : A_osc_err_upper,
    "A_osc_sd" : A_osc_sd,
    "A_osc_distribution" : list(r.variables[1].samples),
    "n_osc" : n_osc,
    "n_osc_err_lower" : n_osc_err_lower,
    "n_osc_err_upper" : n_osc_err_upper,
    "n_osc_sd" : n_osc_sd,
    "n_osc_distribution" : list(r.variables[2].samples),
    "xi_osc" : xi_osc,
    "xi_osc_err_lower" : xi_osc_err_lower,
    "xi_osc_err_upper" : xi_osc_err_upper,
    "xi_osc_sd" :xi_osc_sd,
    "xi_osc_distribution" : list(r.variables[3].samples),
    "theta_osc" : theta_osc,
    "theta_osc_err_lower" : theta_osc_err_lower,
    "theta_osc_err_upper" : theta_osc_err_upper,
    "theta_osc_sd" : theta_osc_sd,
    "theta_osc_distribution" : list(r.variables[4].samples),
    "alpha_dilute" : alpha_dilute,
    "alpha_dilute_err_lower" : alpha_dilute_err_lower,
    "alpha_dilute_err_upper" : alpha_dilute_err_upper,
    "alpha_dilute_sd" : alpha_dilute_sd,
    "alpha_dilute_distribution" : list(r2.variables[0].samples),
    "A_dilute" : A_dilute,
    "A_dilute_err_lower" : A_dilute_err_lower,
    "A_dilute_err_upper" : A_dilute_err_upper,
    "A_dilute_sd" : A_dilute_sd,
    "A_dilute_distribution" : list(r2.variables[1].samples),
    "n_dilute" : n_dilute,
    "n_dilute_err_lower" : n_dilute_err_lower,
    "n_dilute_err_upper" : n_dilute_err_upper,
    "n_dilute_sd" : n_dilute_sd,
    "n_dilute_distribution" : list(r2.variables[2].samples),
    "bayes_factor" : a,
    "oscillatory_evidence": r.ln_evidence.nominal_value,
    "dilute_evidence": r2.ln_evidence.nominal_value,
    "oscillatory_evidence_sd": r.ln_evidence.std_dev,
    "dilute_evidence_sd": r2.ln_evidence.std_dev,
    }

    with open(path_to_input[:-11] + 'outputs.json', 'w') as outfile:
        json.dump(data, outfile)


    fig, ax = plotting.plot_corner(r)
    fig.set_size_inches(10, 10)

    alpha_offset_str ="-" + str("{:e}".format(ax[3,0].dataLim._points[0][0])[-2:] )
    xi_offset_str = "-" +str("{:e}".format(ax[3,0].dataLim._points[0][1])[-2:])


    ax[4,0].set_xlabel(r"$\alpha$" + r" / m$^{-1}$")
    ax[4,1].set_xlabel(r"$A$")
    ax[4,2].set_xlabel(r"$n_{\infty}$")
    ax[4,3].set_xlabel(r"$\xi$" + r" / m$^{-1}$")
    ax[4,4].set_xlabel(r"$\theta$")

    ax[1,0].set_ylabel(r"$A$")
    ax[2,0].set_ylabel(r"$n_{\infty}$")
    ax[3,0].set_ylabel(r"$\xi$" + r" / m$^{-1}$")
    ax[4,0].set_ylabel(r"$\theta$")


    ax[1,0].tick_params(axis='y', rotation=0)
    ax[2,0].tick_params(axis='y', rotation=0)
    ax[3,0].tick_params(axis='y', rotation=0)
    ax[4,0].tick_params(axis='y', rotation=0)

    ax[4,0].tick_params(axis='x', rotation=90)
    ax[4,1].tick_params(axis='x', rotation=90)
    ax[4,2].tick_params(axis='x', rotation=90)
    ax[4,3].tick_params(axis='x', rotation=90)
    ax[4,4].tick_params(axis='x', rotation=90)


    ax[4,3].xaxis.offsetText.set_visible(False)
    ax[4,3].xaxis.set_label_text(r"$\xi$" + " " +r"$\times10^{{{}}}$".format(xi_offset_str)+ r" / m$^{-1}$")

    ax[4,0].xaxis.offsetText.set_visible(False)
    ax[4,0].xaxis.set_label_text(r"$\alpha$" + " " +r"$\times10^{{{}}}$".format(alpha_offset_str) + r" / m$^{-1}$")

    ax[3,0].yaxis.offsetText.set_visible(False)
    ax[3,0].yaxis.set_label_text(r"$\xi$" + " " +r"$\times10^{{{}}}$".format(xi_offset_str)+ r" / m$^{-1}$")

    plt.savefig(path_to_input[:-11] + "corner_oscillatory.pdf")


    fig, ax = plotting.plot_corner(r2)
    fig.set_size_inches(8, 8)

    alpha_offset_str = "-" + str("{:e}".format(ax[2,0].dataLim._points[0][0])[-2:] )


    ax[2,0].set_xlabel(r"$\alpha$" + r" / m$^{-1}$")
    ax[2,1].set_xlabel(r"$A$")
    ax[2,2].set_xlabel(r"$n_{\infty}$")

    ax[1,0].set_ylabel(r"$A$")
    ax[2,0].set_ylabel(r"$n_{\infty}$")


    ax[1,0].tick_params(axis='y', rotation=0)
    ax[2,0].tick_params(axis='y', rotation=0)

    ax[2,0].tick_params(axis='x', rotation=90)
    ax[2,1].tick_params(axis='x', rotation=90)
    ax[2,2].tick_params(axis='x', rotation=90)



    ax[2,0].xaxis.offsetText.set_visible(False)
    ax[2,0].xaxis.set_label_text(r"$\alpha$" + " " +r"$\times10^{{{}}}$".format(alpha_offset_str) + r" / m$^{-1}$")

    plt.savefig(path_to_input[:-11] + "corner_dilute.pdf")



print(sys.argv[2])

with open(sys.argv[2]) as jsonfile:
    data = json.load(jsonfile)

bounds = ((1e6, 1e10), (0, 25), (2.3,3.3), (1e6, 1e10), (0, 2*np.pi))

sim = Simulation(data["x"], data["y"], data["yerr"], bounds, sys.argv[2])

worker(sim)
