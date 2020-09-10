## ------------------------------------
## pqc_analyse_functions.py
##
## Set of analysis function for PQC measurements.
##
## Status:
##
## ------------------------------------

import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from collections import namedtuple



## Constants
## ------------------------------------

kBoltzmann = 8.61733 * 10**(-5) # [ev/K]
kZero = 273.15 # K
kVacuumPermitivity = 8.854 * 10**(-12) # [F/m]
kUnitCharge = 1.602 * 10**(-19) # [C]




## Helper functions
## ------------------------------------

def params(names):
    def params(f):
        def params(*args, **kwargs):
           return namedtuple(f.__name__, names)(*f(*args, **kwargs))
        return params
    return params



@params('a, b, x_fit, spl_dev')
def line_regr_with_cuts(x, y, cut_param, debug=0):
    """ 
    Linear Regression with Cuts: 
    - Normalise data set
    - Get 1st derivate of 2nd order spline fit
    - Only use data points with local slope exceeding cut_param

    Parameters:
    x ... x
    y ... y
    cut_param ... used to cut on 1st derivative of x axis
    
    Returns:
    i_max ... max. current
    i_800 ... current @ 800V
    i_600 ... current @ 600V
    """
    
    # init 
    a = b = x_fit = spl_dev = -1
    
    # get spline fit, requires strictlty increasing array
    y_norm = y / np.max(y)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)

    # only use data points if local slope is above cut_param
    idx_fit = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) > cut_param) ]
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            x_fit = x[ idx_fit[0]:idx_fit[-1]+1 ]
            y_fit = y[ idx_fit[0]:idx_fit[-1]+1 ]
            a, b = np.polyfit(x_fit, y_fit, 1)
        except np.RankWarning:
            print("The array has too few data points. Try changing the cut_param parameter.")
        except (ValueError, TypeError, IndexError):
            print("The array seems empty. Try changing the cut_param parameter.")
        
    return a, b, x_fit, spl_dev




## Main Analysis Functions 
## ------------------------------------

@params('i_max, i_800, i_600')
def analyse_iv(v, i, debug=0):
    """ 
    Diode IV: Extract current in standard situation. 

    Parameters:
    v ... voltage
    i ... current
    
    Returns:
    i_max ... max. current
    i_800 ... current @ 800V
    i_600 ... current @ 600V
    """
    
    i_max = i_800 = i_600 = -1
    
    i_max = i[ np.argmax(v) ]
    i_800 = [ i[k] for k in range(len(v)) if v[k] == 800 ]
    i_600 = [ i[k] for k in range(len(v)) if v[k] == 600 ]
    
    return i_max, i_800, i_600



@params('v_dep1, v_dep2, rho, conc, a_rise, b_rise, v_rise, a_const, b_const, v_const, spl_dev')
def analyse_cv(v, c, area=1.56e-6, carrier='electrons', cut_param=0.1, debug=0):
    """ 
    Diode CV: Extract depletion voltage and resistivity.

    Parameters:
    v ... voltage
    c ... capacitance
    area ... implant size in [m^2]
    carrier ... majority charge carriers ['holes', 'electrons']
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    v_dep1 ... full depletion voltage via inflection
    v_dep2 ... full depletion voltage via intersection
    rho ... resistivity
    conc ... bulk doping concentration
    """
    
    # init
    v_dep = rho = n_dop = -1
    
    # invert and square
    c = 1./c**2
    
    # get spline fit, requires strictlty increasing array
    y_norm = c / np.max(c)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)

    # get regions for indexing
    idx_rise = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) > cut_param) ]
    idx_const = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < 0.01) ]

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        
        try:
            v_rise = v[ idx_rise[0]:idx_rise[-1]+1 ]
            v_const = v[ idx_const[0]:idx_const[-1]+1 ]
            c_rise = c[ idx_rise[0]:idx_rise[-1]+1 ]
            c_const = c[ idx_const[0]:idx_const[-1]+1 ]
            
            # line fits to each region
            a_rise, b_rise = np.polyfit(v_rise, c_rise, 1)
            a_const, b_const = np.polyfit(v_const, c_const, 1)
        
            if carrier == 'holes':
                mu = 450 * 1e-4
            elif carrier == 'electrons':
                mu = 1350 * 1e-4
            else:
                raise ValueError('Not a valid type of majority carrier.')

            # full depletion voltage via max. 1st derivative
            v_dep1 = v[np.argmax(spl_dev)]
        
            # full depletion via intersection
            v_dep2 = (b_const - b_rise) / (a_rise - a_const)
        
            # rest
            conc = 2. / (1.6e-19 * 11.9 * 8.854e-12 * a_rise * area**2)
            rho = 1. / (mu * q * conc)
            
        except np.RankWarning:
            print("The array has too few data points. Try changing the cut_param parameter.")
            
        except (ValueError, TypeError, IndexError):
            print("The array seems empty. Try changing the cut_param parameter.")
    
    return v_dep1, v_dep2, rho, conc, a_rise, b_rise, v_rise, a_const, b_const, v_const, spl_dev



@params('v_fb1, v_fb2, c_acc, c_inv, t_ox, n_ox, a_acc, b_acc, v_acc, a_dep, b_dep, v_dep, a_inv, b_inv, v_inv,  spl_dev')
def analyse_mos(v, c, cut_param=0.0015, debug=0):
    """ 
    Metal oxide Capacitor: Extract flatband voltage, oxide thickness and charge density. 

    Parameters:
    v ... voltage
    c ... capacitance
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    v_fb1 ... flatband voltage via inflection
    v_fb2 ... flatband voltage via intersection
    t_ox ... oxide thickness
    n_ox ... oxide charge density
    """
    
    ## init
    v_fb1 = v_fb2 =  t_ox = n_ox = -1
    
    # take average of last 5 samples for accumulation and inversion capacitance
    c_acc = np.mean(c[-5:])
    c_inv = np.mean(c[:5])
    
    # get spline fit, requires strictlty increasing array
    y_norm = c / np.max(c)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)

    # get regions for indexing
    idx_acc = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param and v[i] > v[np.argmax(spl_dev)]) ]
    idx_dep = [ i for i in range(len(spl_dev)) if (v[i] > v[np.argmax(spl_dev)] - 0.25  and v[i] < v[np.argmax(spl_dev)] + 0.25) ]
    idx_inv = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param and v[i] < v[np.argmin(spl_dev)]) ]
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
            
        try:
            v_acc = v[ idx_acc[0]:idx_acc[-1]+1 ]
            v_dep = v[ idx_dep[0]:idx_dep[-1]+1 ]
            v_inv = v[ idx_inv[0]:idx_inv[-1]+1 ]
            c_acc = c[ idx_acc[0]:idx_acc[-1]+1 ]
            c_dep = c[ idx_dep[0]:idx_dep[-1]+1 ]
            c_inv = c[ idx_inv[0]:idx_inv[-1]+1 ]
            
            # line fits to each region
            a_acc, b_acc = np.polyfit(v_acc, c_acc, 1)
            a_dep, b_dep = np.polyfit(v_dep, c_dep, 1)
            a_inv, b_inv = np.polyfit(v_inv, c_inv, 1)
        
            # flatband voltage via inflection
            v_fb1 = v[np.argmax(spl_dev)]
            
            # flatband voltage via intersection
            v_fb2 = (b_acc - b_dep) / (a_dep - a_acc)
            
            # note 1: Phi_MS of -0.69V is used as standard value, this correpsonds to a p-type bulk doping of 5e12 cm^-3
            # note 2: We apply the bias voltage to the backplane while keeping the gate to ground, V_fb is therefore positive
            n_ox = np.mean(c_acc) / (1.602e-19 * (0.1290**2)) * (0.69 + v_fb2)
            t_ox = 3.9 * 8.85e-12 * (0.001290**2) / np.mean(c_acc) * 1e6
        
        except np.RankWarning:
            print("The array has too few data points. Try changing the cut_param parameter.")
            
        except (ValueError, TypeError, IndexError):
            print("The array seems empty. Try changing the cut_param parameter.")
    
    return v_fb1, v_fb2, c_acc, c_inv, t_ox, n_ox, a_acc, b_acc, v_acc, a_dep, b_dep, v_dep, a_inv, b_inv, v_inv,  spl_dev



@params('i_surf, i_bulk, i_acc, i_dep, i_inv, v_acc, v_dep, v_inv, spl_dev')
def analyse_gcd(v, i, cut_param=0.01, debug=0):
    """ Gate Controlled Diode: Generation currents.

    Parameters:
    v ... voltage
    i ... current
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    i_surf ... surface generation current
    i_bulk ... bulk generation current
    """
    
    # get spline fit, requires strictlty increasing array
    y_norm = i / np.max(i)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)
    
    # get regions for indexing
    idx_acc = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param and v[i] > v[np.argmax(spl_dev)]) ]
    idx_dep = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param and v[i] > v[np.argmax(spl_dev)] - 2.5  and v[i] < v[np.argmax(spl_dev)] + 2.5) ]
    idx_inv = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param and v[i] < v[np.argmin(spl_dev)]) ]
    v_acc = v[ idx_acc[0]:idx_acc[-1]+1 ]
    v_dep = v[ idx_dep[0]:idx_dep[-1]+1 ]
    v_inv = v[ idx_inv[0]:idx_inv[-1]+1 ]
    i_acc = i [ idx_acc[0]:idx_acc[-1]+1 ]
    i_dep = i [ idx_dep[0]:idx_dep[-1]+1 ]
    i_inv = i [ idx_inv[0]:idx_inv[-1]+1 ]
    
    # surface and bulk generation current
    i_surf = np.mean(i_dep) - np.mean(i_inv)
    i_bulk = np.mean(i_inv) - np.mean(i_acc)
    
    # phi_f = (8.617e-5 * 295 / 1.) * np.log(9.65e9 / 5e12)
    # t_life = 1.6e-19 * n_i * A * (W_g - W_g0) / I_b
    
    return i_surf, i_bulk, i_acc, i_dep, i_inv, v_acc, v_dep, v_inv, spl_dev
    
    ## umdrehen



@params('v_th1, v_th2, a, b, spl_dev')
def analyse_fet(v, i, debug=0):
    """
    Field Effect Transistor: Threshold voltage.

    Parameters:
    v ... voltage
    i ... current
    
    Returns:
    v_th1 ... threshold voltage via max. 2nd derivative
    v_th2 ... threshold voltage via intersection
    """
    
    # init
    v_th1 = v_th2 = -1
    
    # get spline fit, requires strictlty increasing array
    y_norm = i / np.max(i)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)
    
    # get tangent at max. of 1st derivative
    i_0 = i[np.argmax(spl_dev)]
    v_0 = v[np.argmax(spl_dev)]
    a = np.max(spl_dev) / (v[np.argmax(spl_dev)] - v[np.argmax(spl_dev)-1])
    b = i_0 - a*v_0
    
    # threshold voltage via max. 2nd derivative
    v_th1 = v[np.argmax(spl(x_norm, 2))]
    
    # threshold voltage via intersection
    v_th2 = - b / a
    
    return v_th1, v_th2, a, b, spl_dev



@params('r_sheet, a, b, x_fit, spl_dev')
def analyse_van_der_pauw(i, v, cut_param=0.05, debug=0):
    """
    Van der Pauw: Extract sheet resistance.

    Parameters:
    i ... current
    v ... voltage
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    r_sheet ... resistance per square
    """
    
    a, b, x_fit, spl_dev = line_regr_with_cuts(i, v, cut_param, debug)
    r_sheet = np.pi / np.log(2) * a
  
    return r_sheet, a, b, x_fit, spl_dev



@params('t_line, a, b, x_fit, spl_dev')
def analyse_linewidth(i, v, r_sheet=-1, cut_param=0.05, debug=0):
    """
    Linewidth: Extract linewidth.

    Parameters:
    i ... current
    v ... voltage
    r_sheet ... sheet resistance
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    t_line ... linewidth in [um]
    """
    
    a, b, x_fit, spl_dev = line_regr_with_cuts(i, v, cut_param, debug)
    t_line = r_sheet * 128.5 * 1./a
    
    if r_sheet == -1:
        t_line = -1
  
    return t_line, a, b, x_fit, spl_dev



@params('r_contact, a, b, x_fit, spl_dev')
def analyse_cbkr(i, v, r_sheet=-1, cut_param=0.05, debug=0):
    """
    Cross Bridge Kelvin Resistance Structure: Extract contact resistance.

    Parameters:
    i ... current
    v ... voltage
    r_sheet ... sheet resistance
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    r_contact ... contact resistance
    """
    
    a, b, x_fit, spl_dev = line_regr_with_cuts(i, v, cut_param, debug)
    
    # note: The contact isn't symmetric. It's 12.5 by 13.5 um. Solution for now is to use 13 um. 
    d = 13 # contact size
    w = 33 # diffusion width
    r_c = a - (4*r_sheet*d**2) / (3*w**2) * (1 + d/(2*w - 2*d))
    
    if r_sheet == -1:
        r_contact = -1
  
    return r_contact, a, b, x_fit, spl_dev



@params('r_contact, a, b, x_fit, spl_dev')
def analyse_contact(i, v, debug=0):
    """
    Contact Chain: Extract metal-implant contact resistance.

    Parameters:
    i ... current
    v ... voltage
    r_sheet ... sheet resistance
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    r_contact ... contact resistance
    """
    
    a, b, x_fit, spl_dev = line_regr_with_cuts(v, i, cut_param, debug)
    r_contact = a
  
    return r_contact, a, b, x_fit, spl_dev



@params('r_sheet, a, b, x_fit, spl_dev')
def analyse_cross(i, v, debug=0):
    """
    Cross: Extract sheet resistance.

    Parameters:
    i ... current
    v ... voltage
    cut_param ... used to cut on 1st derivative to id voltage regions
    
    Returns:
    r_sheet ... resistance per square
    """
    
    a, b, x_fit, spl_dev = line_regr_with_cuts(i, v, cut_param, debug)
    r_sheet = np.pi / np.log(2) * a
  
    return r_sheet, a, b, x_fit, spl_dev



@params('rho_sq')
def analyse_meander(i, v, w=10, nsq=12835, debug=0):
    """
    Meander: Calculates specific resistance per square.

    Parameters:
    i ... current
    v ... voltage
    w ... strip width, use [5, 10] for [polysilicon, metal]
    nsq ... number of squares, use [476, 12853] for [polysilicon, metal]
    
    Returns:
    rho_sq ... specific resistance per square
    """
    
    r = v/i
    rho_sq = r * w / nsq 
    
    return rho_sq



@params('v_bd')
def analyse_breakdown(v, i, debug=0):
    """
    Breakdown: Get oxide breakdown.

    Parameters:
    v ... voltage
    i ... current
    
    Returns:
    v_bd  ... breakdown voltage
    """

    v_bd = v[-1]
    
    return v_bd



@params('c_mean, c_median')
def analyse_capacitor(v, c, debug=0):
    """
    Test capacitors: Get mean capacitance.

    Parameters:
    v ... voltage
    c ... capacitance
    
    Returns:
    c_mean ... mean capacitance  
    c_median ... median capacitance
    """
    
    c_mean = np.mean(c)
    c_median = np.median(c)
    
    return c_mean, c_median



