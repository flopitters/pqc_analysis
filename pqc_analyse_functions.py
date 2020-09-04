## ------------------------------------
## pqc_analyse_functions.py
##
## Set of analysis function for PQC measurements.
##
## Status:
##
## ------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline



## Function Definitions
## ------------------------------------

def analyse_iv(v, i, debug=0):
    return -1
    
    
def analyse_cv(v, c, debug=0):
    return -1
    

def analyse_mos(v, c, cut_param=0.0015, debug=0):

    ## init
    vfb_1 = -1
    vfb_2 = -1
    t_ox = -1
    n_ox = -1
    
    # take average of last 5 samples for accumulation and inversion capacitance
    c_acc = np.mean(c[-5:])
    c_inv = np.mean(c[:5])
    
    # get spline fit, requires strictlty increasing array
    y_norm = c / np.max(c)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)

    # get regions for indexing
    idx_inv = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param and v[i] < v[np.argmin(spl_dev)] ]
    idx_dep = [ i for i in range(len(spl_dev)) if (v[i] > v[np.argmax(spl_dev)] - 0.25  and v[i] < v[np.argmax(spl_dev)] + 0.25 ]
    idx_acc = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param and v[i] > v[np.argmax(spl_dev)]) ]
    v_acc = v[ idx_acc[0]:idx_acc[-1] ]
    v_dep = v[ idx_dep[0]:idx_dep[-1] ]
    v_inv = v[ idx_inv[0]:idx_inv[-1] ]
    c_acc = c[ idx_acc[0]:idx_acc[-1] ]
    c_dep = c[ idx_dep[0]:idx_dep[-1] ]
    c_inv = c[ idx_inv[0]:idx_inv[-1] ]
    
    # line fits to each region
    try:
        a_acc, b_acc = np.polyfit(v_acc, c_acc, 1)
        a_dep, b_dep = np.polyfit(v_dep, c_dep, 1)
        a_inv, b_inv = np.polyfit(v_inv, c_inv, 1)

        # flatband voltage via max. 1st derivative
        vfb_1 = v[np.argmax(spl_dev)]
        
        # flatband voltage via intersection
        vfb_2 = (b_acc - b_dep) / (a_dep - a_acc)
        
        # note 1: Phi_MS of -0.69V is used as standard value, this correpsonds to a p-type bulk doping of 5e12 cm^-3
        # note 2: We apply the bias voltage to the backplane while keeping the gate to ground, V_fb is therefore positive
        n_ox = np.mean(c_acc) / (1.6e-19 * (0.1290**2)) * (0.69 + vfb_2) 
        t_ox = 3.9 * 8.85e-12 * (0.001290**2) / np.mean(c_acc) * 1e6
    
    except (ValueError, TypeError):
        print("The array seems empty. Try changing the cut_param parameter.")
    
    return vfb_1, vfb_2, c_acc, c_inv, t_ox, n_ox, a_acc, b_acc, v_acc, a_dep, b_dep, v_dep, a_inv, b_inv, v_inv,  spl_dev



def analyse_gcd(v, i, debug=0):
    return -1



def analyse_fet(v, i, debug=0):

    # init
    v_th = -1
    
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
    
    v_th = - b / a
    
    return v_th, a, b, spl_dev



def analyse_van_der_pauw(i, v, cut_param=0.01, debug=0):

    # init
    r_sheet = -1
    a = -1
    b = -1
    
    # get spline fit, requires strictlty increasing array
    y_norm = v / np.max(v)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)

    # note: only use data points if local slope is above cut_param
    # 
    idx_fit = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) > cut_param) ]
    x_fit = i[ idx_fit[0]:idx_fit[-1] ]
    y_fit = v[ idx_fit[0]:idx_fit[-1] ]
    
    a, b = np.polyfit(x_fit, y_fit, 1)
    r_sheet = np.pi / np.log(2) * a
    
    try:
        a, b = np.polyfit(x_fit, y_fit, 1)
        r_sheet = np.pi / np.log(2) * a
    except (ValueError, TypeError):
        print("The array seems empty. Try changing the cut_param parameter.")

    return r_sheet, a, b, x_fit, spl_dev



def analyse_linewidth(i, v, cut_param=0.01, debug=0):
    
    # init
    r_sheet = -1
    a = -1
    b = -1
    
    # get spline fit, requires strictlty increasing array
    y_norm = v / np.max(v)
    x_norm = np.arange(len(y_norm))
    spl = CubicSpline(x_norm, y_norm)
    spl_dev = spl(x_norm, 1)

    # only use data points if local slope is above cut_param
    idx_fit = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) > cut_param) ]
    x_fit = i[ idx_fit[0]:idx_fit[-1] ]
    y_fit = v[ idx_fit[0]:idx_fit[-1] ]
    
    try:
        a, b = np.polyfit(x_fit, y_fit, 1)
        r_sheet = np.pi / np.log(2) * a
    except (ValueError, TypeError):
        print("The array seems empty. Try changing the cut_param parameter.")
  
    return r_sheet, a, b, x_fit, spl_dev




def analyse_meander(v, i, debug=0):

    # resistance
    r = v/i
    
    return r



def analyse_breakdown(v, i, debug=0):
    
    # breakdown voltage
    v_bd = v[-1]
    
    return v_bd
