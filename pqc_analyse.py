#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from optparse import OptionParser
from pqc_analyse_functions import *




## Function Definitions
## ------------------------------------

def analyse_iv_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing diode IV for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=27)
    t, v, i_tot, i =  dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3]
    lbl = '_'.join(fn.split('_')[4:9])
    
    ## analyse
    analyse_iv(v, i)
    
    ## plot
    plt.plot(v, i, ls=' ', marker='s', ms=3, label=lbl)
    plt.annotate('V$_\mathrm{fb}$: %.1f V\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (-1, np.mean(dat[:, 5]), np.mean(dat[:, 6])) + r'$\%$', (0.70,0.2), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))     
    plt.legend(loc='upper left')
    plt.xlabel('voltage [V]')
    plt.ylabel('current [A]')
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_diode_iv.pdf')
    plt.clf()
    
    return v,i,lbl
    


def analyse_cv_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing diode CV for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=14, encoding='utf-8')
    t, v, i_tot, c, c2, r =  dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5]
    lbl = '_'.join(fn.split('_')[4:9])
    
    ## analyse
    analyse_cv(v, c)
    
    ## plot
    plt.plot(v, c, ls=' ', marker='s', ms=3, label=lbl)
    plt.annotate('V$_\mathrm{fb}$: %.1f V\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (-1, np.mean(dat[:, 7]), np.mean(dat[:, 8])) + r'$\%$', (0.70,0.2), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))
    plt.legend(loc='upper left')
    plt.xlabel('voltage [V]')
    plt.ylabel('capactiance [F]')
    plt.ylim([0, +1e-11])
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_diode_cv.pdf')
    plt.clf()
    
    plt.plot(v, 1/c**2, ls=' ', marker='s', ms=3, label=lbl)
    plt.annotate('V$_\mathrm{fb}$: %.1f V\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (-1, np.mean(dat[:, 7]), np.mean(dat[:, 8])) + r'$\%$', (0.70,0.2), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))
    plt.legend(loc='upper right')
    plt.xlabel('voltage [V]')
    plt.ylabel('1/C^2 [1/F^2]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_diode_1c2v.pdf')
    plt.clf()
    
    return v,c,lbl
    


def analyse_mos_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing MOS CV for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=14, encoding='UTF-8')
    t, v, i_tot, c, c2, r =  dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5]
    lbl = '_'.join(fn.split('_')[4:9])
    
    ## analyse
    vfb_1, vfb_2, c_acc, c_inv, t_ox, n_ox, a_acc, b_acc, v_acc, a_dep, b_dep, v_dep, a_inv, b_inv, v_inv,  spl_dev = analyse_mos(v, c)

    ## plot
    plt.plot(v, c, ls=' ', marker='s', ms=3, label=lbl)
    # plt.plot(v, spl_dev, 'g--', label='1st derivative')
    plt.plot(v_acc, a_acc*v_acc+b_acc, '-r')
    plt.plot(v_dep, a_dep*v_dep+b_dep, '-r')
    plt.plot(v_inv, a_inv*v_inv+b_inv, '-r')
    plt.plot(v, a_dep*v+b_dep, '--r')
    plt.plot(v, a_acc*v+b_acc, '--r')
    
    x_loc = 0.66
    y_loc = 0.145
    if vfb_1 > 4:
        x_loc = 0.11
    plt.annotate('V$_\mathrm{fb}$: %.1f V (via intersection)\nV$_\mathrm{fb}$: %.1f V (via derivative)\n\nt$_\mathrm{ox}$: %.2f um\nn$_\mathrm{ox}$: %.2e cm$^{-2}$\n\nC$_\mathrm{acc}$: %.1e F\nC$_\mathrm{inv}$: %.1e F\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (vfb_2, vfb_1, t_ox, n_ox, np.mean(c_acc), np.mean(c_inv), np.mean(dat[:, 7]), np.mean(dat[:, 8])) + r'$\%$', (x_loc, y_loc), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))           
    plt.legend(loc='upper left')
    plt.xlabel('voltage [V]')
    plt.ylabel('capacitance [F]')
    plt.ylim([0, 1e-10])
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_mos.pdf')
    plt.clf()
    
    return v, c, vfb_2, t_ox, n_ox, lbl



def analyse_gcd_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing GCD for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=33, encoding='UTF-8')
    t, v, i_em, i_vsrc, i_hvsrc, v_bias =  dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5]
    lbl = '_'.join(fn.split('_')[4:9])
    
    ## analyse
    analyse_gcd(v, i_em)

    ## plot
    plt.plot(v, i_em, ls=' ', marker='s', ms=3, label="i_em")
    plt.annotate('V$_\mathrm{fb}$: %.1f V\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (-1, np.mean(dat[:, 7]), np.mean(dat[:, 8])) + r'$\%$', (0.70,0.2), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))
    plt.legend(loc='upper left')
    plt.xlabel('voltage [V]')
    plt.ylabel('current [A]')
    # plt.ylim([-9e-7, +1e-7])
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_gcd.pdf')
    plt.clf()
    
    return v, i_em, lbl


def analyse_fet_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing FET for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=33, encoding='UTF-8')
    t, v, i_em, i_vsrc, i_hvsrc, v_bias =  dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5]
    lbl = '_'.join(fn.split('_')[4:9])
  
    ## analyse
    v_th, a, b, spl_dev = analyse_fet(v, i_em)
              
    ## plot
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(v, i_em, ls=' ', marker='s', ms=3, label="transfer characteristics")
    ax1.set_xlabel(r'V$_\mathrm{GS}$ [V]')
    ax1.set_ylabel(r'I$_\mathrm{D}$ [A]')
    ax1.set_ylim([-1e-6, +5e-6])

    ax2 = ax1.twinx()
    lns2 = ax2.plot(v, spl_dev, ls=' ', marker='s', ms=3, color='tab:orange', label="transconductance")
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel(r'g$_\mathrm{m}$ [S]', color='tab:orange')
    
    lns3 = ax1.plot(v, a*v+b, '--r', label="tangent")

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper left')
    
    plt.annotate('V$_\mathrm{th}$: %.2f + 0.05 V\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (v_th, np.mean(dat[:, 7]), np.mean(dat[:, 8])) + r'$\%$', (0.13, 0.6), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_fet.pdf')
    plt.clf()
    
    return v, i_em, v_th, lbl



def analyse_van_der_pauw_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing Van der Pauw for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=13)
    t, i, v =  dat[:, 0], dat[:, 1], dat[:, 2]
    lbl = '_'.join(fn.split('_')[4:12])
    
    ## analyse
    r_sheet, a, b, x_fit, spl_dev = analyse_van_der_pauw(i, v)
            
    ## plot
    plt.plot(x_fit, a*x_fit+b, '-r')
    plt.plot(i, v, ls=' ', marker='s', ms=3, label=lbl)
    plt.annotate('R$_\mathrm{sheet}$: %.2e $\Omega$/sq\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (r_sheet, np.mean(dat[:, 4]), np.mean(dat[:, 5])) + r'$\%$', (0.20, 0.6), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))
    plt.legend(loc='upper left')
    plt.xlabel('current [A]')
    plt.ylabel('voltage [V]')
    # plt.ylim([-9e-7, +1e-7])
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_van_der_pauw.pdf')
    plt.clf()
    
    return i, v, r_sheet, lbl



def analyse_linewidth_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing linewidth for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=13)
    t, i, v =  dat[:, 0], dat[:, 1], dat[:, 2]
    lbl = '_'.join(fn.split('_')[4:9])
    
    ## analyse
    r_sheet, a, b, x_fit, spl_dev = analyse_linewidth(i, v, cut_param=0.01, debug=0)

    ## plot
    plt.plot(i, v, ls=' ', ms=3, marker='s', label=lbl)
    plt.plot(x_fit, a*x_fit+b, '-r')
    plt.annotate('R$_\mathrm{sheet}$: %.2e $\Omega$/sq\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (r_sheet, np.mean(dat[:, 4]), np.mean(dat[:, 5])) + r'$\%$', (0.20, 0.6), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))
    plt.legend(loc='upper left')
    plt.xlabel('current [A]')
    plt.ylabel('voltage [V]')
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_linewidth.pdf')
    plt.clf()
    
    return v, i, r_sheet, lbl



def analyse_meander_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing Meander for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=27)
    t, v, i_tot, i =  dat[0], dat[1], dat[2], dat[3]
    
    ## analyse
    r = analyse_meander(v, i, debug=0)
    
    return v, i, r



def analyse_breakdown_data(fn, debug=0):
    path = '/'.join(fn.split('/')[:-1])
    fn = fn.split('/')[-1]
    if debug:
        print(" -- Analysing Si02 breakdown for file:\n\t%s" % fn)
    
    ## read data
    dat = np.genfromtxt(path + '/' + fn, skip_header=27)
    t, v, i_tot, i =  dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3]
    lbl = '_'.join(fn.split('_')[4:9])

    ## analyse
    v_bd = analyse_breakdown(v, i)
    
    ## plot
    plt.plot(v, i, ls=' ', marker='s', ms=3, label=lbl)
    plt.annotate('V$_\mathrm{bd}$: %d V\n\nT$_\mathrm{avg}$: %.1f $^\circ C$\nH$_\mathrm{avg}$: %.1f ' % \
        (v_bd, np.mean(dat[:, 5]), np.mean(dat[:, 6])) + r'$\%$', (0.20,0.5), xycoords='figure fraction', color='tab:blue', \
        bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round,pad=0.5'))
    plt.legend(loc='upper left')
    plt.xlabel('voltage [V]')
    plt.ylabel('current [A]')
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(path + '/' + fn[:-4] + '_breakdown.pdf')
    plt.clf()
    
    return v, i, v_bd, lbl


    

def analyse_folder(path, debug=0):
    
    # fetch files
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[-4:] == '.txt']
    
    # loop over and look for fitting analysis
    for f in file_list:
        vals = f.split('_')
        # if 'iv' in [v.lower() for v in f.split('_')]:
        #     v, i, lbl = analyse_iv_data(path + '/' + f)
        #
        # if 'cv' in [v.lower() for v in f.split('_')]:
        #     v, c, lbl  = analyse_cv_data(path + '/' + f)
        #
        # if 'mos' in [v.lower() for v in f.split('_')]:
        #     v, c, vfb_2, t_ox, n_ox, lbl = analyse_mos_data(path + '/' + f)
        #     print(lbl, vfb_2, t_ox, n_ox)

        # if 'gcd' in [v.lower() for v in f.split('_')]:
        #     v, i, lbl = analyse_gcd_data(path + '/' + f)
        #
        # if 'fet' in [v.lower() for v in f.split('_')]:
        #     v, i, v_th, lbl = analyse_fet_data(path + '/' + f)
        #     print(lbl, v_th)

        if 'van-der-pauw' in [v.lower() for v in f.split('_')]:
            i, v, r_sheet, lbl = analyse_van_der_pauw_data(path + '/' + f)
            print(lbl, r_sheet)

        if 'linewidth' in [v.lower() for v in f.split('_')]:
            analyse_linewidth_data(path + '/' + f)

        # if 'meander' in [v.lower() for v in f.split('_')]:
    #         r = analyse_meander_data(path + '/' + f)
    #
    #     if 'breakdown' in [v.lower() for v in f.split('_')]:
    #         v_bd = analyse_breakdown_data(path + '/' + f)



## Main Executable
## ------------------------------------

def main():
    usage = "usage: ./pqc_analyse.py -i [path_to_file] -s [serial_number] -l [location] -o [operator] [options]"

    parser = OptionParser(usage=usage, version="prog 0.1")
    parser.add_option("-f", "--folder", action="store", dest="input_path", type="string", \
        help="path to input folder")

    parser.add_option("--ex", "--examples", action="store_true", dest="fExamples",  help="print examples") 
    
    (options, args) = parser.parse_args()
    
    if options.fExamples:
        print("\nSome example commands for running this script\n")
        print("-- ./pqc_analyse.py -i /data/hgc/pqc/HPK_8in_LD_2019_TS_1001_UL_300um_5V")

    elif options.input_path:
        path = options.input_path
        analyse_folder(path)
        
    else:
        paths = [
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_1001_UL_300um_5V',
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_1002_UL_300um_2V',
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_1003_UL_300um_2V',
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_1101_UL_300um_5V',
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_1102_UL_300um_2V',
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_1103_UL_300um_2V',
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_2001_UL_200um_5V',
            # '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_2002_UL_200um_2V',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_2003_UL_200um_2V',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_2101_UL_200um_5V',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_2102_UL_200um_2V',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_8in_LD_2019_TS_2103_UL_200um_2V',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_VPX33234_016_PSS_HM-EE',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_VPX33234_017_PSS_HM-EE',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_VPX33234_018_PSS_HM-EE',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_VPX33234_019_PSS_HM-EE',
            '/Users/Home/Documents/Works/pqc/pqc_analysis/data/HPK_VPX33234_020_PSS_HM-EE'
        ]

        for path in paths:
            analyse_folder(path)
 


if __name__ == "__main__":
    main()

