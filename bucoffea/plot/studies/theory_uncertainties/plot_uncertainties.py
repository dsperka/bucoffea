#!/usr/bin/env python

import os
import sys
import re
import uproot
import argparse
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from pprint import pprint
from tqdm import tqdm

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to the ROOT file having the theory uncertainties.')
    parser.add_argument('-y', '--years', nargs='*', type=int, default=[2017, 2018], help='Years to plot the uncertainties for.')
    args = parser.parse_args()
    return args

def plot_theory_uncertainties(args):
    """
    Plot the theory uncertainties on the V+jets transfer factors, from the given ROOT file.
    """
    f = uproot.open(args.inpath)

    processes = {
        'ratio_z_qcd' : r'QCD $Z(\nu\nu)$ / $W(\ell\nu)$',
        'ratio_z_ewk' : r'EWK $Z(\nu\nu)$ / $W(\ell\nu)$',
        'ratio_gjets_qcd' : r'QCD $\gamma + jets$ / $Z(\nu\nu)$',
        'ratio_gjets_ewk' : r'EWK $\gamma + jets$ / $Z(\nu\nu)$',
    }

    # Colors per uncertainty
    color_mapping = {
        'muf' : '#31a354',
        'mur' : '#08306b',
        'pdf' : '#fc4e2a',
        'ewkcorr' : '#feb24c',
    }

    for year in tqdm(args.years, desc="Plotting theory uncertainties"):
        uncertainties = [
            key.decode('utf-8') for key in f.keys() if key.decode('utf-8').startswith('uncertainty') and str(year) in key.decode('utf-8')
        ]

        for process, proc_label in processes.items():
            has_process = lambda x: process in x
            # Get the uncertainties for this process
            uncs = list(filter(has_process, uncertainties))
            if not uncs:
                continue
                
            fig, ax = plt.subplots()
            for unc in uncs:
                histogram = f[unc]

                temp = re.findall('(muf|mur|pdf|ewkcorr).*(up|down)', unc)[0]
                label = '_'.join(temp)

                kwargs = {
                    'label' : label,
                    'linestyle' : '-' if temp[1] == 'up' else '--',
                    'color' : color_mapping[temp[0]],
                }

                hep.histplot(
                    histogram.values,
                    histogram.edges,
                    ax=ax,
                    **kwargs,
                )
            
            ax.legend(ncol=2, title='Uncertainty')

            ax.set_xlabel('CNN score')
            ax.set_ylabel('Uncertainty on Transfer Factor')

            ax.set_xlim(0,1)
            ax.set_ylim(0.7,1.3)

            ax.text(0,1,proc_label,
                fontsize=14,
                ha='left',
                va='bottom',
                transform=ax.transAxes,
            )

            outpath = pjoin(args.outdir, f'uncertainties_{process}_{year}.pdf')
            fig.savefig(outpath)
            plt.close(fig)

def main():
    args = parse_cli()

    # Save the plots under a sub-directory of the input directory
    outdir = pjoin(os.path.dirname(args.inpath), 'plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    args.outdir = outdir

    plot_theory_uncertainties(args)

if __name__ == '__main__':
    main()