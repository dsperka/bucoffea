#!/usr/bin/env python

import os
import sys
import re
import uproot
import argparse
import numpy as np

from matplotlib import pyplot as plt
from coffea import hist
from tqdm import tqdm
from klepto.archives import dir_archive

from bucoffea.plot.util import (
    merge_extensions, 
    merge_datasets, 
    scale_xs_lumi, 
    rebin_histogram,
    )

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to the merged accumulator input.')
    parser.add_argument('-v', '--variable', help='The variable to plot the uncertainties for.', default='cnn_score', choices=['cnn_score','mjj'])
    args = parser.parse_args()
    return args


def plot_uncertainty(acc,
    outputrootfile,
    uncertainty: str,
    dataset: str,
    dataset_tag: str,
    nuisance: str,
    variable: str,
    region: str,
    outdir: str,
    ) -> None:
    """
    Plot a specified uncertainty source for a given dataset, as a function
    of the given variable.

    "uncertainty" parameter is a regular expression specifying which 
    uncertainties to plot.
    """
    distribution = f'{variable}_unc'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = rebin_histogram(h, variable)

    h = h.integrate("region", region).integrate("dataset", re.compile(dataset))

    # Plot the variations for the given uncertainty source
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (1, 1)}, sharex=True)

    uncertainties_to_plot = [
        x.name for x in h.identifiers("uncertainty") if re.match(f"{uncertainty}.*", x.name)
    ]

    for unc in uncertainties_to_plot:
        hist.plot1d(h.integrate("uncertainty", unc),
            ax=ax,
            clear=False,
        )

    ax.get_legend().remove()

    ax.text(0,1,dataset_tag,
        fontsize=14,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    # Plot ratios
    ratio_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
    }
    h_nom = h.integrate("uncertainty", f"{uncertainty}Nom")
    for unc in uncertainties_to_plot:
        if "Nom" in unc:
            continue

        hist.plotratio(
            h.integrate("uncertainty", unc),
            h_nom,
            ax=rax,
            unc="num",
            label=unc,
            clear=False,
            error_opts=ratio_err_opts,
        )

        # Save the ratios into an output ROOT file
        ratio = h.integrate("uncertainty", unc).values()[()] / h_nom.values()[()]

        unc_name = f'{nuisance}_up' if 'Up' in unc else f'{nuisance}_down'

        outputrootfile[f'{dataset_tag}_{unc_name}'] = (ratio, h_nom.axis("score").edges())

    rax.legend(title="Uncertainty")
    rax.set_ylabel("Ratio")
    rax.set_ylim(0.7,1.3)
    rax.grid(True)

    filename = f'{dataset_tag}_{nuisance}.pdf'
    outpath = pjoin(outdir, filename)
    fig.savefig(outpath)
    plt.close(fig)


def main():
    args = parse_cli()

    acc = dir_archive(args.inpath)
    acc.load('sumw')

    outtag = os.path.basename(args.inpath.rstrip('/'))

    # Output directory to save plots
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Uncertainty sources we will be plotting
    uncertainties = {
        "prefire" : {
            "dataset"     : "VBF_HToInvisible.*M125.*2017",
            "dataset_tag" : "VBF_HToInvisible_2017",
            "nuisance"    : "CMS_L1prefire_2017",
            "region"      : "sr_vbf_no_veto_all",
            "outfile"     : f"vbf_prefire_uncs_{args.variable}.root",
        },
        "puSF" : {
            "dataset"     : "VBF_HToInvisible.*M125.*2017",
            "dataset_tag" : "VBF_HToInvisible_2017",
            "nuisance"    : "CMS_pileup",
            "region"      : "sr_vbf_no_veto_all",
            "outfile"     : f"vbf_pileup_uncs_{args.variable}.root",
        },
    }

    for uncertainty, item in tqdm(uncertainties.items(), desc="Plotting uncertainties"):
        # The ROOT file to save ratios of up and down variations
        outputrootfile = uproot.recreate(pjoin(outdir, item["outfile"]))

        plot_uncertainty(acc,
            uncertainty=uncertainty,
            variable=args.variable,
            dataset=item["dataset"],
            dataset_tag=item["dataset_tag"],
            nuisance=item["nuisance"],
            region=item["region"],
            outdir=outdir,
            outputrootfile=outputrootfile,
        )

if __name__ == '__main__':
    main()