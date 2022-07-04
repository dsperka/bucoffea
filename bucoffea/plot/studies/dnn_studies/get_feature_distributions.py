#!/usr/bin/env python

import os
import sys
import re
import csv
import warnings
import argparse
import numpy as np

from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from tqdm import tqdm

from bucoffea.plot.util import (
    merge_extensions, 
    merge_datasets, 
    scale_xs_lumi, 
    rebin_histogram, 
    get_dataset_tag
)

pjoin = os.path.join

# Ignore RuntimeWarnings from coffea
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='coffea')

REGION_TAGS = {
    'sr_vbf' : 'Signal Region',
    'cr_vbf_qcd' : 'HF-noise CR',
    'cr_1m_vbf' : r'$W(\mu\nu)$ CR',
    'cr_1e_vbf' : r'$W(e\nu)$ CR',
    'cr_2m_vbf' : r'$Z(\mu\mu)$ CR',
    'cr_2e_vbf' : r'$Z(ee)$ CR',
    'cr_g_vbf' : r'$\gamma$+jets CR',
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to coffea accumulator.')
    parser.add_argument('-r', '--region', default='.*', help='Regex specifying the regions to plot.')
    parser.add_argument('-f', '--feature', default='.*', help='Regex specifying the features to plot.')
    
    args = parser.parse_args()
    return args

def get_feature_distribution(acc,
    writer: csv.writer,
    outdir: str,
    feature: str, 
    region: str,
    datasets: str,
    ) -> None:
    """
    For the given datasets and the analysis region, plot the feature distribution 
    for the combined classes, and compute the mean and standard deviation.
    """
    acc.load(feature)
    h = acc[feature]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)

    h = rebin_histogram(h, feature)

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (2, 1)}, sharex=True)

    hist.plot1d(
        h[re.compile(datasets)],
        ax=ax,
        overlay="dataset",
    )
    ax.set_yscale('log')
    ax.set_ylim(1e0,1e6)

    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        new_label = get_dataset_tag(label)
        handle.set_label(new_label)
    
    ax.legend(title="Dataset", handles=handles)

    ax.text(0,1,REGION_TAGS[region],
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes,
    )

    hist.plot1d(
        h[re.compile(datasets)].integrate("dataset"),
        ax=rax,
    )
    rax.set_yscale('log')
    rax.set_ylim(1e0,1e6)
    
    # Compute mean and standard deviation for the total dataset
    h = h[re.compile(datasets)].integrate("dataset")
    xcenters = h.axes()[0].centers()
    sumw = h.values()[()]

    mean = np.average(xcenters, weights=sumw)
    rax.axvline(mean, ymin=0, ymax=1, color='red', lw=2, label=f"Mean: {mean:.2f}")

    variance = np.average((xcenters - mean)**2, weights=sumw)
    std = np.sqrt(variance)

    rax.fill_betweenx((1e0,1e6), mean-std, mean+std, alpha=0.3, color="red", label=f"$\pm 1\sigma: {std:.2f}$")

    handles, labels = rax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label == "None":
            handle.set_label("Total")
    
    rax.legend(handles=handles)

    # Write to CSV file
    writer.writerow([feature, mean, std])

    outpath = pjoin(outdir, f"{feature}.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def main():
    args = parse_cli()
    inpath = args.inpath
    acc = dir_archive(inpath)
    acc.load('sumw')

    outtag = re.findall('merged_.*', inpath)[0].rstrip('/')

    dataset_dict = {
        'sr_vbf' : {
            'data' : 'MET_2017', 
            'mc'   : '(VBF_HToInvisible.*M125|ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EWK.*|WJetsToLNu_Pt-FXFX.*).*2017', 
        },
        'cr_vbf_qcd' : {
            'data'   : 'MET_2017', 
            'mc'     : '(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EWK.*|WJetsToLNu_Pt-FXFX.*).*2017', 
        },
        'cr_1m_vbf': {
            'data' : 'MET_2017',
            'mc'   : '(EWKW.*|DYJetsToLL_Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*2017',
        },
        'cr_2m_vbf': {
            'data' : 'MET_2017',
            'mc'   : '(EWKZ.*ZToLL.*|DYJetsToLL_Pt_FXFX.*).*2017',
        },
        'cr_1e_vbf': {
            'data' : 'EGamma_2017',
            'mc'   : '(EWKW.*|EWKZ.*ZToLL.*|DYJetsToLL_Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*2017',
        },
        'cr_2e_vbf': {
            'data' : 'EGamma_2017',
            'mc'   : '(EWKZ.*ZToLL.*|DYJetsToLL_Pt_FXFX.*).*2017',
        },
        'cr_g_vbf' : {
            'data' : 'EGamma_2017',
            'mc'   : '(GJets_DR-0p4.*|VBFGamma.*).*2017',
        },
    }

    regions = dataset_dict.keys()

    # List of features we want to plot and get mean+std for
    features = [
        "mjj",
        "detajj",
        "dphijj",
        "ak4_pt0",
        "ak4_pt1",
        "ak4_eta0",
        "ak4_eta1",
        "mjj_maxmjj",
        "detajj_maxmjj",
        "dphijj_maxmjj",
        "ak4_pt0_maxmjj",
        "ak4_pt1_maxmjj",
        "ak4_eta0_maxmjj",
        "ak4_eta1_maxmjj",
        "ht",
        "recoil",
        "dphi_ak40_met",
        "dphi_ak41_met",
    ]
    
    for region in regions:
        if not re.match(args.region, region):
            continue
        
        dataset_info = dataset_dict[region]
        
        for label, datasets in tqdm(dataset_info.items(), desc=f"Region {region}", leave=False):
            outdir = f'./output/{outtag}/features/{region}/{label}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            # The CSV file which will have the mean and standard deviation values per feature
            csvdir = outdir.replace(label, 'csv')
            if not os.path.exists(csvdir):
                os.makedirs(csvdir)
            
            csvpath = pjoin(csvdir, f'{region}_{label}_feature_props.csv')
            
            with open(csvpath, 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['Feature name', 'Mean', 'Standard deviation'])

                for feature in tqdm(features, desc="Plotting features", leave=False):
                    if not re.match(args.feature, feature):
                        continue
                    
                    get_feature_distribution(
                        acc,
                        writer=writer,
                        outdir=outdir,
                        feature=feature,
                        region=region,
                        datasets=datasets,
                    )

if __name__ == '__main__':
    main()