#!/usr/bin/env python

import os
import sys
import re
import argparse
import numpy as np

from klepto.archives import dir_archive
from tabulate import tabulate
from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict

from bucoffea.helpers.paths import bucoffea_path
from bucoffea.plot.util import (
    merge_datasets, 
    merge_extensions, 
    scale_xs_lumi, 
)

pjoin = os.path.join


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='sr_vbf', help='Name of the region to look at.')
    parser.add_argument('--year', type=int, default=2017, help='Year to look at.')
    args = parser.parse_args()
    return args


def preprocess(h,acc):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    # h = merge_datasets(h)

    return h

def compare_cutflows(acc1, acc2, dataset, region='sr_vbf'):
    cutflow_name = f'cutflow_{region}'
    for acc in [acc1, acc2]:
        acc.load(cutflow_name)

    c1 = acc1[cutflow_name]
    c2 = acc2[cutflow_name]

    # Sort the cutflows in descending order
    c1 = OrderedDict(sorted(c1[dataset].items(), key=lambda item: item[1], reverse=True))
    c2 = OrderedDict(sorted(c2[dataset].items(), key=lambda item: item[1], reverse=True))

    table = []
    for cut in c1.keys():
        table.append([cut, c1[cut], c2[cut]])

    print(f'\n{dataset}\n')
    print(tabulate(table, headers=["Cut Name", "Yield 1", "Yield 2"], floatfmt=".0f"))

def compare_integrals(acc1, acc2, distribution1, distribution2, region='sr_vbf', year=2017):
    """
    Given two accumulators compare the total sample integrals per dataset.
    """
    acc1.load(distribution1)
    acc2.load(distribution2)

    h1 = preprocess(acc1[distribution1], acc1)
    h2 = preprocess(acc2[distribution2], acc2)

    table = []

    # Regular expressions specifying which datasets we want to look at for each region
    datasets_to_show = {
        "sr_vbf" : re.compile(f"(Z\dJetsToNuNu|WJetsToLNu|MET|EWK|.*HToInvisible.*).*{year}"),
        "cr_1m_vbf" : re.compile(f"(WJetsToLNu_Pt|MET|EWK).*{year}"),
        "cr_2m_vbf" : re.compile(f"(DYJetsToLL_Pt|MET|EWK).*{year}"),
        "cr_1e_vbf" : re.compile(f"(WJetsToLNu_Pt|Single(Ele|Pho)|EWK).*{year}"),
        "cr_2e_vbf" : re.compile(f"(DYJetsToLL_Pt|Single(Ele|Pho)|EWK).*{year}"),
        "cr_g_vbf" : re.compile(f"(GJets_DR-0p4|VBFGamma|Single(Ele|Pho)).*{year}"),
    }

    datasets = h1.identifiers("dataset")

    for dataset in tqdm(datasets):
        if not re.match(datasets_to_show[region], dataset.name):
            continue
        
        # Naming convention problem for MET 2017E dataset
        if re.match("MET.*2017E", dataset.name):
            dataset1 = "MET_ver3_2017E"
            dataset2 = "MET_ver1_2017E"
        else:
            dataset1 = dataset2 = dataset
        try:
            hist1 = h1.integrate("dataset", dataset1).integrate("region", region)
            hist2 = h2.integrate("dataset", dataset2).integrate("region", region)

            integral1 = np.sum(hist1.values()[()])
            integral2 = np.sum(hist2.values()[()])
        
            diff = (integral1-integral2) / integral1 * 100
        
            if np.isnan(diff) or np.isinf(diff):
                continue

            table.append([dataset, integral1, integral2, diff])
        
        except KeyError:
            continue

    print(tabulate(table, headers=["Process", "Integral1", "Integral2", "Diff (%)"], floatfmt=".3f"))

    # Also write this out to an output file
    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f"integral_comparison_{region}.txt")
    with open(outpath, 'w+') as f:
        f.write(f'Data region: {region}\n')
        f.write('Integral1: merged_2021-10-13_vbfhinv_ULv8_05Feb21\n')
        f.write('Integral2: merged_2022-06-06_vbfhinv_ULv8_05Feb21_withJetImages\n\n')
        f.write(tabulate(table, headers=["Process", "Integral1", "Integral2", "Diff (%)"], floatfmt=".3f"))

    print(f'Data saved at: {outpath}')


def main():
    args = parse_cli()

    # Analysis version
    inpath1 = bucoffea_path('submission/merged_2021-10-13_vbfhinv_ULv8_05Feb21')

    # ML version
    inpath2 = bucoffea_path('submission/merged_2022-06-06_vbfhinv_ULv8_05Feb21_withJetImages')

    for inpath in [inpath1, inpath2]:
        assert os.path.exists(inpath), f"Cannot find input: {inpath}"

    acc1 = dir_archive(inpath1)
    acc2 = dir_archive(inpath2)

    for acc in [acc1, acc2]:
        acc.load('sumw')

    compare_integrals(
        acc1, acc2,
        'mjj', 'cnn_score',
        region=args.region,
        year=args.year,
    )

    dataset_for_region = {
        'sr_vbf'    : 'MET_ver1_2017C',
        'cr_1m_vbf' : 'MET_ver1_2017C',
        'cr_2m_vbf' : 'MET_ver1_2017C',
        'cr_1e_vbf' : 'SingleElectron_ver1_2017C',
        'cr_2e_vbf' : 'SingleElectron_ver1_2017C',
        'cr_g_vbf'  : 'SinglePhoton_ver1_2017C',
    }

    compare_cutflows(acc1, acc2, dataset=dataset_for_region[args.region], region=args.region)

if __name__ == '__main__':
    main()