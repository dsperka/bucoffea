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

# Paths to the merged accumulators:
# Analysis version
PATH_TO_FIRST_ACC  = bucoffea_path("submission/merged_2022-07-20_vbfhinv_ULv8_05Feb21_2017")
# ML version
PATH_TO_SECOND_ACC = bucoffea_path("submission/merged_2022-07-25_vbfhinv_14Jul22_jetImages_processed")

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
        "cr_1m_vbf" : re.compile(f"(WJetsToLNu_Pt|MET|EWKW).*{year}"),
        "cr_2m_vbf" : re.compile(f"(DYJetsToLL.*Pt|MET|EWKZ).*{year}"),
        "cr_1e_vbf" : re.compile(f"(WJetsToLNu_Pt|Single(Ele|Pho)|EWKW).*{year}"),
        "cr_2e_vbf" : re.compile(f"(DYJetsToLL.*Pt|Single(Ele|Pho)|EWKZ).*{year}"),
        "cr_g_vbf" : re.compile(f"(GJets_DR-0p4|VBFGamma|Single(Ele|Pho)).*{year}"),
    }

    datasets = h1.identifiers("dataset")

    for dataset in tqdm(datasets):
        if not re.match(datasets_to_show[region], dataset.name):
            continue
        
        # Naming convention problems for several datasets
        if re.match("MET.*2017E", dataset.name):
            dataset1 = "MET_ver3_2017E"
            dataset2 = "MET_ver1_2017E"
        elif re.match("SingleElectron.*2017E", dataset.name):
            dataset1 = "SingleElectron_ver2_2017E"
            dataset2 = "SingleElectron_ver1_2017E"
        elif re.match("DYJetsToLL.*Pt.*", dataset.name):
            if "50To100" in dataset.name:
                continue
            dataset1 = dataset.name
            dataset2 = re.sub("Pt", "LHEFilterPtZ", dataset.name)
        else:
            dataset1 = dataset2 = dataset
        
        try:
            hist1 = h1.integrate("dataset", dataset1).integrate("region", region)
            hist2 = h2.integrate("dataset", dataset2).integrate("region", region)

            integral1 = np.sum(hist1.values(overflow='over')[()])
            integral2 = np.sum(hist2.values(overflow='over')[()])
        
            diff = (integral1-integral2) / integral1 * 100
        
            if np.isnan(diff) or np.isinf(diff):
                continue

            table.append([dataset, integral1, integral2, diff])
        
        except KeyError:
            continue

    # Print the table
    print(tabulate(table, headers=["Process", "Integral1", "Integral2", "Diff (%)"], floatfmt=".3f"))

    # Also write this out to an output file
    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f"integral_comparison_{region}.txt")
    with open(outpath, 'w+') as f:
        f.write(f'Data region: {region}\n')
        f.write(f'Integral1: {os.path.basename(PATH_TO_FIRST_ACC)}\n')
        f.write(f'Integral2: {os.path.basename(PATH_TO_SECOND_ACC)}\n\n')
        f.write(tabulate(table, headers=["Process", "Integral1", "Integral2", "Diff (%)"], floatfmt=".3f"))

    print(f'Data saved at: {outpath}')


def compare_yields_for_dataset(acc1, acc2, distribution1, distribution2, dataset, regions, year=2017):
    """Compare yields across regions for a given dataset."""
    acc1.load(distribution1)
    acc2.load(distribution2)

    h1 = preprocess(acc1[distribution1], acc1)
    h2 = preprocess(acc2[distribution2], acc2)

    table = []

    for region in tqdm(regions, desc=f"Comparing dataset: {dataset}"):
        hist1 = h1.integrate("region", region).integrate("dataset", dataset)
        hist2 = h2.integrate("region", region).integrate("dataset", dataset)

        integral1 = np.sum(hist1.values()[()])
        integral2 = np.sum(hist2.values()[()])
    
        diff = (integral1-integral2) / integral1 * 100
    
        if np.isnan(diff) or np.isinf(diff):
            continue

        table.append([region, integral1, integral2, diff])


    print(f'{dataset}\n')
    print(tabulate(table, headers=["Region", "Integral1", "Integral2", "Diff (%)"], floatfmt=".3f"))

    # Also write this out to an output file
    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f"comparison_{dataset}.txt")
    with open(outpath, 'w+') as f:
        f.write(f'Dataset: {dataset}\n')
        f.write(f'Integral1: {os.path.basename(PATH_TO_FIRST_ACC)}\n')
        f.write(f'Integral2: {os.path.basename(PATH_TO_SECOND_ACC)}\n\n')
        f.write(tabulate(table, headers=["Region", "Integral1", "Integral2", "Diff (%)"], floatfmt=".3f"))

    print(f'Data saved at: {outpath}')


def main():
    args = parse_cli()

    for inpath in [PATH_TO_FIRST_ACC, PATH_TO_SECOND_ACC]:
        assert os.path.exists(inpath), f"Cannot find input: {inpath}"

    acc1 = dir_archive(PATH_TO_FIRST_ACC)
    acc2 = dir_archive(PATH_TO_SECOND_ACC)

    for acc in [acc1, acc2]:
        acc.load('sumw')

    compare_integrals(
        acc1, acc2,
        'mjj', 'cnn_score',
        region=args.region,
        year=args.year,
    )

    dataset_for_region = {
        'sr_vbf'    : 'MET_ver1_2017D',
        'cr_1m_vbf' : 'MET_ver1_2017D',
        'cr_2m_vbf' : 'MET_ver1_2017D',
        'cr_1e_vbf' : 'SingleElectron_ver1_2017C',
        'cr_2e_vbf' : 'SingleElectron_ver1_2017C',
        'cr_g_vbf'  : 'SinglePhoton_ver1_2017C',
    }

    compare_cutflows(acc1, acc2, dataset=dataset_for_region[args.region], region=args.region)

    compare_yields_for_dataset(acc1, acc2,
        'mjj', 'cnn_score',
        dataset='MET_ver1_2017D',
        regions=['sr_vbf', 'cr_1m_vbf', 'cr_2m_vbf'],
    )

if __name__ == '__main__':
    main()