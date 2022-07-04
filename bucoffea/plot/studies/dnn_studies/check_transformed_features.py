#!/usr/bin/env python

import os
import sys
import re

from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from tqdm import tqdm

from bucoffea.plot.util import (
    merge_extensions, 
    merge_datasets, 
    scale_xs_lumi, 
    get_dataset_tag, 
    rebin_histogram
)

pjoin = os.path.join

def plot_transformed_feature(acc, 
    outtag: str,
    feature: str, 
    dataset_regex: str, 
    region: str='sr_vbf_no_veto_all'
    ) -> None:
    """
    Plot the given transformed features for different datasets.
    """
    acc.load(feature)
    h = acc[feature]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = rebin_histogram(h, feature)

    h = h.integrate('region', region)

    fig, ax = plt.subplots()
    hist.plot1d(h[re.compile(dataset_regex)],
        ax=ax,
        overlay='dataset',
        density=True,
    )

    handles, labels = ax.get_legend_handles_labels()

    for handle, label in zip(handles, labels):
        pretty_label = get_dataset_tag(label)
        handle.set_label(pretty_label)
    
    ax.legend(title='Dataset', handles=handles)
    
    ax.set_ylabel('Normalized Counts')

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{feature}.pdf')
    fig.savefig(outpath)
    plt.close(fig)


def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')

    outtag = re.findall('merged_.*', inpath)[0].rstrip('/')

    features = [
        'mjj',
        'detajj',
        'mjj_transformed',
        'detajj_transformed',
        'dnn_score',
    ]

    for feature in tqdm(features, desc="Plotting features"):
        plot_transformed_feature(acc,
            outtag=outtag,
            feature=feature,
            dataset_regex="(ZNJetsToNuNu|EWKZ2Jets.*ZToNuNu|VBF_HToInv).*2017"
        )


if __name__ == '__main__':
    main()