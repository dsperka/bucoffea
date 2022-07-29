#!/usr/bin/env python
import copy
import os
import re
import uproot
import ROOT as r
import numpy as np

from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
from coffea import hist

from bucoffea.plot.util import (
    merge_datasets, 
    merge_extensions, 
    scale_xs_lumi, 
    URTH1
    )

from legacy_monojet import suppress_negative_bins

pjoin = os.path.join


def datasets(year, include_sr_data=False):
    """Datasets to read for each region."""
    data = {
        'cr_1m_vbf' : re.compile(f'MET_{year}'),
        'cr_2m_vbf' : re.compile(f'MET_{year}'),
        'cr_1e_vbf' : re.compile(f'EGamma_{year}'),
        'cr_2e_vbf' : re.compile(f'EGamma_{year}'),
        'cr_g_vbf'  : re.compile(f'EGamma_{year}'),
        'sr_vbf_no_veto_all' : re.compile(f'nomatch'),
    }
    
    # If we're unblinding, we want the data in the signal region as well
    if include_sr_data:
        data['sr_vbf'] = re.compile(f'MET_{year}')

    mc = {
        'sr_vbf_no_veto_all' : re.compile(f'(ttH_HToInvisible_M125.*|WH_WToQQ_Hinv_M125.*|ZH_ZToQQ_HToInv.*M125.*|(VBF|GluGlu)_HToInvisible.*M125.*|ggZH.*|ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|DYJetsToLL.*FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'cr_1m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|DYJetsToLL.*FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'cr_1e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|DYJetsToLL.*FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'cr_2m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|DYJetsToLL.*FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'cr_2e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|DYJetsToLL.*FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'cr_g_vbf' : re.compile(f'(GJets_DR-0p4.*|VBFGamma.*|QCD_data.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'sr_vbf' : re.compile('nomatch')
    }

    return data, mc

def legacy_dataset_name_vbf(dataset) -> str:
    """
    Given the dataset name, return the short-name to use to save under the ROOT file.
    """
    # Signal datasets
    m = re.match("VBF_HToInvisible_M(\d+)(_withDipoleRecoil)?(_PSweights)?_pow_pythia8_201[0-9]", dataset)
    if m:
        mh = m.groups()[0]
        if mh=="125":
            return "vbf"
        else:
            return f"vbf{mh}"
    
    m = re.match("ttH_HToInvisible_M(\d+)_pow_pythia8_201[0-9]", dataset)
    if m:
        mh = m.groups()[0]
        if mh=="125":
            return "tth"
        else:
            return f"tth{mh}"

    m = re.match("ZH_ZToQQ_HToInvisible_M(\d+)(_PSweights)?_pow_pythia8_201[0-9]", dataset)
    if m:
        mh = m.groups()[0]
        if mh=="125":
            return "zh"
        else:
            return f"zh{mh}"

    m = re.match("WH_WToQQ_Hinv_M(\d+)_201[0-9]", dataset)
    if m:
        mh = m.groups()[0]
        if mh=="125":
            return "wh"
        else:
            return f"wh{mh}"

    m = re.match("GluGlu_HToInvisible_M(\d+)_HiggspTgt190_pow_pythia8_201[0-9]", dataset)
    if m:
        mh = m.groups()[0]
        if mh=="125":
            return "ggh"
        else:
            return f"ggh{mh}"

    m = re.match("ggZH_ZToQQ_HToInvisible_M(\d+)(_PSweights)?_pow_pythia8_201[0-9]", dataset)
    if m:
        mh = m.groups()[0]
        if mh=="125":
            return "ggzh"
        else:
            return f"ggzh{mh}"

    m = re.match("WH_HToInv_JHU_ptH150_201[0-9]", dataset)
    if m:
        return "wh_jhu"
    m = re.match("ZH_HToInv_JHU_ptH150_201[0-9]", dataset)
    if m:
        return "zh_jhu"

    # Look for other background dataset patterns
    patterns = {
        'EWKZ\d?Jets.*ZToLL.*' : 'ewkzll',
        'EWKZ\d?Jets.*ZToNuNu.*' : 'ewkzjets',
        'EWKW.*' : 'ewkwjets',
        'QCD.*' : 'qcd',
        'Top.*' : 'top',
        'Diboson.*' : 'diboson',
        '(MET|EGamma).*' : 'data',
        'WJetsToLNu.*' : 'qcdwjets',
        'ZN*JetsToNuNu.*' : 'qcdzjets',
        'DYJets.*' : 'qcdzll',
        'GJets_DR-0p4.*HT.*' : 'qcdgjets',
        'VBFGamma.*' : 'ewkgjets',
    }

    for pat, ret in patterns.items():
        if re.match(pat, dataset):
            return ret

    raise RuntimeError(f'Cannot find legacy region name for dataset: "{dataset}"')

def legacy_region_name(region) -> str:
    """
    Given the region name used in the analysis, return the short
    region name to use while saving histograms to the ROOT file.
    """
    patterns = {
        'cr_2m_.*' : 'Zmm',
        'cr_2e_.*' : 'Zee',
        'cr_1m_.*' : 'Wmn',
        'cr_1e_.*' : 'Wen',
        'cr_g_.*' : 'gjets',
        'sr_.*' : 'signal',
    }

    for pat, ret in patterns.items():
        if re.match(pat, region):
            return ret
    raise RuntimeError(f'Cannot find legacy region name for region: "{region}"')

def recoil_bins_2016():
    return [ 250.,  280.,  310.,  340.,  370.,  400.,
             430.,  470.,  510., 550.,  590.,  640.,
             690.,  740.,  790.,  840.,  900.,  960.,
             1020., 1090., 1160., 1250., 1400.]

def mjj_bins_2016():
    return [200., 400., 600., 900., 1200., 1500.,
            2000., 2750., 3500., 5000.]


def nn_score_ax() -> hist.Bin:
    """Returns the new binning for the neural network score."""
    new_ax = hist.Bin("score", "Neural network score", 25, 0, 1)
    return new_ax


def export_coffea_histogram(h, overflow='over', axname='score', suppress_last_bin=False):
    '''Helper function to: coffea histogram -> (sumw, xedges) with the desired overflow behavior.'''
    if h.dim() != 1:
        raise RuntimeError('The dimension of the histogram must be 1.')

    sumw, sumw2 = h.values(overflow=overflow, sumw2=True)[()]
    xedges = h.axis(axname).edges()

    # Add the contents of the overflow to the last bin
    if overflow == 'over':
        sumw[-2] += sumw[-1]
        sumw2[-2] += sumw2[-1]
        sumw = np.r_[0, sumw[:-1], 0]
        sumw2 = np.r_[0, sumw2[:-1], 0]

    # If we want to suppress the last mjj bin, well here we go
    if suppress_last_bin:
        sumw = np.r_[sumw[:-2], 0, 0]
        sumw2 = np.r_[sumw2[:-2], 0, 0]

    return URTH1(edges=xedges, sumw=sumw, sumw2=sumw2)

def legacy_limit_input_vbf(acc,
    distribution='cnn_score',
    outdir='./output', 
    unblind=False, 
    years=[2017, 2018], 
    one_fifth_unblind=False, 
    mcscales=None,
    ) -> None:
    """
    Writes ROOT TH1 histograms to file as a limit input.

    :param acc: Accumulator (processor output)
    :type acc: coffea.processor.accumulator
    :param outdir: Output directory
    :type outdir: string
    """
    regions = [
                'cr_2m_vbf',
                'cr_1m_vbf',
                'cr_2e_vbf',
                'cr_1e_vbf',
                'cr_g_vbf',
                'sr_vbf_no_veto_all'
                ]
    
    # If unblinding, add observed data from the signal region
    if unblind or one_fifth_unblind:
        regions.append("sr_vbf")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get the distribution and pre-process
    h = copy.deepcopy(acc[distribution])
    if distribution in ['cnn_score', 'dnn_score']:
        newax = nn_score_ax()
        axname = 'score'
    elif distribution == 'mjj':
        newax = hist.Bin('mjj','$M_{jj}$ (GeV)', mjj_bins_2016())
        axname = 'mjj'
    else:
        raise RuntimeError(f'Limit input for VBF is not supported for distribution: {distribution}')
    
    # Rebin the distribution
    h = h.rebin(h.axis(newax.name), newax)
    
    h = merge_extensions(h, acc)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    for year in tqdm(years, desc="Making limit input files"):
        # Dump dataset mapping into txt files
        infofile = pjoin(outdir, f'dataset_mapping_{year}.txt')

        with open(infofile, 'w+') as infof:
            # Output ROOT file we're going to save (per year)
            f = uproot.recreate(pjoin(outdir, f'legacy_limit_vbf_{year}.root'))
            data, mc = datasets(year, include_sr_data=unblind or one_fifth_unblind)

            # Loop over regions and make histograms
            for region in regions:

                # Get the histogram for this region
                ih = h.integrate('region', region)

                # Keep track of the dataset name and histogram mapping in a table
                table = {
                    'Dataset name': [],
                    'Histogram name': [],
                }

                for dataset in map(str, ih.axis('dataset').identifiers()):
                    if not (data[region].match(dataset) or mc[region].match(dataset)):
                        continue

                    h_cof = ih.integrate('dataset', dataset)
                    
                    # If we're applying 1/5th unblinding on signal region, down-scale the MC by 0.2
                    if one_fifth_unblind and region == 'sr_vbf_no_veto_all':
                        h_cof.scale(0.2)

                    # If there is additional MC scaling specified via command line, do it here
                    if mc[region].match(dataset) and region in mcscales:
                        scale_val = mcscales[region]
                        h_cof.scale(scale_val)
                    
                    th1 = export_coffea_histogram(h_cof, axname=axname)
                    
                    try:
                        # Determine the histogram name that we will use to save this dataset
                        histo_name = f'{legacy_region_name(region)}_{legacy_dataset_name_vbf(dataset)}'
                        table['Dataset name'].append(dataset)
                        table['Histogram name'].append(histo_name)
                    except:
                        print(f"Skipping {dataset}")
                        continue

                    # Write the TH1 histogram object to output ROOT file
                    f[histo_name] = th1

                infof.write(f'Region: {region}\n\n')
                infof.write(tabulate(table, headers='keys'))
                infof.write('\n\n')
        
        # Add signal region data, if we're not unblinding
        if not (unblind or one_fifth_unblind):
            f[f'{legacy_region_name("sr_vbf")}_data'] = f[f'{legacy_region_name("sr_vbf")}_qcdzjets']
    
    # Merge the 2017 and 2018 histograms into a single file
    # under separate sub-directories
    merge_legacy_inputs(outdir)

def merge_legacy_inputs(outdir):
    '''
    Workaround for uproot's lack of subdirectory support.
    '''

    files = defaultdict(dict)
    for fname in os.listdir(outdir):
        m = re.match('legacy_limit_([a-z]*)_(\d+).root', fname)
        if not m:
            continue
        category, year = m.groups()
        files[year][category] = pjoin(outdir, fname)

    outfile = r.TFile(pjoin(outdir, f'legacy_limit_vbf.root'),'RECREATE')
    for year, ifiles in files.items():
        for category, file in ifiles.items():
            subdir = outfile.mkdir(f'category_{category}_{year}')
            infile = r.TFile(file)
            for key in infile.GetListOfKeys():
                print(key)
                h = key.ReadObj().Clone()
                h.SetTitle(h.GetName())
                h.SetDirectory(subdir)
                h.GetXaxis().SetTitle('mjj')
                suppress_negative_bins(h)
                subdir.Write()
