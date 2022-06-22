#!/usr/bin/env python

import os
import uproot
import re
import sys
import argparse
import numpy as np
import ROOT as r

from coffea import hist
from coffea.hist.export import export1d
from coffea.hist.plot import plot1d
from coffea.util import load

from pprint import pprint
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from bucoffea.plot.util import scale_xs_lumi, merge_extensions, merge_datasets

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to the merged coffea accumulator.')
    parser.add_argument('-v', '--variable', default='cnn_score', help='Name of the variable to compute theory uncertainties.')
    parser.add_argument('-y', '--years', nargs='*', type=int, default=[2017, 2018], help='List of years to run.')
    args = parser.parse_args()
    return args

def from_coffea(inpath, outfile, variable='cnn_score', years=[2017, 2018]):
    """
    Load the histograms from the coffea accumulator, convert them into an Uproot format
    and save them in a ROOT file for later use.
    """
    acc = dir_archive(
                        inpath,
                        serialized=True,
                        compression=0,
                        memsize=1e3,
                        )

    # Merging, scaling, etc
    acc.load('sumw')
    acc.load('sumw_pileup')
    acc.load('nevents')
    
    # The list of histograms we are interested in
    distributions = [
        variable,
        f'{variable}_unc',
        f'{variable}_noewk',
    ]

    for distribution in distributions:
        acc.load(distribution)
        acc[distribution] = merge_extensions(
                                            acc[distribution],
                                            acc, 
                                            reweight_pu=not ('nopu' in distribution)
                                            )
        scale_xs_lumi(acc[distribution])
        acc[distribution] = merge_datasets(acc[distribution])

        # Do rebinning based on the variable
        if variable == 'mjj':
            mjj_ax = hist.Bin('mjj', r'$M_{jj}$ (GeV)', [200, 400, 600, 900, 1200, 1500, 2000, 2750, 3500, 5000])
            acc[distribution] = acc[distribution].rebin(acc[distribution].axis('mjj'), mjj_ax)
        elif variable in ['cnn_score', 'dnn_score']:
            score_ax = hist.Bin("score", "Neural network score", 25, 0, 1)
            acc[distribution] = acc[distribution].rebin(acc[distribution].axis('score'), score_ax)
        else:
            raise ValueError(f'Unsupported variable name: {variable}')

    pprint(acc[distribution].axis('dataset').identifiers())
    f = uproot.recreate(outfile)
    
    for year in years:
        # QCD Z(vv)
        h_z = acc[variable][re.compile(f'ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX_{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        f[f'z_qcd_{variable}_nominal_{year}'] = export1d(h_z)

        # QCD W(lv)
        h_w = acc[variable][re.compile(f'WJetsToLNu_Pt-FXFX_{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        f[f'w_qcd_{variable}_nominal_{year}'] = export1d(h_w)

        # QCD gamma+jets
        h_ph = acc[variable][re.compile(f'GJets_DR-0p4_HT_MLM_{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        f[f'gjets_qcd_{variable}_nominal_{year}'] = export1d(h_ph)

        # Scale + PDF variations for QCD Z(vv)
        h_z_unc = acc[f'{variable}_unc'][re.compile(f'ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX_{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        for unc in map(str, h_z_unc.axis('uncertainty').identifiers()):
            if 'goverz' in unc or 'ewkcorr' in unc:
                continue
            h = h_z_unc.integrate(h_z_unc.axis('uncertainty'), unc)
            f[f'z_qcd_{variable}_{unc}_{year}'] = export1d(h)

        # EWK variations for QCD Z(vv)
        # Get EWK down variation first
        h_z_unc_ewk = acc[f'{variable}_noewk'][re.compile(f'ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX_{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        f[f'z_qcd_{variable}_unc_w_ewkcorr_overz_common_down_{year}'] = export1d(h_z_unc_ewk)

        # Get EWK up variation
        h_z_unc_ewk.scale(-1)
        h_z_diff = h_z.copy().add(h_z_unc_ewk)
        h_z_unc_ewk_down = h_z.add(h_z_diff) 
        f[f'z_qcd_{variable}_unc_w_ewkcorr_overz_common_up_{year}'] = export1d(h_z_unc_ewk_down)

        # EWK variations for QCD W(lv)
        # Get EWK down variation first
        h_w_unc_ewk = acc['mjj_noewk'][re.compile(f'WJetsToLNu_Pt-FXFX_{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        f[f'w_qcd_{variable}_unc_w_ewkcorr_overz_common_down_{year}'] = export1d(h_w_unc_ewk)

        # Get EWK up variation
        h_w_unc_ewk.scale(-1)
        h_w_diff = h_w.copy().add(h_w_unc_ewk)
        h_w_unc_ewk_down = h_w.add(h_w_diff)
        f[f'w_qcd_{variable}_unc_w_ewkcorr_overz_common_up_{year}'] = export1d(h_w_unc_ewk_down)

        # Scale + PDF variations for QCD gamma+jets
        h_ph_unc = acc[f'{variable}_unc'][re.compile(f'GJets_DR-0p4_HT_MLM_{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        for unc in map(str, h_ph_unc.axis('uncertainty').identifiers()):
            if 'zoverw' in unc or 'ewkcorr' in unc:
                continue
            h = h_ph_unc.integrate(h_ph_unc.axis('uncertainty'), unc)
            f[f'gjets_qcd_{variable}_{unc}_{year}'] = export1d(h)

        # EWK variations for QCD photons
        # Get EWK down variation first
        h_ph_unc_ewk = acc[f'{variable}_noewk'][re.compile(f'GJets_DR-0p4_HT_MLM_{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        f[f'gjets_qcd_{variable}_unc_w_ewkcorr_overz_common_down_{year}'] = export1d(h_ph_unc_ewk)

        # Get EWK up variation
        h_ph_unc_ewk.scale(-1)
        h_ph_diff = h_ph.copy().add(h_ph_unc_ewk)
        h_ph_unc_ewk_down = h_ph.add(h_ph_diff)
        f[f'gjets_qcd_{variable}_unc_w_ewkcorr_overz_common_up_{year}'] = export1d(h_ph_unc_ewk_down)

        # EWK V
        h_z = acc[variable][re.compile(f'EWKZ2Jets.*ZToNuNu.*{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        f[f'z_ewk_{variable}_nominal_{year}'] = export1d(h_z)

        h_w = acc[variable][re.compile(f'EWKW2Jets_WToLNu_M-50-mg_{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        f[f'w_ewk_{variable}_nominal_{year}'] = export1d(h_w)

        h_ph = acc[variable][re.compile(f'VBFGamma.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        f[f'gjets_ewk_{variable}_nominal_{year}'] = export1d(h_ph)

        # Scale + PDF variations for EWK Z
        h_z_unc = acc[f'{variable}_unc'][re.compile(f'EWKZ2Jets.*ZToNuNu.*{year}')].integrate('region', 'sr_vbf_no_veto_all').integrate('dataset')
        for unc in map(str, h_z_unc.axis('uncertainty').identifiers()):
            if 'goverz' in unc or 'ewkcorr' in unc:
                continue
            h = h_z_unc.integrate(h_z_unc.axis('uncertainty'), unc)
            f[f'z_ewk_{variable}_{unc}_{year}'] = export1d(h)

        # Scale + PDF variations for EWK photons
        h_ph_unc = acc[f'{variable}_unc'][re.compile(f'VBFGamma.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        for unc in map(str, h_ph_unc.axis('uncertainty').identifiers()):
            if 'zoverw' in unc or 'ewkcorr' in unc:
                continue
            h = h_ph_unc.integrate(h_ph_unc.axis('uncertainty'), unc)
            f[f'gjets_ewk_{variable}_{unc}_{year}'] = export1d(h)


def make_ratios(infile, variable='cnn_score', years=[2017, 2018]):
    """
    Given the individual shape variations stored in infile, compute and
    save the variations in the Z(vv)/W(lv) and gamma+jets/Z(vv) ratios.
    """
    f = r.TFile(infile)
    of = r.TFile(infile.replace('.root','_ratio.root'),'RECREATE')
    of.cd()

    # Z(vv) / W(lv) ratios (scale + PDF variations)
    for source in ['ewk','qcd']:
        for year in years:
            denominator = f.Get(f'w_{source}_{variable}_nominal_{year}')
            for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
                if not name.startswith(f'z_{source}'):
                    continue
                if not f"{year}" in name or 'ewkcorr' in name:
                    continue
                ratio = f.Get(name).Clone(f'ratio_{name}')
                ratio.Divide(denominator)
                ratio.SetDirectory(of)
                ratio.Write()
    
    # Z(vv) / W(lv) ratios (up and down EWK variations)
    for year in years:
        for vartype in ['up', 'down']:
            varied_z_name = f'z_qcd_{variable}_unc_w_ewkcorr_overz_common_{vartype}_{year}'
            varied_w = f.Get(f'w_qcd_{variable}j_unc_w_ewkcorr_overz_common_{vartype}_{year}')
            varied_ratio = f.Get(varied_z_name).Clone(f'ratio_{varied_z_name}')
            varied_ratio.Divide(varied_w)
            varied_ratio.SetDirectory(of)
            varied_ratio.Write()  

        nominal_z_name = f'z_qcd_{variable}_nominal_{year}'
        nominal_w = f.Get(f'w_qcd_{variable}_nominal_{year}')
        nominal_ratio = f.Get(nominal_z_name).Clone(f'ratio_{nominal_z_name}')
        nominal_ratio.Divide(nominal_w)
        nominal_ratio.SetDirectory(of)    
        nominal_ratio.Write()  

    # Gamma+jets / Z(vv) ratios (scale + PDF variations)
    for source in ['ewk','qcd']:
        for year in years:
            denominator = f.Get(f'z_{source}_{variable}_nominal_{year}')
            for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
                if not name.startswith(f'gjets_{source}'):
                    continue
                if not f"{year}" in name or 'ewkcorr' in name:
                    continue
                ratio = f.Get(name).Clone(f'ratio_{name}')
                ratio.Divide(denominator)
                ratio.SetDirectory(of)
                ratio.Write()

    # Gamma+jets / Z(vv) ratios (up and down EWK variations)
    for year in years:
        for vartype in ['up', 'down']:
            varied_g_name = f'gjets_qcd_{variable}_unc_w_ewkcorr_overz_common_{vartype}_{year}'
            varied_z = f.Get(f'z_qcd_{variable}_unc_w_ewkcorr_overz_common_{vartype}_{year}')
            varied_ratio = f.Get(varied_g_name).Clone(f'ratio_{varied_g_name}')
            varied_ratio.Divide(varied_z)
            varied_ratio.SetDirectory(of)
            varied_ratio.Write()

        nominal_g_name = f'gjets_qcd_{variable}_nominal_{year}'
        nominal_z = f.Get(f'z_qcd_{variable}_nominal_{year}')
        nominal_ratio = f.Get(nominal_g_name).Clone(f'ratio_{nominal_g_name}')
        nominal_ratio.Divide(nominal_z)
        nominal_ratio.SetDirectory(of)
        nominal_ratio.Write()

    of.Close()
    return str(of.GetName())

def make_uncertainties(infile, variable='cnn_score', years=[2017, 2018]):
    """
    Given the variations in Z/W and gamma/Z ratios, compute uncertainties
    on the ratios and save them.
    """

    f = r.TFile(infile)
    of = r.TFile(infile.replace('_ratio','_ratio_unc'),'RECREATE')
    of.cd()

    # Uncertainty in Z / W ratios (scale + PDF variations)
    for source in ['ewk','qcd']:
        for year in years:
            nominal = f.Get(f'ratio_z_{source}_{variable}_nominal_{year}')
            for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
                m = bool(re.match(f'.*z_{source}_{variable}_unc_(.*)_{year}', name)) and 'ewkcorr' not in name
                if not m:
                    continue
                ratio = f.Get(name)
                variation = ratio.Clone(f'uncertainty_{name}')

                # Content: Varied ratio / Nominal ratio
                variation.Divide(nominal)

                variation.SetDirectory(of)
                variation.Write()
                
                ratio.SetDirectory(of)
                ratio.Write()
    
    # Uncertainty in Z / W ratios (up and down EWK variations)
    for year in years:
        nominal = f.Get(f'ratio_z_qcd_{variable}_nominal_{year}')
        for vartype in ['up', 'down']:
            varied_name = f'ratio_z_qcd_{variable}_unc_w_ewkcorr_overz_common_{vartype}_{year}'
            varied = f.Get(varied_name)
            # Variation: (varied Z / W) / (nominal Z / W)
            variation = varied.Clone(f'uncertainty_{varied_name}')
            variation.Divide(nominal)
    
            variation.SetDirectory(of)
            variation.Write()

            varied.SetDirectory(of)
            varied.Write()
    
    # Copy the EWK uncertainty for QCD Z / W ratio
    # apply to the EWK Z / W ratio (for now)
    for year in years:
        for vartype in ['up', 'down']:
            qcd_unc_name = f'uncertainty_ratio_z_qcd_{variable}_unc_w_ewkcorr_overz_common_{vartype}_{year}'
            qcd_unc = of.Get(qcd_unc_name)
            ewk_unc = qcd_unc.Clone(f'{qcd_unc_name.replace("qcd", "ewk")}')

            ewk_unc.SetDirectory(of)
            ewk_unc.Write()

    # Uncertainty in GJets / Z ratios (scale + PDF variations)
    for source in ['ewk','qcd']:
        for year in years:
            nominal = f.Get(f'ratio_gjets_{source}_{variable}_nominal_{year}')
            for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
                m = bool(re.match(f'.*gjets_{source}_{variable}_unc_(.*)_{year}', name)) and 'ewkcorr' not in name
                if not m:
                    continue
                ratio = f.Get(name)
                variation = ratio.Clone(f'uncertainty_{name}')

                # Content: Varied ratio / Nominal ratio
                variation.Divide(nominal)

                variation.SetDirectory(of)
                variation.Write()
                
                ratio.SetDirectory(of)
                ratio.Write()

    # Uncertainty in GJets / Z ratios (up and down EWK variations)
    for year in years:
        nominal = f.Get(f'ratio_gjets_qcd_{variable}_nominal_{year}')
        for vartype in ['up', 'down']:
            varied_name = f'ratio_gjets_qcd_{variable}_unc_w_ewkcorr_overz_common_{vartype}_{year}'
            varied = f.Get(varied_name)
            # Variation: (varied Z / W) / (nominal Z / W)
            variation = varied.Clone(f'uncertainty_{varied_name}')
            variation.Divide(nominal)
    
            variation.SetDirectory(of)
            variation.Write()

            varied.SetDirectory(of)
            varied.Write()
    
    # Copy the EWK uncertainty for QCD GJets / Z ratio
    # apply to the EWK GJets / Z ratio (for now)
    for year in years:
        for vartype in ['up', 'down']:
            qcd_unc_name = f'uncertainty_ratio_gjets_qcd_{variable}_unc_w_ewkcorr_overz_common_{vartype}_{year}'
            qcd_unc = of.Get(qcd_unc_name)
            ewk_unc = qcd_unc.Clone(f'{qcd_unc_name.replace("qcd", "ewk")}')

            ewk_unc.SetDirectory(of)
            ewk_unc.Write()

    of.Close()
    return str(of.GetName())


def main():
    args = parse_cli()
    inpath = args.inpath

    # Get the output tag for output directory naming
    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]
    
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = pjoin(outdir, f'vbf_z_w_gjets_theory_unc.root')

    # Prepare histograms for individual variations for Z(vv), W(lv) and gamma+jets
    from_coffea(
        inpath, 
        outfile,
        variable=args.variable,
        years=args.years,
    )
    
    # Get variations of the ratios
    outfile = make_ratios(
        outfile, 
        variable=args.variable,
        years=args.years,
    )
    
    # Compute uncertainties in the ratios and save them
    make_uncertainties(
        outfile,
        variable=args.variable,
        years=args.years,
    )

if __name__ == "__main__":
    main()
