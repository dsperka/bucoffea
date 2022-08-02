#!/usr/bin/env python
import argparse
import os
import re
import sys
import uproot
import numpy as np
import mplhep as hep

from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from coffea import hist
from coffea.hist import poisson_interval
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio, lumi
from bucoffea.helpers.paths import bucoffea_path
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

# Suppress true_divide warnings
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'small',
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

Bin = hist.Bin

binnings = {
#    'dimuon_mass': Bin('dimuon_mass', r'$M_{\mu\mu} \ (GeV)$', 75, 0., 75.)
}

ylims = {
#    'dimuon_mass' : (1e0,1e8)
}


legend_labels = {
    'GJets_(DR-0p4).*' : "QCD $\\gamma$+jets",
    '(VBFGamma|GJets_SM.*EWK).*' : "EWK $\\gamma$+jets",
    'DY.*' : "QCD Z$\\rightarrow\\ell\\ell$",
    'EWKZ.*ZToLL.*' : "EWK Z$\\rightarrow\\ell\\ell$",
    'WN*J.*LNu.*' : "QCD W$\\rightarrow\\ell\\nu$",
    'EWKW.*LNu.*' : "EWK W$\\rightarrow\\ell\\nu$",
    'ZN*JetsToNuNu.*.*' : "QCD Z$\\rightarrow\\nu\\nu$",
    'EWKZ.*ZToNuNu.*' : "EWK Z$\\rightarrow\\nu\\nu$",
    'QCD.*' : "QCD Estimation",
    'Top.*' : "Top quark",
    'Diboson.*' : "WW/WZ/ZZ",
    'MET|Single(Electron|Photon|Muon)|EGamma.*' : "Data",
    'VBF_HToInv.*' : "VBF H(inv)",
}
legend_titles = {
    'cr_2m_zp2mu2nu' : 'Dimuon Selection'  
}

colors = {
    'DY.*' : '#ffffcc',
    'EWKW.*' : '#c6dbef',
    'EWKZ.*ZToLL.*' : '#d5bae2',
    'EWKZ.*ZToNuNu.*' : '#c4cae2',
    '.*Diboson.*' : '#4292c6',
    'Top.*' : '#6a51a3',
    '.*HF (N|n)oise.*' : '#08306b',
    '.*TT.*' : '#6a51a3',
    '.*ST.*' : '#9e9ac8',
    'ZN*JetsToNuNu.*' : '#31a354',
    'WJets.*' : '#feb24c',
    'GJets_(DR-0p4|HT).*' : '#fc4e2a',
    '(VBFGamma|GJets_SM.*EWK).*' : '#a76b51',
    'QCD.*' : '#a6bddb',
}

def plot_data_mc(acc, outtag, year, data, mc, data_region, mc_region, distribution='mjj', plot_signal=True, mcscale=1, binwnorm=None, fformats=['pdf'], qcd_file=None, jes_file=None, ulxs=True):
    """
    Main plotter function to create a stack plot of data to background estimation (from MC).
    """
    print("test1")
    acc.load(distribution)
    h = acc[distribution]
    print("test2")

    # Set up overflow bin for mjj
    overflow = 'none'

    # Pre-processing of the histogram, merging datasets, scaling w.r.t. XS and lumi
    print("identifiers 1")
    print(h[mc].identifiers('dataset'))
    h = merge_extensions(h, acc, reweight_pu=False)
    print("identifiers 2")
    print(h[mc].identifiers('dataset'))
    scale_xs_lumi(h, ulxs=ulxs, mcscale=mcscale)
    h = merge_datasets(h)
    print("identifiers 3")
    print(h[mc].identifiers('dataset'))

    print("test3")

    ## Rebin the histogram if necessary
    if distribution in binnings.keys():
        new_ax = binnings[distribution]
        h = h.rebin(new_ax.name, new_ax)

    # Specifically rebin dphitkpf distribution: Merge the bins in the tails
    # Annoying approach but it works (due to float precision problems)
    #elif distribution == 'dphitkpf':
    #    new_bins = [ibin.lo for ibin in h.identifiers('dphi') if ibin.lo < 2] + [3.5]
    #    
    #    new_ax = hist.Bin('dphi', r'$\Delta\phi_{TK,PF}$', new_bins)
    #    h = h.rebin('dphi', new_ax)

    print("test4")

    print("test5")

    # This sorting messes up in SR for some reason
    if data_region != 'sr_vbf':
        h.axis('dataset').sorting = 'integral'

    h_data = h.integrate('region', data_region)
    h_mc = h.integrate('region', mc_region)
    
    print("test6")

    # Get the QCD template (HF-noise estimation), only to be used in the signal region
    if 'sr_vbf' in data_region:
        # If a path to HF-noise estimate file has been given, use it!
        # Otherwise, take the one from the relevant output directory
        if qcd_file:
            qcdfilepath = qcd_file
        else:    
            qcdfilepath = f'output/{outtag}/qcd_estimate/vbfhinv_hf_estimate.root'
        
        # Make sure that the HF-noise estimate ROOT file points to a valid path
        assert os.path.exists(qcdfilepath), f"HF-noise file cannot be found: {qcdfilepath}"

        h_qcd = uproot.open(qcdfilepath)[f'qcd_estimate_{distribution}_{year}']

    print("test7")

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

    # Build the MC stack
    print("identifiers")
    print(h[mc].identifiers('dataset'))
    datasets = list(map(str, h[mc].identifiers('dataset')))
    print("mc stack datasets")
    print(datasets)
    
    print("test8")

    plot_info = {
        'label' : datasets,
        'sumw' : [],
    }

    for dataset in datasets:
        sumw = h_mc.integrate('dataset', dataset).values(overflow=overflow)[()]

        plot_info['sumw'].append(sumw)

    print("test9")

    # Add the HF-noise contribution (for signal region only)
    if data_region == 'sr_vbf':
        plot_info['label'].insert(6, 'HF Noise Estimate')
        plot_info['sumw'].insert(6, h_qcd.values * mcscale)

    fig, ax, rax = fig_ratio()
    
    print("test10")

    # Plot data
    hist.plot1d(h_data[data], ax=ax, overflow=overflow, overlay='dataset', binwnorm=binwnorm, error_opts=data_err_opts)

    xedges = h_data.integrate('dataset').axes()[0].edges(overflow=overflow)

    print("test11")

    # Plot MC stack
    hep.histplot(plot_info['sumw'], xedges, 
        ax=ax,
        label=plot_info['label'], 
        histtype='fill',
        binwnorm=binwnorm,
        stack=True
        )

    print("test12")

    # Plot VBF H(inv) signal if we want to
    if plot_signal:
        signal = re.compile(f'VBF_HToInvisible.*withDipoleRecoil.*{year}')

        signal_line_opts = {
            'linestyle': '-',
            'color': 'crimson',
        }

        h_signal = h.integrate('region', mc_region)[signal]
        h_signal.scale(mcscale)

        hist.plot1d(
            h_signal,
            ax=ax,
            overlay='dataset',
            overflow=overflow,
            line_opts=signal_line_opts,
            binwnorm=binwnorm,
            clear=False
        )

    print("test13")

    ax.set_yscale('log')
    if distribution == 'mjj':
        ax.set_ylim(1e-3,1e5)
        ax.set_ylabel('Events / GeV')
    elif distribution == 'cnn_score':
        if data_region in ['cr_2m_vbf', 'cr_2e_vbf']:
            ax.set_ylim(1e-2,1e4)
        else:
            ax.set_ylim(1e-1,1e4)
    else:
        if distribution in ylims.keys():
            ax.set_ylim(ylims[distribution])
        else:
            ax.set_ylim(1e0,1e8)
        ax.set_ylabel('Events')
    
    print("test14")

    if distribution == 'mjj':
        ax.set_xlim(left=0.)

    ax.yaxis.set_ticks_position('both')

    print("test15")

    # Update legend labels and plot styles
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        for datasetregex, new_label in legend_labels.items():
            col = None
            if re.match(datasetregex, label):
                handle.set_label(new_label)
            for k, v in colors.items():
                if re.match(k, label):
                    col = v
                    break

            if col:
                handle.set_color(col)
                handle.set_linestyle('-')
                handle.set_edgecolor('k')

    try:
        ax.legend(title=legend_titles[data_region], handles=handles, ncol=2)
    except KeyError:
        ax.legend(handles=handles, ncol=2)

    print("test16")

    # Plot ratio
    h_data = h_data.integrate('dataset', data)
    h_mc = h_mc.integrate('dataset', mc)

    print("test17")

    sumw_data, sumw2_data = h_data.values(overflow=overflow, sumw2=True)[()]
    sumw_mc = h_mc.values(overflow=overflow)[()]
    
    # Add the HF-noise contribution to the background expectation
    if data_region == 'sr_vbf':
        sumw_mc = sumw_mc + h_qcd.values * mcscale

    r = sumw_data / sumw_mc
    rerr = np.abs(poisson_interval(r, sumw2_data / sumw_mc**2) - r)

    r[np.isnan(r) | np.isinf(r)] = 0.
    rerr[np.isnan(rerr) | np.isinf(rerr)] = 0.

    print("test18")

    hep.histplot(
        r,
        xedges,
        yerr=rerr,
        ax=rax,
        histtype='errorbar',
        **data_err_opts
    )

    print("test19")

    xlabels = {
        'mjj': r'$M_{jj} \ (GeV)$',
        'ak4_eta0': r'Leading Jet $\eta$',
        'ak4_eta1': r'Trailing Jet $\eta$',
    }

    if distribution in xlabels.keys():
        ax.set_xlabel(xlabels[distribution])
        rax.set_xlabel(xlabels[distribution])
    
    print("test20")

    rax.set_ylabel('Data / MC')
    rax.set_ylim(0.5,1.5)
    loc1 = MultipleLocator(0.2)
    loc2 = MultipleLocator(0.1)
    rax.yaxis.set_major_locator(loc1)
    rax.yaxis.set_minor_locator(loc2)

    rax.yaxis.set_ticks_position('both')

    sumw_denom, sumw2_denom = h_mc.values(overflow=overflow, sumw2=True)[()]

    unity = np.ones_like(sumw_denom)
    denom_unc = poisson_interval(unity, sumw2_denom / sumw_denom ** 2)
    opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}
    
    print("test21")

    rax.fill_between(
        xedges,
        np.r_[denom_unc[0], denom_unc[0, -1]],
        np.r_[denom_unc[1], denom_unc[1, -1]],
        **opts
    )

    # If a JES/JER uncertainty file is given, plot the uncertainties in the ratio pad
    if jes_file and distribution == 'mjj':
        jes_src = uproot.open(jes_file)
        h_jerUp = jes_src[f'MTR_{year}_jerUp']
        h_jerDown = jes_src[f'MTR_{year}_jerDown']
        h_jesTotalUp = jes_src[f'MTR_{year}_jesTotalUp']
        h_jesTotalDown = jes_src[f'MTR_{year}_jesTotalDown']

        # Combine JER + JES
        jecUp = 1 + np.hypot(np.abs(h_jerUp.values - 1), np.abs(h_jesTotalUp.values - 1))
        jecDown = 1 - np.hypot(np.abs(h_jerDown.values - 1), np.abs(h_jesTotalDown.values - 1))

        # Since we're looking at data/MC, take the reciprocal of these variations
        jecUp = 1/jecUp
        jecDown = 1/jecDown

        opts = {"step": "post", "facecolor": "blue", "alpha": 0.3, "linewidth": 0, "label": "JES+JER"}

        rax.fill_between(
            xedges,
            np.r_[jecUp, jecUp[-1]],
            np.r_[jecDown, jecDown[-1]],
            **opts
        )

        rax.legend()

    print("test22")

    rax.grid(axis='y',which='both',linestyle='--')

    rax.axhline(1., xmin=0, xmax=1, color=(0,0,0,0.4), ls='--')

    fig.text(0., 1., '$\\bf{CMS}$ internal',
                fontsize=14,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes
               )

    fig.text(1., 1., f'{lumi(year, mcscale):.1f} fb$^{{-1}}$ ({year})',
                fontsize=14,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
               )

    print("test23")

    outdir = f'./output/{outtag}/{data_region}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # For each file format (PDF, PNG etc.), save the plot
    for fformat in fformats:
        outpath = pjoin(outdir, f'{data_region}_data_mc_{distribution}_{year}.{fformat}')
        fig.savefig(outpath)

    plt.close(fig)
