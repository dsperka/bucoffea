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

recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]

binnings = {
    'mjj': Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500.]),
    'cnn_score': Bin('score', r'CNN score', 25, 0, 1),
    'ak4_pt0': Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt1': Bin('jetpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(40,600,20)) + list(range(600,1000,20)) ),
    'ak4_phi0' : Bin("jetphi", r"Leading AK4 jet $\phi$", 50,-np.pi, np.pi),
    'ak4_phi1' : Bin("jetphi", r"Trailing AK4 jet $\phi$", 50,-np.pi, np.pi),
    'ak4_nef0' : Bin('frac', 'Leading Jet Neutral EM Frac', 50, 0, 1),
    'ak4_nef1' : Bin('frac', 'Trailing Jet Neutral EM Frac', 50, 0, 1),
    'ak4_nhf0' : Bin('frac', 'Leading Jet Neutral Hadronic Frac', 50, 0, 1),
    'ak4_nhf1' : Bin('frac', 'Trailing Jet Neutral Hadronic Frac', 50, 0, 1),
    'ak4_chf0' : Bin('frac', 'Leading Jet Charged Hadronic Frac', 50, 0, 1),
    'ak4_chf1' : Bin('frac', 'Trailing Jet Charged Hadronic Frac', 50, 0, 1),
    'ak4_central_eta' : Bin("jeteta", r"More Central Jet $\eta$", 50, -5, 5),
    'ak4_forward_eta' : Bin("jeteta", r"More Forward Jet $\eta$", 50, -5, 5),
    'extra_ak4_mult' : Bin("multiplicity", r"Additional AK4 Jet Multiplicity", 10, -0.5, 9.5),
    # 'dphitkpf' : Bin('dphi', r'$\Delta\phi_{TK,PF}$', 50, 0, 3.5),
    'met' : Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(0,500,50)) + list(range(500,1000,100)) + list(range(1000,2000,250))),
    'met_phi' : Bin("phi", r"$\phi_{MET}$", 50, -np.pi, np.pi),
    'calomet_pt' : Bin('met',r'$p_{T,CALO}^{miss,no-\ell}$ (GeV)',list(range(0,500,50)) + list(range(500,1000,100)) + list(range(1000,2000,250))),
    'calomet_phi' : Bin('phi',r'$\phi_{MET}^{CALO}$', 50, -np.pi, np.pi),
    'ak4_mult' : Bin("multiplicity", r"AK4 multiplicity", 10, -0.5, 9.5),
    'electron_pt' : hist.Bin('pt',r'Electron $p_{T}$ (GeV)',list(range(0,600,20))),
    'electron_pt0' : hist.Bin('pt',r'Leading electron $p_{T}$ (GeV)',list(range(0,600,20))),
    'electron_pt1' : hist.Bin('pt',r'Trailing electron $p_{T}$ (GeV)',list(range(0,600,20))),
    'electron_mt' : hist.Bin('mt',r'Electron $M_{T}$ (GeV)',list(range(0,800,20))),
    'muon_pt' : hist.Bin('pt',r'Muon $p_{T}$ (GeV)',list(range(0,600,20))),
    'muon_pt0' : hist.Bin('pt',r'Leading muon $p_{T}$ (GeV)',list(range(0,600,20))),
    'muon_pt1' : hist.Bin('pt',r'Trailing muon $p_{T}$ (GeV)',list(range(0,600,20))),
    'muon_mt' : hist.Bin('mt',r'Muon $M_{T}$ (GeV)',list(range(0,800,20))),
    'photon_pt0' : hist.Bin('pt',r'Photon $p_{T}$ (GeV)',list(range(200,600,20)) + list(range(600,1000,20)) ),
    'recoil' : hist.Bin('recoil','Recoil (GeV)', recoil_bins_2016),
    'dphijr' : Bin("dphi", r"min $\Delta\phi(j,recoil)$", 50, 0, 3.5),
    'dimuon_mass' : hist.Bin('dilepton_mass',r'M($\mu^{+}\mu^{-}$)',30,60,120),
    'dielectron_mass' : hist.Bin('dilepton_mass',r'M($e^{+}e^{-}$)',30,60,120),
    'mjj_transformed' : hist.Bin('transformed', r'Rescaled $M_{jj}$', 50, -5, 5),
    'detajj_transformed' : hist.Bin('transformed', r'Rescaled $\Delta\eta_{jj}$', 50, -5, 5),
    'dphijj_transformed' : hist.Bin('transformed', r'Rescaled $\Delta\phi_{jj}$', 50, -5, 5),
}

ylims = {
    'ak4_eta0' : (1e-3,1e8),
    'ak4_eta1' : (1e-3,1e8),
    'ak4_nef0' : (1e0,1e8),
    'ak4_nef1' : (1e0,1e8),
    'ak4_nhf0' : (1e0,1e8),
    'ak4_nhf1' : (1e0,1e8),
    'ak4_chf0' : (1e0,1e8),
    'ak4_chf1' : (1e0,1e8),
    'vecb' : (1e-1,1e9),
    'vecdphi' : (1e0,1e9),
    'dphitkpf' : (1e0,1e9),
    'met' : (1e-3,1e5),
    'ak4_mult' : (1e-1,1e8),
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

legend_labels_IC = {
    'DY.*' : "$Z(\\ell\\ell)$ + jets (strong)",
    'EWKZ.*ZToLL.*' : "$Z(\\ell\\ell)$ + jets (VBF)",
    'WN*J.*LNu.*' : "$W(\\ell\\nu)$ + jets (strong)",
    'EWKW.*LNu.*' : "$W(\\ell\\nu)$ + jets (VBF)",
    'ZN*JetsToNuNu.*.*' : "$Z(\\nu\\nu)$ + jets (strong)",
    'EWKZ.*ZToNuNu.*' : "$Z(\\nu\\nu)$ + jets (VBF)",
    'QCD.*' : "QCD Estimation",
    'Top.*' : "Top quark",
    'Diboson.*' : "Dibosons",
    'MET|Single(Electron|Photon|Muon)|EGamma.*' : "Data",
    'VBF_HToInv.*' : "VBF, $B(H\\rightarrow inv)=1.0$",
}

legend_titles = {
    'sr_vbf' : 'VBF Signal Region',
    'cr_1m_vbf' : r'VBF $1\mu$ Region',
    'cr_2m_vbf' : r'VBF $2\mu$ Region',
    'cr_1e_vbf' : r'VBF $1e$ Region',
    'cr_2e_vbf' : r'VBF $2e$ Region',
    'cr_g_vbf' : r'VBF $\gamma$ Region',
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

colors_IC = {
    'ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*' : (122, 189, 255),
    'EWKZ2Jets.*ZToNuNu.*' : (186, 242, 255),
    'DYJetsToLL_Pt.*FXFX.*' : (71, 191, 57),
    'EWKZ2Jets.*ZToLL.*' : (193, 255, 189),
    'WJetsToLNu_Pt.*FXFX.*' : (255, 182, 23),
    'EWKW2Jets.*WToLNu.*' : (252, 228, 159),
    'Diboson.*' : (0, 128, 128),
    'Top.*' : (148, 147, 146),
    '.*HF (N|n)oise.*' : (174, 126, 230),
}

def plot_data_mc(acc, outtag, year, data, mc, data_region, mc_region, distribution='mjj', plot_signal=True, mcscale=1, binwnorm=None, fformats=['pdf'], qcd_file=None, jes_file=None, ulxs=True):
    """
    Main plotter function to create a stack plot of data to background estimation (from MC).
    """
    acc.load(distribution)
    h = acc[distribution]

    # Set up overflow bin for mjj
    overflow = 'none'
    if distribution == 'mjj':
        overflow = 'over'

    # Pre-processing of the histogram, merging datasets, scaling w.r.t. XS and lumi
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h, ulxs=ulxs, mcscale=mcscale)
    h = merge_datasets(h)

    # Rebin the histogram if necessary
    if distribution in binnings.keys():
        new_ax = binnings[distribution]
        h = h.rebin(new_ax.name, new_ax)

    # Specifically rebin dphitkpf distribution: Merge the bins in the tails
    # Annoying approach but it works (due to float precision problems)
    elif distribution == 'dphitkpf':
        new_bins = [ibin.lo for ibin in h.identifiers('dphi') if ibin.lo < 2] + [3.5]
        
        new_ax = hist.Bin('dphi', r'$\Delta\phi_{TK,PF}$', new_bins)
        h = h.rebin('dphi', new_ax)

    # This sorting messes up in SR for some reason
    if data_region != 'sr_vbf':
        h.axis('dataset').sorting = 'integral'

    h_data = h.integrate('region', data_region)
    h_mc = h.integrate('region', mc_region)
    
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

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

    # Build the MC stack
    datasets = list(map(str, h[mc].identifiers('dataset')))

    plot_info = {
        'label' : datasets,
        'sumw' : [],
    }

    for dataset in datasets:
        sumw = h_mc.integrate('dataset', dataset).values(overflow=overflow)[()]

        plot_info['sumw'].append(sumw)

    # Add the HF-noise contribution (for signal region only)
    if data_region == 'sr_vbf':
        plot_info['label'].insert(6, 'HF Noise Estimate')
        plot_info['sumw'].insert(6, h_qcd.values * mcscale)

    fig, ax, rax = fig_ratio()
    
    # Plot data
    hist.plot1d(h_data[data], ax=ax, overflow=overflow, overlay='dataset', binwnorm=binwnorm, error_opts=data_err_opts)

    xedges = h_data.integrate('dataset').axes()[0].edges(overflow=overflow)

    # Plot MC stack
    hep.histplot(plot_info['sumw'], xedges, 
        ax=ax,
        label=plot_info['label'], 
        histtype='fill',
        binwnorm=binwnorm,
        stack=True
        )

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

    ax.set_yscale('log')
    if distribution == 'mjj':
        ax.set_ylim(1e-3,1e5)
        ax.set_ylabel('Events / GeV')
    elif distribution == 'cnn_score':
        if data_region in ['cr_2m_vbf', 'cr_2e_vbf']:
            ax.set_ylim(1e-2,1e5)
        else:
            ax.set_ylim(1e-1,1e6)
    else:
        if distribution in ylims.keys():
            ax.set_ylim(ylims[distribution])
        else:
            ax.set_ylim(1e0,1e4)
        ax.set_ylabel('Events')
    
    if distribution == 'mjj':
        ax.set_xlim(left=0.)

    ax.yaxis.set_ticks_position('both')

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

    # Plot ratio
    h_data = h_data.integrate('dataset', data)
    h_mc = h_mc.integrate('dataset', mc)

    sumw_data, sumw2_data = h_data.values(overflow=overflow, sumw2=True)[()]
    sumw_mc = h_mc.values(overflow=overflow)[()]
    
    # Add the HF-noise contribution to the background expectation
    if data_region == 'sr_vbf':
        sumw_mc = sumw_mc + h_qcd.values * mcscale

    r = sumw_data / sumw_mc
    rerr = np.abs(poisson_interval(r, sumw2_data / sumw_mc**2) - r)

    r[np.isnan(r) | np.isinf(r)] = 0.
    rerr[np.isnan(rerr) | np.isinf(rerr)] = 0.

    hep.histplot(
        r,
        xedges,
        yerr=rerr,
        ax=rax,
        histtype='errorbar',
        **data_err_opts
    )

    xlabels = {
        'mjj': r'$M_{jj} \ (GeV)$',
        'ak4_eta0': r'Leading Jet $\eta$',
        'ak4_eta1': r'Trailing Jet $\eta$',
    }

    if distribution in xlabels.keys():
        ax.set_xlabel(xlabels[distribution])
        rax.set_xlabel(xlabels[distribution])
    
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

    rax.grid(axis='y',which='both',linestyle='--')

    rax.axhline(1., xmin=0, xmax=1, color=(0,0,0,0.4), ls='--')

    fig.text(0., 1., '$\\bf{CMS}$ internal',
                fontsize=14,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes
               )

    fig.text(1., 1., f'VBF, {lumi(year, mcscale):.1f} fb$^{{-1}}$ ({year})',
                fontsize=14,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
               )

    outdir = f'./output/{outtag}/{data_region}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # For each file format (PDF, PNG etc.), save the plot
    for fformat in fformats:
        outpath = pjoin(outdir, f'{data_region}_data_mc_{distribution}_{year}.{fformat}')
        fig.savefig(outpath)

    plt.close(fig)
