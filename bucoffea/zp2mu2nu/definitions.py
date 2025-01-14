import copy
import re
from coffea import hist

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from awkward import JaggedArray
import numpy as np
from bucoffea.helpers import object_overlap, sigmoid3, dphi
from bucoffea.helpers.dataset import extract_year
from bucoffea.helpers.paths import bucoffea_path
from bucoffea.helpers.gen import find_first_parent
from bucoffea.monojet.definitions import accu_int, defaultdict_accumulator_of_empty_column_accumulator_float16, defaultdict_accumulator_of_empty_column_accumulator_int64,defaultdict_accumulator_of_empty_column_accumulator_bool
from pprint import pprint

def zp2mu2nu_accumulator(cfg):
    dataset_ax = Cat("dataset", "Primary dataset")
    unc_ax = Cat("uncertainty", "Uncertainty weight variation")
    variation_ax = Cat("variation", "Uncertainty weight variation")
    region_ax = Cat("region", "Selection region")
    type_ax = Cat("type", "Type")

    met_ax = Bin("met", r"$p_{T}^{miss}$ (GeV)", 300, 0, 300)
    recoil_ax = Bin("recoil", r"Recoil (GeV)", 200, 0, 200)

    pt_ax = Bin("pt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    pt_ax_mu = Bin("pt", r"$p_{T}$ (GeV)", 100, 0, 100)
    pt_ax_el = Bin("pt", r"$p_{T}$ (GeV)", [10,20,35,50,100,200,500])
    pt_ax_tau = Bin("pt", r"$p_{T}$ (GeV)", [18,20,25,30,35,40,500,1000])

    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 100, 0, 4000)
    mt_ax = Bin("mt", r"$M_{T}$ (GeV)", 250, 0, 250)
    eta_ax_mu = Bin("eta", r"$\eta$", 48, -2.4, 2.4)
    eta_ax_el = Bin("eta", r"$\eta$", [-2.5, -2.0, -1.56, -1.44, -0.8, 0, 0.8, 1.44,1.56,2.0,2.5])
    abseta_ax_mu = Bin("abseta", r"$|\eta|$", [0,0.9,1.2,2.1,2.4])

    eta_ax_coarse = Bin("eta", r"$\eta$", 25, -5, 5)
    phi_ax_mu = Bin("phi", r"$\phi$", 50,-np.pi, np.pi)
    phi_ax_coarse = Bin("phi", r"$\phi$", 20,-np.pi, np.pi)

    dilepton_mass_ax = Bin("dilepton_mass", r"$M(\ell\ell)$ (GeV)", 100,0,200)

    weight_type_ax = Cat("weight_type", "Weight type")
    weight_ax = Bin("weight_value", "Weight",100,0.5,1.5)
    weight_wide_ax = Bin("weight_value", "Weight",100,-10,10)

    nvtx_ax = Bin('nvtx','Number of vertices',100,-0.5,99.5)
    rho_ax = Bin('rho','Energy density',100, 0, 100)
    frac_ax = Bin('frac','Fraction', 50, 0, 1)

    Hist = hist.Hist

    items = {}

    items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["met_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax_mu)

    multiplicity_ax = Bin("multiplicity", r"multiplicity", 10, -0.5, 9.5)
    # Multiplicity histograms
    for cand in ['ak4', 'ak8', 'bjet', 'loose_ele', 'loose_muo', 'tight_ele', 'tight_muo', 'tau', 'photon','hlt_single_muon','muons_hltmatch']:
        items[f"{cand}_mult"] = Hist(cand, dataset_ax, region_ax, multiplicity_ax)

    #items["extra_ak4_mult"] = Hist(cand, dataset_ax, region_ax, multiplicity_ax)

    items["mu1_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax_mu)
    items["mu1_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax_mu)
    items["mu1_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax_mu)

    items["mu2_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax_mu)
    items["mu2_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax_mu)
    items["mu2_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax_mu)

    items["dimuon_mass"] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)
    items["dimuon_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax_mu)
    items["dimuon_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax_mu)
    items["dimuon_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax_mu)

    items["mt_mu1mu2"] = Hist("Counts", dataset_ax, region_ax, mt_ax)
    items["mt_mu1met"] = Hist("Counts", dataset_ax, region_ax, mt_ax)
    items["mt_mu2met"] = Hist("Counts", dataset_ax, region_ax, mt_ax)
    items["mt_dimuonmet"] = Hist("Counts", dataset_ax, region_ax, mt_ax)

#    items["electron_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
#    items["electron_pt_eta"] = Hist("Counts", dataset_ax, region_ax, pt_ax_el, eta_ax_el)
#    items["electron_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
#    items["electron_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
#    items["electron_pt0"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
#    items["electron_eta0"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
#    items["electron_phi0"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
#    items["electron_pt1"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
#    items["electron_eta1"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
#    items["electron_phi1"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
#    items["electron_mt"] = Hist("Counts", dataset_ax, region_ax, mt_ax)

#    items["dielectron_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
#    items["dielectron_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
#    items["dielectron_mass"] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)

#    items['photon_pt0'] = Hist("Counts", dataset_ax, region_ax, pt_ax)
#    items['photon_eta0'] = Hist("Counts", dataset_ax, region_ax, eta_ax)
#    items['photon_phi0'] = Hist("Counts", dataset_ax, region_ax, phi_ax)

#    items["tau_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax_tau)

    # One cutflow counter per region
    regions = zp2mu2nu_regions(cfg).keys()
    for region in regions:
        if region=="inclusive":
            continue
        items[f'cutflow_{region}']  = processor.defaultdict_accumulator(accu_int)

    items['nevents'] = processor.defaultdict_accumulator(float)
    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)
    items['sumw_pileup'] = processor.defaultdict_accumulator(float)

    items['selected_events'] = processor.defaultdict_accumulator(list)
    items['kinematics'] = processor.defaultdict_accumulator(list)

    items['weights'] = Hist("Weights", dataset_ax, region_ax, weight_type_ax, weight_ax)
    items['weights_wide'] = Hist("Weights", dataset_ax, region_ax, weight_type_ax, weight_wide_ax)
    items['npv'] = Hist('Number of primary vertices', dataset_ax, region_ax, nvtx_ax)
    items['npvgood'] = Hist('Number of good primary vertices', dataset_ax, region_ax, nvtx_ax)
    items['npv_nopu'] = Hist('Number of primary vertices (No PU weights)', dataset_ax, region_ax, nvtx_ax)
    items['npvgood_nopu'] = Hist('Number of good primary vertices (No PU weights)', dataset_ax, region_ax, nvtx_ax)

    items['rho_all'] = Hist(r'$\rho$ for all PF candidates', dataset_ax, region_ax, rho_ax)
    items['rho_central'] = Hist(r'$\rho$ for central PF candidates', dataset_ax, region_ax, rho_ax)
    items['rho_all_nopu'] = Hist(r'$\rho$ for all PF candidates (No PU weights)', dataset_ax, region_ax, rho_ax)
    items['rho_central_nopu'] = Hist(r'$\rho$ for central PF candidates (No PU weights)', dataset_ax, region_ax, rho_ax)

    items['tree_float16'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_float16)
    items['tree_int64'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_int64)
    items['tree_bool'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_bool)
    return  processor.dict_accumulator(items)

def zp2mu2nu_regions(cfg):
    common_cuts = [
#        'veto_ele',
#        'veto_muo',
        'filt_met',
#        'mindphijr',
#        'recoil',
#        'two_jets',
#        'leadak4_pt_eta',
#        'leadak4_id',
#        'trailak4_pt_eta',
#        'trailak4_id',
#        'hemisphere',
#        'mjj',
#        'dphijj',
#        'detajj',
#        'veto_photon',
#        'veto_tau',
#        'veto_b',
#        'leadak4_clean'
    ]

    regions = {}
    # regions['inclusive'] = ['inclusive']

    # Dimuon region
    #cr_2m_cuts = ['trig_mu','two_muons', 'at_least_one_tight_mu', 'dimuon_mass', 'veto_ele', 'dimuon_charge'] + common_cuts + ['dpfcalo_cr']
    cr_2m_cuts = ['trig_mu','two_muons', 'two_tight_muons', 'mu_pt_trig_safe', 'dimuon_mass', 'dimuon_charge'] + common_cuts 
    regions['cr_2m_zp2mu2nu'] = cr_2m_cuts

    ## Single muon CR
    #cr_1m_cuts = ['trig_met','one_muon', 'at_least_one_tight_mu',  'veto_ele'] + common_cuts[1:] + ['dpfcalo_cr']
    #cr_1m_cuts.remove('veto_muo')
    #regions['cr_1m_vbf'] = cr_1m_cuts

    ## Dielectron CR
    #cr_2e_cuts = ['trig_ele','two_electrons', 'at_least_one_tight_el', 'dielectron_mass', 'veto_muo', 'dielectron_charge'] + common_cuts[2:] + ['dpfcalo_cr']
    ## cr_2e_cuts.remove('veto_ele')
    #regions['cr_2e_vbf'] = cr_2e_cuts

    ## Single electron CR
    #cr_1e_cuts = ['trig_ele','one_electron', 'at_least_one_tight_el', 'veto_muo','met_el'] + common_cuts[1:] + ['dpfcalo_cr', 'no_el_in_hem']
    ## cr_1e_cuts.remove('veto_ele')
    #regions['cr_1e_vbf'] =  cr_1e_cuts

    ## Photon CR
    #cr_g_cuts = ['trig_photon', 'one_photon', 'at_least_one_tight_photon','photon_pt'] + common_cuts + ['dpfcalo_cr']
    #cr_g_cuts.remove('veto_photon')
    #regions['cr_g_vbf'] = cr_g_cuts

    return regions


def met_xy_correction(df, met_pt, met_phi):
    '''Apply MET XY corrections (UL based).'''
    import yaml

    correction_src_file = bucoffea_path('data/met/metxycorr.yaml')

    with open(correction_src_file) as f:
        xycorr = yaml.load(f.read(),Loader=yaml.SafeLoader)

    npv = df['PV_npvs']

    met_px = met_pt * np.cos(met_phi)
    met_py = met_pt * np.sin(met_phi)

    def correction(a,b):
        return -(a * npv + b)

    dataset = df['dataset']
    year = extract_year(df['dataset'])

    # Get the correction factors, depending on the run (if data)
    if df['is_data']:
        # Extract the run information from the dataset name
        run = re.findall('201\d([A-F])', dataset)[0]
        # Get the corrections for this run
        xycorrections = xycorr[year][run]

    else:
        # No run info needed, just extract the corrections for MC, based on year
        xycorrections = xycorr[year]['MC']

    # Extract the coefficients for the X and Y corrections
    xcorr_coef = ( xycorrections['X']['a'], xycorrections['X']['b'] )
    ycorr_coef = ( xycorrections['Y']['a'], xycorrections['Y']['b'] )

    met_xcorr = correction(*xcorr_coef)
    met_ycorr = correction(*ycorr_coef)

    corr_met_px = met_px + met_xcorr
    corr_met_py = met_py + met_ycorr

    corr_met_pt = np.hypot(corr_met_px, corr_met_py)
    corr_met_phi = np.arctan2(corr_met_py, corr_met_px)

    return corr_met_pt, corr_met_phi

def pileup_sf_variations(df, evaluator, cfg):
    if cfg.SF.PILEUP.MODE == 'nano':
        pu_weights = {
            "puSFUp" : df['puWeightUp'],
            "puSFDown" : df['puWeightDown'],
            "puSFNom" : df['puWeight'],
        }
    else:
        raise NotImplementedError(f'No implementation for cfg.PILEUP.MODE: {cfg.SF.PILEUP.MODE}')

    return pu_weights

def setup_candidates(df, cfg):
    if df['is_data']:
        # All data
        jes_suffix = '_nom'
        jes_suffix_met = '_nom'
    else:
        # MC, all years
        jes_suffix = '_nom'
        if cfg.MET.JER:
            jes_suffix_met = '_jer'
        else:
            jes_suffix_met = '_nom'

    muons = JaggedCandidateArray.candidatesfromcounts(
        df['nMuon'],
        pt=df['Muon_pt'],
        eta=df['Muon_eta'],
        abseta=np.abs(df['Muon_eta']),
        phi=df['Muon_phi'],
        mass=0 * df['Muon_pt'],
        charge=df['Muon_charge'],
        looseId=df['Muon_looseId'],
        iso=df["Muon_pfRelIso04_all"],
        tightId=df['Muon_tightId'],
        dxy=df['Muon_dxy'],
        dz=df['Muon_dz']
    )

    # For MC, add the matched gen-particle info for checking
    if not df['is_data']:
        kwargs = {'genpartflav' : df['Muon_genPartFlav']}
        muons.add_attributes(**kwargs) 

    # All muons must be at least loose
    muons = muons[muons.looseId \
                    & (muons.iso < cfg.MUON.CUTS.LOOSE.ISO) \
                    & (muons.pt > cfg.MUON.CUTS.LOOSE.PT) \
                    & (muons.abseta<cfg.MUON.CUTS.LOOSE.ETA) \
                    ]

    #electrons = JaggedCandidateArray.candidatesfromcounts(
    #    df['nElectron'],
    #    pt=df['Electron_pt'],
    #    eta=df['Electron_eta'],
    #    abseta=np.abs(df['Electron_eta']),
    #    etasc=df['Electron_eta']+df['Electron_deltaEtaSC'],
    #    absetasc=np.abs(df['Electron_eta']+df['Electron_deltaEtaSC']),
    #    phi=df['Electron_phi'],
    #    mass=0 * df['Electron_pt'],
    #    charge=df['Electron_charge'],
    #    looseId=(df[cfg.ELECTRON.BRANCH.ID]>=1),
    #    tightId=(df[cfg.ELECTRON.BRANCH.ID]==4),
    #    dxy=np.abs(df['Electron_dxy']),
    #    dz=np.abs(df['Electron_dz']),
    #    barrel=np.abs(df['Electron_eta']+df['Electron_deltaEtaSC']) <= 1.4442
    #)

    ## For MC, add the matched gen-particle info for checking
    #if not df['is_data']:
    #    kwargs = {'genpartflav' : df['Electron_genPartFlav']}
    #    electrons.add_attributes(**kwargs) 

    # All electrons must be at least loose
    #pass_dxy = (electrons.barrel & (np.abs(electrons.dxy) < cfg.ELECTRON.CUTS.LOOSE.DXY.BARREL)) \
    #| (~electrons.barrel & (np.abs(electrons.dxy) < cfg.ELECTRON.CUTS.LOOSE.DXY.ENDCAP))

    #pass_dz = (electrons.barrel & (np.abs(electrons.dz) < cfg.ELECTRON.CUTS.LOOSE.DZ.BARREL)) \
    #| (~electrons.barrel & (np.abs(electrons.dz) < cfg.ELECTRON.CUTS.LOOSE.DZ.ENDCAP))

    #electrons = electrons[electrons.looseId \
    #                                & (electrons.pt>cfg.ELECTRON.CUTS.LOOSE.PT) \
    #                                & (electrons.absetasc<cfg.ELECTRON.CUTS.LOOSE.ETA) \
    #                                & pass_dxy \
    #                                & pass_dz
    #                                ]

    #if cfg.OVERLAP.ELECTRON.MUON.CLEAN:
    #    electrons = electrons[object_overlap(electrons, muons, dr=cfg.OVERLAP.ELECTRON.MUON.DR)]

    #taus = JaggedCandidateArray.candidatesfromcounts(
    #    df['nTau'],
    #    pt=df['Tau_pt'],
    #    eta=df['Tau_eta'],
    #    abseta=np.abs(df['Tau_eta']),
    #    phi=df['Tau_phi'],
    #    mass=0 * df['Tau_pt'],
    #    decaymode=df[cfg.TAU.BRANCH.ID],
    #    iso=df[cfg.TAU.BRANCH.ISO]
    #)

    # For MC, add the matched gen-particle info for checking
    #if not df['is_data']:
    #    kwargs = {'genpartflav' : df['Tau_genPartFlav']}
    #    taus.add_attributes(**kwargs) 

    #taus = taus[ (taus.decaymode) \
    #            & (taus.pt > cfg.TAU.CUTS.PT)\
    #            & (taus.abseta < cfg.TAU.CUTS.ETA) \
    #            & ((taus.iso&2)==2)]

    #if cfg.OVERLAP.TAU.MUON.CLEAN:
    #    taus = taus[object_overlap(taus, muons, dr=cfg.OVERLAP.TAU.MUON.DR)]
    #if cfg.OVERLAP.TAU.ELECTRON.CLEAN:
    #    taus = taus[object_overlap(taus, electrons, dr=cfg.OVERLAP.TAU.ELECTRON.DR)]

    # choose the right branch name for photon ID bitmap depending on the actual name in the file (different between nano v5 and v7)
    if cfg.PHOTON.BRANCH.ID in df.keys():
        PHOTON_BRANCH_ID = cfg.PHOTON.BRANCH.ID
    else:
        PHOTON_BRANCH_ID = cfg.PHOTON.BRANCH.IDV7
    photons = JaggedCandidateArray.candidatesfromcounts(
        df['nPhoton'],
        pt=df['Photon_pt'],
        eta=df['Photon_eta'],
        abseta=np.abs(df['Photon_eta']),
        phi=df['Photon_phi'],
        mass=0*df['Photon_pt'],
        looseId= (df[PHOTON_BRANCH_ID]>=1) & df['Photon_electronVeto'],
        mediumId=(df[PHOTON_BRANCH_ID]>=2) & df['Photon_electronVeto'],
        r9=df['Photon_r9'],
        barrel=df['Photon_isScEtaEB'],
    )
    photons = photons[photons.looseId \
              & (photons.pt > cfg.PHOTON.CUTS.LOOSE.pt) \
              & (photons.abseta < cfg.PHOTON.CUTS.LOOSE.eta)
              ]

    if cfg.OVERLAP.PHOTON.MUON.CLEAN:
        photons = photons[object_overlap(photons, muons, dr=cfg.OVERLAP.PHOTON.MUON.DR)]
    #if cfg.OVERLAP.PHOTON.ELECTRON.CLEAN:
    #    photons = photons[object_overlap(photons, electrons, dr=cfg.OVERLAP.PHOTON.ELECTRON.DR)]

    ak4 = JaggedCandidateArray.candidatesfromcounts(
        df['nJet'],
        pt=df[f'Jet_pt{jes_suffix}'] if (df['is_data'] or cfg.AK4.JER) else df[f'Jet_pt{jes_suffix}']/df['Jet_corr_JER'],
        ptnano=df['Jet_pt'],
        eta=df['Jet_eta'],
        abseta=np.abs(df['Jet_eta']),
        phi=df['Jet_phi'],
        mass=np.zeros_like(df['Jet_pt']),
        looseId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        tightId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        puid=((df['Jet_puId']&2>0) | ((df[f'Jet_pt{jes_suffix}'] if (df['is_data'] or cfg.AK4.JER) else df[f'Jet_pt{jes_suffix}']/df['Jet_corr_JER'])>50)), # medium pileup jet ID
        csvv2=df["Jet_btagCSVV2"],
        deepcsv=df['Jet_btagDeepB'],
        nef=df['Jet_neEmEF'],
        cef=df['Jet_chEmEF'],
        nhf=df['Jet_neHEF'],
        chf=df['Jet_chHEF'],
        ptraw=df['Jet_pt']*(1-df['Jet_rawFactor']),
        nconst=df['Jet_nConstituents'],
        hadflav= 0*df['Jet_pt'] if df['is_data'] else df['Jet_hadronFlavour'],
    )

    if not df['is_data']:
        ak4.add_attributes(jercorr=df['Jet_corr_JER'])

    # B jets have their own overlap cleaning,
    # so deal with them before applying filtering to jets
    btag_discriminator = getattr(ak4, cfg.BTAG.algo)
    btag_cut = cfg.BTAG.CUTS[cfg.BTAG.algo][cfg.BTAG.wp]
    bjets = ak4[
        (ak4.pt > cfg.BTAG.PT) \
        & (ak4.abseta < cfg.BTAG.ETA) \
        & (btag_discriminator > btag_cut)
    ]

    ak4 = ak4[ak4.looseId]

    if cfg.OVERLAP.BTAG.MUON.CLEAN:
        bjets = bjets[object_overlap(bjets, muons, dr=cfg.OVERLAP.BTAG.MUON.DR)]
    #if cfg.OVERLAP.BTAG.ELECTRON.CLEAN:
    #    bjets = bjets[object_overlap(bjets, electrons, dr=cfg.OVERLAP.BTAG.ELECTRON.DR)]
    #if cfg.OVERLAP.BTAG.PHOTON.CLEAN:
    #    bjets = bjets[object_overlap(bjets, photons, dr=cfg.OVERLAP.BTAG.PHOTON.DR)]

    if cfg.OVERLAP.AK4.MUON.CLEAN:
        ak4 = ak4[object_overlap(ak4, muons, dr=cfg.OVERLAP.AK4.MUON.DR)]
    #if cfg.OVERLAP.AK4.ELECTRON.CLEAN:
    #    ak4 = ak4[object_overlap(ak4, electrons, dr=cfg.OVERLAP.AK4.ELECTRON.DR)]
    if cfg.OVERLAP.AK4.PHOTON.CLEAN:
        ak4 = ak4[object_overlap(ak4, photons, dr=cfg.OVERLAP.AK4.PHOTON.DR)]

    # No EE v2 fix in UL
    if cfg.RUN.ULEGACYV8:
        met_branch = 'MET'
    else:
        if extract_year(df['dataset']) == 2017:
            met_branch = 'METFixEE2017'
        else:
            met_branch = 'MET'

    met_pt = df[f'{met_branch}_pt{jes_suffix_met}']
    met_phi = df[f'{met_branch}_phi{jes_suffix_met}']

#    return met_pt, met_phi, ak4, bjets, muons, electrons, taus, photons, jet_images
    return met_pt, met_phi, ak4, bjets, muons, photons


def gen_match_check_leptons(leptons, weights):
    '''
    Return the updated lepton weight after checking if the leptons in the event are matched to a proper GEN-level lepton.
    '''
    # For muons and electrons, we require the gen particle flavor to be non-zero 
    gen_match_ok = leptons.genpartflav > 0
    weights[~gen_match_ok] = 1.

    return weights

#def candidate_weights(weights, df, evaluator, muons, electrons, photons, cfg):
def candidate_weights(weights, df, evaluator, muons, photons, cfg):
    year = extract_year(df['dataset'])
    # Muon ID and Isolation for tight and loose WP
    # Function of pT, eta (Order!)
    weight_muons_id_tight = evaluator['muon_id_tight'](muons[df['is_tight_muon']].abseta, muons[df['is_tight_muon']].pt)
    weight_muons_iso_tight = evaluator['muon_iso_tight'](muons[df['is_tight_muon']].abseta, muons[df['is_tight_muon']].pt)

    if cfg.SF.DIMUO_ID_SF.USE_AVERAGE:
        tight_dimuons = muons[df["is_tight_muon"]].distincts()
        t0 = (evaluator['muon_id_tight'](tight_dimuons.i0.pt, tight_dimuons.i0.abseta) \
             * evaluator['muon_iso_tight'](tight_dimuons.i0.pt, tight_dimuons.i0.abseta)).prod()
        t1 = (evaluator['muon_id_tight'](tight_dimuons.i1.pt, tight_dimuons.i1.abseta) \
             * evaluator['muon_iso_tight'](tight_dimuons.i1.pt, tight_dimuons.i1.abseta)).prod()
        l0 = (evaluator['muon_id_loose'](tight_dimuons.i0.pt, tight_dimuons.i0.abseta) \
             * evaluator['muon_iso_loose'](tight_dimuons.i0.pt, tight_dimuons.i0.abseta)).prod()
        l1 = (evaluator['muon_id_loose'](tight_dimuons.i1.pt, tight_dimuons.i1.abseta) \
             * evaluator['muon_iso_loose'](tight_dimuons.i1.pt, tight_dimuons.i1.abseta)).prod()
        weights_2m_tight = 0.5*( l0 * t1 + l1 * t0)
        weights.add("muon_id_iso_tight", weight_muons_id_tight*weight_muons_iso_tight*(tight_dimuons.counts!=1) + weights_2m_tight*(tight_dimuons.counts==1))
    else:
        w_muon_id_iso_tight = weight_muons_id_tight*weight_muons_iso_tight
        if cfg.MUON.GENCHECK:
            w_muon_id_iso_tight = gen_match_check_leptons(muons[df['is_tight_muon']], w_muon_id_iso_tight).prod()
        else:
            w_muon_id_iso_tight = w_muon_id_iso_tight.prod()
        
        weights.add("muon_id_iso_tight", w_muon_id_iso_tight )

    w_muon_id_loose = evaluator['muon_id_loose'](muons[~df['is_tight_muon']].abseta, muons[~df['is_tight_muon']].pt)
    w_muon_iso_loose = evaluator['muon_iso_loose'](muons[~df['is_tight_muon']].abseta, muons[~df['is_tight_muon']].pt)
    
    if cfg.MUON.GENCHECK:
        w_muon_id_loose = gen_match_check_leptons(muons[~df['is_tight_muon']], w_muon_id_loose).prod()
        w_muon_iso_loose = gen_match_check_leptons(muons[~df['is_tight_muon']], w_muon_iso_loose).prod()
    else:
        w_muon_id_loose = w_muon_id_loose.prod()
        w_muon_iso_loose = w_muon_iso_loose.prod()
    
    weights.add("muon_id_loose", w_muon_id_loose)
    weights.add("muon_iso_loose", w_muon_iso_loose)

#    # Electron ID and reco
#    # Function of eta, pT (Other way round relative to muons!)
#
#    # For 2017 and 2018 (both years in UL), the reco SF is split below/above 20 GeV
#    if cfg.RUN.ULEGACYV8 or extract_year(df['dataset']) == 2017:
#        high_et = electrons.pt>20
#        ele_reco_sf = evaluator['ele_reco'](electrons.etasc[high_et], electrons.pt[high_et])
#        ele_reco_sf_low_pt = evaluator['ele_reco_pt_lt_20'](electrons.etasc[~high_et], electrons.pt[~high_et])
#
#    else:
#        ele_reco_sf = evaluator['ele_reco'](electrons.etasc, electrons.pt)
#    
#    if cfg.ELECTRON.GENCHECK:
#        ele_reco_sf = gen_match_check_leptons(electrons[high_et], ele_reco_sf).prod()
#        ele_reco_sf_low_pt = gen_match_check_leptons(electrons[~high_et], ele_reco_sf_low_pt).prod()
#    else:
#        ele_reco_sf = ele_reco_sf.prod()
#        ele_reco_sf_low_pt = ele_reco_sf_low_pt.prod()
#    
#    weights.add("ele_reco", ele_reco_sf * ele_reco_sf_low_pt)
#    # ID/iso SF is not split
#    # in case of 2 tight electrons, we want to apply 0.5*(T1L2+T2L1) instead of T1T2
#    weights_electrons_tight = evaluator['ele_id_tight'](electrons[df['is_tight_electron']].etasc, electrons[df['is_tight_electron']].pt)
#    if cfg.SF.DIELE_ID_SF.USE_AVERAGE:
#        tight_dielectrons = electrons[df["is_tight_electron"]].distincts()
#        l0 = evaluator['ele_id_loose'](tight_dielectrons.i0.etasc, tight_dielectrons.i0.pt).prod()
#        t0 = evaluator['ele_id_tight'](tight_dielectrons.i0.etasc, tight_dielectrons.i0.pt).prod()
#        l1 = evaluator['ele_id_loose'](tight_dielectrons.i1.etasc, tight_dielectrons.i1.pt).prod()
#        t1 = evaluator['ele_id_tight'](tight_dielectrons.i1.etasc, tight_dielectrons.i1.pt).prod()
#        weights_2e_tight = 0.5*( l0 * t1 + l1 * t0)
#        weights.add("ele_id_tight", weights_electrons_tight*(tight_dielectrons.counts!=1) + weights_2e_tight*(tight_dielectrons.counts==1))
#    else:
#        if cfg.ELECTRON.GENCHECK:
#            weights_electrons_tight = gen_match_check_leptons(electrons[df['is_tight_electron']], weights_electrons_tight).prod()
#        else:
#            weights_electrons_tight = weights_electrons_tight.prod()
#        
#        weights.add("ele_id_tight", weights_electrons_tight)
#    
#    weights_electrons_loose = evaluator['ele_id_loose'](electrons[~df['is_tight_electron']].etasc, electrons[~df['is_tight_electron']].pt)
#    if cfg.ELECTRON.GENCHECK:
#        weights_electrons_loose = gen_match_check_leptons(electrons[~df['is_tight_electron']], weights_electrons_loose).prod()
#    else:
#        weights_electrons_loose = weights_electrons_loose.prod()
#    
#    weights.add("ele_id_loose", weights_electrons_loose)

    # Photon ID and electron veto
    if cfg.SF.PHOTON.USETNP:
        weights.add("photon_id_tight", evaluator['photon_id_tight_tnp'](np.abs(photons[df['is_tight_photon']].eta)).prod())
    else:
        weights.add("photon_id_tight", evaluator['photon_id_tight'](photons[df['is_tight_photon']].eta, photons[df['is_tight_photon']].pt).prod())

    if year == 2016:
        csev_weight = evaluator["photon_csev"](photons.abseta, photons.pt).prod()
    elif year == 2017:
        csev_sf_index = 0.5 * photons.barrel + 3.5 * ~photons.barrel + 1 * (photons.r9 > 0.94) + 2 * (photons.r9 <= 0.94)
        csev_weight = evaluator['photon_csev'](csev_sf_index).prod()
    elif year == 2018:
        csev_weight = evaluator['photon_csev'](photons.pt, photons.abseta).prod()
    csev_weight[csev_weight==0] = 1
    weights.add("photon_csev", csev_weight)

    return weights
