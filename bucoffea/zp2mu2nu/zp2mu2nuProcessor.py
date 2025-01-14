import copy
import coffea.processor as processor
import re
import numpy as np
import pandas as pd
from dynaconf import settings as cfg

from bucoffea.helpers.tensorflow import (
                            load_model, 
                            prepare_data_for_cnn
                            )

from bucoffea.helpers.pytorch import (
    load_pytorch_state_dict,
    scale_features_for_dnn,
    FullyConnectedNN,
)

from bucoffea.helpers import (
                              bucoffea_path,
                              dphi,
                              evaluator_from_config,
                              mask_and,
                              mask_or,
                              min_dphi_jet_met,
                              mt,
                              recoil,
                              weight_shape,
                              candidates_in_hem,
                              electrons_in_hem,
                              calculate_vecB,
                              calculate_vecDPhi
                              )
from bucoffea.helpers.dataset import (
                                      extract_year,
                                      is_data,
                                      is_lo_g,
                                      is_lo_w,
                                      is_lo_z,
                                      is_lo_g_ewk,
                                      is_lo_w_ewk,
                                      is_lo_z_ewk,
                                      is_nlo_w,
                                      is_nlo_z,
                                      is_lo_znunu
                                      )
from bucoffea.helpers.gen import (
                                  setup_gen_candidates,
                                  setup_dressed_gen_candidates,
                                  setup_lhe_cleaned_genjets,
                                  fill_gen_v_info
                                 )
from bucoffea.helpers.weights import (
                                  get_veto_weights,
                                  btag_weights,
                                  get_varied_ele_sf,
                                  get_varied_muon_sf
                                 )
from bucoffea.monojet.definitions import (
#                                          candidate_weights,
                                          pileup_weights,
#                                          setup_candidates, 
                                          theory_weights_vbf,
                                          photon_trigger_sf,
                                          photon_impurity_weights,
                                          data_driven_qcd_dataset,
                                          get_nlo_ewk_weights,
                                          )

from bucoffea.vbfhinv.definitions import (
#                                           vbfhinv_accumulator,
#                                           vbfhinv_regions,
                                           ak4_em_frac_weights,
                                           met_trigger_sf,
                                           apply_hfmask_weights,
                                           apply_hf_weights_for_qcd_estimation,
                                           apply_endcap_weights,
                                           hfmask_sf,
                                           met_xy_correction,
                                           pileup_sf_variations
                                         )

from bucoffea.zp2mu2nu.definitions import (
                                           zp2mu2nu_accumulator,
                                           zp2mu2nu_regions,
                                           setup_candidates,
                                           candidate_weights
                                         )

def trigger_selection(selection, df, cfg):
    pass_all = np.zeros(df.size) == 0
    pass_none = ~pass_all
    dataset = df['dataset']

    if df['is_data']:
        selection.add('filt_met', mask_and(df, cfg.FILTERS.DATA))
    else:
        selection.add('filt_met', mask_and(df, cfg.FILTERS.MC))
    selection.add('trig_met', mask_or(df, cfg.TRIGGERS.MET))

    # Electron trigger overlap
    if df['is_data']:
        if "SinglePhoton" in dataset:
            # Backup photon trigger, but not main electron trigger
            trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE_BACKUP) & (~mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE))
        elif "SingleElectron" in dataset:
            # Main electron trigger, no check for backup
            trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE)
        elif "EGamma" in dataset:
            # 2018 has everything in one stream, so simple OR
            trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE_BACKUP) | mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE)
        else:
            trig_ele = pass_none
    else:
        trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE_BACKUP) | mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE)

    selection.add('trig_ele', trig_ele)

    # Photon trigger:
    if (not df['is_data']) or ('SinglePhoton' in dataset) or ('EGamma' in dataset):
        trig_photon = mask_or(df, cfg.TRIGGERS.PHOTON.SINGLE)
    else:
        trig_photon = pass_none
    selection.add('trig_photon', trig_photon)

    # Muon trigger
    selection.add('trig_mu', mask_or(df, cfg.TRIGGERS.MUON.SINGLE))

    return selection

class zp2mu2nuProcessor(processor.ProcessorABC):
    def __init__(self, blind=False):
        self._year=None
        self._blind=blind
        self._configure()
        self._accumulator = zp2mu2nu_accumulator(cfg)

    @property
    def accumulator(self):
        return self._accumulator

    def _configure(self, df=None):
        cfg.DYNACONF_WORKS="merge_configs"
        cfg.MERGE_ENABLED_FOR_DYNACONF=True
        cfg.SETTINGS_FILE_FOR_DYNACONF = bucoffea_path("config/zp2mu2nu.yaml")

        # Reload config based on year
        if df:
            dataset = df['dataset']
            self._year = extract_year(dataset)
            df["year"] = self._year
            cfg.ENV_FOR_DYNACONF = f"era{self._year}"
        else:
            cfg.ENV_FOR_DYNACONF = f"default"
        cfg.reload()

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        self._configure(df)
        dataset = df['dataset']
        df['is_data'] = is_data(dataset)

        # Candidates
        # Already pre-filtered!
        # All leptons are at least loose
        # Check out setup_candidates for filtering details
        #met_pt, met_phi, ak4, bjets, muons, electrons, taus, photons, jet_images = setup_candidates(df, cfg)
        met_pt, met_phi, ak4, bjets, muons, photons = setup_candidates(df, cfg)

        # Remove jets in accordance with the noise recipe
        if not cfg.RUN.ULEGACYV8 and df['year'] == 2017:
            ak4   = ak4[(ak4.ptraw>50) | (ak4.abseta<2.65) | (ak4.abseta>3.139)]
            bjets = bjets[(bjets.ptraw>50) | (bjets.abseta<2.65) | (bjets.abseta>3.139)]

        # Filtering ak4 jets according to pileup ID
        ak4 = ak4[ak4.puid]

        # Recalculate MET pt and phi based on npv-corrections
        if cfg.MET.XYCORR:
            met_pt_uncorr, met_phi_uncorr = met_pt, met_phi
            met_pt, met_phi = met_xy_correction(df, met_pt, met_phi)

        # Muons
        df['is_tight_muon'] = muons.tightId \
                      & (muons.iso < cfg.MUON.CUTS.TIGHT.ISO) \
                      & (muons.pt>cfg.MUON.CUTS.TIGHT.PT) \
                      & (muons.abseta<cfg.MUON.CUTS.TIGHT.ETA)

        dimuons = muons[df['is_tight_muon']].distincts()
        dimuon_charge = dimuons.i0['charge'] + dimuons.i1['charge']

        df['MT_mu'] = ((muons.counts==1) * mt(muons.pt, muons.phi, met_pt, met_phi)).max()

#        # Electrons
#        df['is_tight_electron'] = electrons.tightId \
#                            & (electrons.pt > cfg.ELECTRON.CUTS.TIGHT.PT) \
#                            & (electrons.absetasc < cfg.ELECTRON.CUTS.TIGHT.ETA)

#        dielectrons = electrons.distincts()
#        dielectron_charge = dielectrons.i0['charge'] + dielectrons.i1['charge']

#        df['MT_el'] = ((electrons.counts==1) * mt(electrons.pt, electrons.phi, met_pt, met_phi)).max()

        selection = processor.PackedSelection()

        # Triggers
        pass_all = np.ones(df.size)==1
        selection.add('inclusive', pass_all)
        selection = trigger_selection(selection, df, cfg)

        # Common selection
#        selection.add('veto_ele', electrons.counts==0)
        selection.add('veto_muo', muons.counts==0)
        selection.add('veto_photon', photons.counts==0)
#        selection.add('veto_tau', taus.counts==0)
#        selection.add('at_least_one_tau', taus.counts>0)

        # B jets are treated using veto weights
        # So accept them in MC, but reject in data
        if df['is_data']:
            selection.add('veto_b', bjets.counts==0)
        else:
            selection.add('veto_b', pass_all)

        df['htmiss'] = ak4[ak4.pt>30].p4.sum().pt
        df['ht'] = ak4[ak4.pt>30].pt.sum()

        # Dimuon CR
        leadmuon_index=muons.pt.argmax()
        selection.add('two_tight_muons', muons[df['is_tight_muon']].counts==2)
        selection.add('mu_pt_trig_safe', muons[df['is_tight_muon']].pt.max() > 25)
        selection.add('two_muons', muons.counts==2)

        selection.add('dimuon_mass', ((dimuons.mass > cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MIN) \
                                    & (dimuons.mass < cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MAX)).any())
        selection.add('dimuon_charge', (dimuon_charge==0).counts==1)


        # Single muon CR
        selection.add('one_muon', muons.counts==1)
        selection.add('mt_mu', df['MT_mu'] < cfg.SELECTION.CONTROL.SINGLEMU.MT)

#        # Diele CR
#        leadelectron_index=electrons.pt.argmax()
#
#        selection.add('one_electron', electrons.counts==1)
#        selection.add('two_electrons', electrons.counts==2)
#        selection.add('at_least_one_tight_el', df['is_tight_electron'].any())
#
#        selection.add('dielectron_mass', ((dielectrons.mass > cfg.SELECTION.CONTROL.DOUBLEEL.MASS.MIN)  \
#                                        & (dielectrons.mass < cfg.SELECTION.CONTROL.DOUBLEEL.MASS.MAX)).any())
#        selection.add('dielectron_charge', (dielectron_charge==0).any())
#
#        # Single Ele CR
#        selection.add('met_el', met_pt > cfg.SELECTION.CONTROL.SINGLEEL.MET)
#        selection.add('mt_el', df['MT_el'] < cfg.SELECTION.CONTROL.SINGLEEL.MT)
#
        # Photon CR
        leadphoton_index=photons.pt.argmax()

        df['is_tight_photon'] = photons.mediumId & photons.barrel

        selection.add('one_photon', photons.counts==1)
        selection.add('at_least_one_tight_photon', df['is_tight_photon'].any())
        selection.add('photon_pt', photons.pt.max() > cfg.PHOTON.CUTS.TIGHT.PT)
        selection.add('photon_pt_trig', photons.pt.max() > cfg.PHOTON.CUTS.TIGHT.PTTRIG)

        # Fill histograms
        output = self.accumulator.identity()

        # Weights
        evaluator = evaluator_from_config(cfg)

        weights = processor.Weights(size=df.size, storeIndividual=True)
        if not df['is_data']:
            weights.add('gen', df['Generator_weight'])

#            try:
#                weights.add('prefire', df['PrefireWeight'])
#            except KeyError:
#                weights.add('prefire', np.ones(df.size))

##            weights = candidate_weights(weights, df, evaluator, muons, electrons, photons, cfg)
#            weights = candidate_weights(weights, df, evaluator, muons, photons, cfg)

#            # B jet veto weights
#            bsf_variations = btag_weights(bjets,cfg)
#            weights.add("bveto", (1-bsf_variations["central"]).prod())

            weights = pileup_weights(weights, df, evaluator, cfg)

        # Sum of all weights to use for normalization
        # TODO: Deal with systematic variations
        output['nevents'][dataset] += df.size
        if not df['is_data']:
            output['sumw'][dataset] +=  df['genEventSumw']
            output['sumw2'][dataset] +=  df['genEventSumw2']
            output['sumw_pileup'][dataset] +=  weights._weights['pileup'].sum()

        regions = zp2mu2nu_regions(cfg)

        # Get veto weights (only for MC)
#        if not df['is_data']:
#            veto_weights = get_veto_weights(df, cfg, evaluator, electrons, muons, taus, do_variations=cfg.RUN.UNCERTAINTIES.VETO_WEIGHTS)
        
        for region, cuts in regions.items():
            if not re.match(cfg.RUN.REGIONREGEX, region):
                continue
            # Run on selected regions only
            exclude = [None]
            if region == 'sr_vbf_no_pu':
                exclude = ['pileup']
            region_weights = copy.deepcopy(weights)

            if not df['is_data']:
#                ### Trigger weights
#                if re.match(r'cr_(\d+)e.*', region):
#                    p_pass_data = 1 - (1-evaluator["trigger_electron_eff_data"](electrons.etasc, electrons.pt)).prod()
#                    p_pass_mc   = 1 - (1-evaluator["trigger_electron_eff_mc"](electrons.etasc, electrons.pt)).prod()
#                    trigger_weight = p_pass_data/p_pass_mc
#                    trigger_weight[np.isnan(trigger_weight)] = 1
#                    region_weights.add('trigger', trigger_weight)
#                if re.match(r'cr_(\d+)m.*', region) or re.match('sr_.*', region):
#                    met_trigger_sf(region_weights, diak4, df, apply_categorized=cfg.RUN.APPLY_CATEGORIZED_SF)
                if re.match(r'cr_g.*', region):
                    photon_trigger_sf(region_weights, photons, df)

#                # Veto weights
#                if re.match('.*no_veto.*', region):
#                    exclude = [
#                            "muon_id_iso_tight",
#                            "muon_id_tight",
#                            "muon_iso_tight",
#                            "muon_id_loose",
#                            "muon_iso_loose",
#                            "ele_reco",
#                            "ele_id_tight",
#                            "ele_id_loose",
#                            "tau_id"
#                        ]
#                    region_weights.add("veto_ele",veto_weights.partial_weight(include=["veto_ele"]))
#                    region_weights.add("veto_muo",veto_weights.partial_weight(include=["veto_muo"]))
#                    region_weights.add("veto_tau",veto_weights.partial_weight(include=["veto_tau"]))

            # This is the default weight for this region
            rweight = region_weights.partial_weight(exclude=exclude)

            # Blinding
            if(self._blind and df['is_data'] and region.startswith('sr')):
                continue

            # Cutflow plot for signal and control regions
            if any(x in region for x in ["sr", "cr", "tr"]):
                output['cutflow_' + region][dataset]['all']+=df.size
                for icut, cutname in enumerate(cuts):
                    output['cutflow_' + region][dataset][cutname] += selection.all(*cuts[:icut+1]).sum()

            mask = selection.all(*cuts)

            one_jet = ak4[ak4.pt>30].counts >= 1                    
            jet1_pt = -999. * np.ones(df.size)
            jet1_pt[one_jet] = ak4[one_jet].pt[:,0].flatten()

            # Check for loose leptons in SR
            event_has_at_least_one_muon = muons.counts > 0
            event_has_at_least_two_muons = muons.counts > 1
            
            two_tight_muons = muons[df["is_tight_muon"]].counts == 2
            
            mu1_pt = -999. * np.ones(df.size)
            mu1_eta = -999. * np.ones(df.size)
            mu1_phi = -999. * np.ones(df.size)
            mu1_pt[two_tight_muons] = muons[two_tight_muons].pt[:,0].flatten()
            mu1_eta[two_tight_muons] = muons[two_tight_muons].eta[:,0].flatten()
            mu1_phi[two_tight_muons] = muons[two_tight_muons].phi[:,0].flatten()
            
            mu2_pt = -999. * np.ones(df.size)
            mu2_eta = -999. * np.ones(df.size)
            mu2_phi = -999. * np.ones(df.size)
            mu2_pt[two_tight_muons] = muons[two_tight_muons].pt[:,1].flatten()
            mu2_eta[two_tight_muons] = muons[two_tight_muons].eta[:,1].flatten()
            mu2_phi[two_tight_muons] = muons[two_tight_muons].phi[:,1].flatten()
            
            dimuon_mass = -999. * np.ones(df.size)
            dimuon_mass[two_tight_muons] = dimuons[two_tight_muons].mass[:,0].flatten()
            
            dimuon_pt = -999. * np.ones(df.size)
            dimuon_pt[two_tight_muons] = dimuons[two_tight_muons].pt[:,0].flatten()
            
            dimuon_eta = -999. * np.ones(df.size)
            dimuon_eta[two_tight_muons] = dimuons[two_tight_muons].eta[:,0].flatten()
            
            dimuon_phi = -999. * np.ones(df.size)
            dimuon_phi[two_tight_muons] = dimuons[two_tight_muons].phi[:,0].flatten()
            
            mt_mu1mu2 = -999. * np.ones(df.size)
            mt_mu1mu2[two_tight_muons] = mt(muons[two_tight_muons].pt[:,0].flatten(), muons[two_tight_muons].phi[:,0].flatten(), \
                                            muons[two_tight_muons].pt[:,1].flatten(), muons[two_tight_muons].phi[:,1].flatten())
            
            mt_mu1met = -999. * np.ones(df.size)
            mt_mu1met[two_tight_muons] = mt(muons[two_tight_muons].pt[:,0].flatten(), muons[two_tight_muons].phi[:,0].flatten(), \
                                            met_pt[two_tight_muons], met_phi[two_tight_muons])
            
            mt_mu2met = -999. * np.ones(df.size)
            mt_mu2met[two_tight_muons] = mt(muons[two_tight_muons].pt[:,1].flatten(), muons[two_tight_muons].phi[:,1].flatten(), \
                                            met_pt[two_tight_muons], met_phi[two_tight_muons])
            
            mt_dimuonmet = -999. * np.ones(df.size)
            mt_dimuonmet[two_tight_muons] = mt(dimuons[two_tight_muons].pt[:,0].flatten(), dimuons[two_tight_muons].phi[:,0].flatten(), \
                                               met_pt[two_tight_muons], met_phi[two_tight_muons])
            

            if cfg.RUN.SAVE.TREE:
                if region in cfg.RUN.SAVE.TREE_REGIONS: 
                    output['tree_int64'][region]["event"]             +=  processor.column_accumulator(df["event"][mask])
                    output['tree_int64'][region]["run"]               +=  processor.column_accumulator(df["run"][mask])
                    output['tree_int64'][region]["lumi"]              +=  processor.column_accumulator(df["luminosityBlock"][mask])
                    
                    output['tree_float16'][region]["nLooseMuon"]        +=  processor.column_accumulator(np.float16(muons.counts[mask]))
                    output['tree_float16'][region]["nTightMuon"]        +=  processor.column_accumulator(np.float16(muons[df['is_tight_muon']].counts[mask]))
#                    output['tree_float16'][region]["nLooseElectron"]    +=  processor.column_accumulator(np.float16(electrons.counts[mask]))
#                    output['tree_float16'][region]["nTightElectron"]    +=  processor.column_accumulator(np.float16(electrons[df['is_tight_electron']].counts[mask]))
#                    output['tree_float16'][region]["nLooseTau"]         +=  processor.column_accumulator(np.float16(taus.counts[mask]))
                    output['tree_float16'][region]["nMediumBJet"]       +=  processor.column_accumulator(np.float16(bjets.counts[mask]))
                    output['tree_float16'][region]["nJets"]       +=  processor.column_accumulator(np.float16(ak4[ak4.pt>30].counts[mask]))

                    output['tree_float16'][region]["jet1_pt"]   +=  processor.column_accumulator(jet1_pt[mask])

                    output['tree_float16'][region]["met_pt"]       +=  processor.column_accumulator(np.float16(met_pt[mask]))
                    output['tree_float16'][region]["met_phi"]       +=  processor.column_accumulator(np.float16(met_phi[mask]))

                    output['tree_float16'][region]["mu1_pt"]   +=  processor.column_accumulator(mu1_pt[mask])
                    output['tree_float16'][region]["mu1_eta"]  +=  processor.column_accumulator(mu1_eta[mask])
                    output['tree_float16'][region]["mu1_phi"]  +=  processor.column_accumulator(mu1_phi[mask])
                    output['tree_float16'][region]["mu2_pt"]   +=  processor.column_accumulator(mu2_pt[mask])
                    output['tree_float16'][region]["mu2_eta"]  +=  processor.column_accumulator(mu2_eta[mask])
                    output['tree_float16'][region]["mu2_phi"]  +=  processor.column_accumulator(mu2_phi[mask])
                    output['tree_float16'][region]["dimuon_mass"]  +=  processor.column_accumulator(dimuon_mass[mask])
                    output['tree_float16'][region]["dimuon_pt"]  +=  processor.column_accumulator(dimuon_pt[mask])
                    output['tree_float16'][region]["dimuon_eta"]  +=  processor.column_accumulator(dimuon_eta[mask])
                    output['tree_float16'][region]["dimuon_phi"]  +=  processor.column_accumulator(dimuon_phi[mask])
                    output['tree_float16'][region]["mt_mu1mu2"]  +=  processor.column_accumulator(mt_mu1mu2[mask])
                    output['tree_float16'][region]["mt_mu1met"]  +=  processor.column_accumulator(mt_mu1met[mask])
                    output['tree_float16'][region]["mt_mu2met"]  +=  processor.column_accumulator(mt_mu2met[mask])
                    output['tree_float16'][region]["mt_dimuonmet"]  +=  processor.column_accumulator(mt_mu2met[mask])

#                    event_has_at_least_one_electron = electrons.counts > 0
#                    event_has_at_least_two_electrons = electrons.counts > 1

#                    lead_electron_pt = -999. * np.ones(df.size)
#                    lead_electron_eta = -999. * np.ones(df.size)
#                    lead_electron_pt[event_has_at_least_one_electron] = electrons[event_has_at_least_one_electron].pt[:,0].flatten()
#                    lead_electron_eta[event_has_at_least_one_electron] = electrons[event_has_at_least_one_electron].eta[:,0].flatten()

#                    trail_electron_pt = -999. * np.ones(df.size)
#                    trail_electron_eta = -999. * np.ones(df.size)
#                    trail_electron_pt[event_has_at_least_two_electrons] = electrons[event_has_at_least_two_electrons].pt[:,1].flatten()
#                    trail_electron_eta[event_has_at_least_two_electrons] = electrons[event_has_at_least_two_electrons].eta[:,1].flatten()

#                    output['tree_float16'][region]["lead_electron_pt"]   +=  processor.column_accumulator(lead_electron_pt[mask])
#                    output['tree_float16'][region]["lead_electron_eta"]  +=  processor.column_accumulator(lead_electron_eta[mask])
#                    output['tree_float16'][region]["trail_electron_pt"]   +=  processor.column_accumulator(trail_electron_pt[mask])
#                    output['tree_float16'][region]["trail_electron_eta"]  +=  processor.column_accumulator(trail_electron_eta[mask])

                    event_has_bjets = bjets[mask].counts != 0
                    bjet_pt = np.where(event_has_bjets, bjets.pt.max()[mask], -999)
                    bjet_eta = np.where(event_has_bjets, bjets[bjets.pt.argmax()].eta.max()[mask], -999)
                    bjet_phi = np.where(event_has_bjets, bjets[bjets.pt.argmax()].phi.max()[mask], -999)
                    output['tree_float16'][region]["lead_bjet_pt"]   +=  processor.column_accumulator(bjet_pt)
                    output['tree_float16'][region]["lead_bjet_eta"]  +=  processor.column_accumulator(bjet_eta)
                    output['tree_float16'][region]["lead_bjet_phi"]  +=  processor.column_accumulator(bjet_phi)

                    for name, w in region_weights._weights.items():
                        output['tree_float16'][region][f"weight_{name}"] += processor.column_accumulator(np.float32(w[mask]))
                    
                    output['tree_float16'][region][f"weight_total"] += processor.column_accumulator(np.float16(rweight[mask]))

            # Save the event numbers of events passing this selection
            if cfg.RUN.SAVE.PASSING:
                output['selected_events'][region] += list(df['event'][mask])


            # Multiplicities
            def fill_mult(name, candidates):
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  multiplicity=candidates[mask].counts,
                                  weight=rweight[mask]
                                  )

            fill_mult('ak4_mult', ak4[ak4.pt>30])
            fill_mult('bjet_mult',bjets)
#            fill_mult('loose_ele_mult',electrons)
#            fill_mult('tight_ele_mult',electrons[df['is_tight_electron']])
            fill_mult('loose_muo_mult',muons)
            fill_mult('tight_muo_mult',muons[df['is_tight_muon']])
#            fill_mult('tau_mult',taus)
            fill_mult('photon_mult',photons)
            # Number of additional jets in the central region, |eta| < 2.5
#            fill_mult('extra_ak4_mult', extra_ak4_central[extra_ak4_central.pt>30])

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  **kwargs
                                  )

            # Monitor weights
            for wname, wvalue in region_weights._weights.items():
                ezfill("weights", weight_type=wname, weight_value=wvalue[mask])
                ezfill("weights_wide", weight_type=wname, weight_value=wvalue[mask])

#            # B tag discriminator
#            btag = getattr(ak4, cfg.BTAG.ALGO)
#            w_btag = weight_shape(btag[mask], rweight[mask])
#            ezfill('ak4_btag', btag=btag[mask].flatten(), weight=w_btag )

            # MET
            ezfill('met',                met=met_pt[mask],            weight=rweight[mask] )
            ezfill('met_phi',            phi=met_phi[mask],           weight=rweight[mask] )
#            ezfill('recoil',             recoil=df["recoil_pt"][mask],      weight=rweight[mask] )
#            ezfill('recoil_phi',         phi=df["recoil_phi"][mask],        weight=rweight[mask] )

            # b-tag weight up and down variations
            if cfg.RUN.UNCERTAINTIES.BTAG_SF and not df['is_data']:
                rw = region_weights.partial_weight(exclude=exclude+['bveto'])
                
                variations = {
                    "btagSFNom" : rw*(1-bsf_variations['central']).prod(),
                    "btagSFUp" : rw*(1-bsf_variations['up']).prod(),
                    "btagSFDown" : rw*(1-bsf_variations['down']).prod(),
                }

#            if cfg.RUN.UNCERTAINTIES.ELECTRON_SF and re.match('cr_(\d)e_vbf', region) and not df['is_data']:
#                eleloose_id_sf, eletight_id_sf, ele_reco_sf = get_varied_ele_sf(electrons, df, evaluator)
#                rw = region_weights.partial_weight(exclude=exclude+['ele_id_tight','ele_id_loose'])
#                for ele_id_variation in eletight_id_sf.keys():
#                    ezfill('mjj_ele_id', 
#                        mjj=df['mjj'][mask], 
#                        variation=ele_id_variation,
#                        weight=(rw * eletight_id_sf[ele_id_variation].prod() * eleloose_id_sf[ele_id_variation].prod())[mask],
#                    )

#                for ele_reco_variation, w in ele_reco_sf.items():
#                    ezfill('mjj_ele_reco',
#                        mjj=df['mjj'][mask],
#                        variation=ele_reco_variation,
#                        weight=(rw * w)[mask]
#                    )

#            if cfg.RUN.UNCERTAINTIES.MUON_SF and re.match('cr_(\d)m_vbf', region) and not df['is_data']:
#                muon_looseid_sf, muon_tightid_sf, muon_looseiso_sf, muon_tightiso_sf = get_varied_muon_sf(muons, df, evaluator)
#                rw = region_weights.partial_weight(exclude=exclude+['muon_id_tight','muon_id_loose'])

#                for mu_id_variation in muon_looseid_sf.keys():
#                    ezfill('mjj_muon_id', 
#                        mjj=df['mjj'][mask], 
#                        variation=mu_id_variation,
#                        weight=(rw * muon_tightid_sf[mu_id_variation].prod() * muon_looseid_sf[mu_id_variation].prod())[mask],
#                    )

#                for mu_iso_variation in muon_looseiso_sf.keys():
#                    ezfill('mjj_muon_iso', 
#                        mjj=df['mjj'][mask], 
#                        variation=mu_iso_variation,
#                        weight=(rw * muon_tightiso_sf[mu_iso_variation].prod() * muon_looseiso_sf[mu_iso_variation].prod())[mask],
#                    )
#
#
#            if cfg.RUN.UNCERTAINTIES.ELECTRON_TRIGGER_SF and not df['is_data'] and re.match('cr_(\d)e_vbf', region):
#                # Note that electrons in the gap do not count in this study
#                mask_electron_nogap = (np.abs(electrons.etasc)<1.4442) | (np.abs(electrons.etasc)>1.566)
#                electrons_nogap = electrons[mask_electron_nogap]
#
#                # Up and down variations: Vary the efficiency in data
#                data_eff_up = evaluator['trigger_electron_eff_data'](electrons_nogap.etasc, electrons_nogap.pt) + evaluator['trigger_electron_eff_data_error'](electrons_nogap.etasc, electrons_nogap.pt)
#                data_eff_down = evaluator['trigger_electron_eff_data'](electrons_nogap.etasc, electrons_nogap.pt) - evaluator['trigger_electron_eff_data_error'](electrons_nogap.etasc, electrons_nogap.pt)
#
#                p_pass_data = 1 - (1-evaluator["trigger_electron_eff_data"](electrons_nogap.etasc, electrons_nogap.pt)).prod()
#                p_pass_data_up = 1 - (1-data_eff_up).prod()
#                p_pass_data_down = 1 - (1-data_eff_down).prod()
#
#                p_pass_mc = 1 - (1-evaluator["trigger_electron_eff_mc"](electrons_nogap.etasc, electrons_nogap.pt)).prod()
#
#                trigger_weight_nom = p_pass_data / p_pass_mc
#                trigger_weight_up = p_pass_data_up / p_pass_mc
#                trigger_weight_down = p_pass_data_down / p_pass_mc
#
#                trigger_weight_nom[np.isnan(trigger_weight_nom) | np.isinf(trigger_weight_nom)] = 1.
#                trigger_weight_up[np.isnan(trigger_weight_up) | np.isinf(trigger_weight_up)] = 1.
#                trigger_weight_down[np.isnan(trigger_weight_down) | np.isinf(trigger_weight_down)] = 1.
#
#                ele_trig_sf = {
#                    "nom" : trigger_weight_nom,
#                    "up" : trigger_weight_up,
#                    "down" : trigger_weight_down,
#                }
#
#                for variation, trigw in ele_trig_sf.items():
#                    rw = region_weights.partial_weight(exclude=exclude+['trigger_ele'])
#                    ezfill(
#                        'mjj_ele_trig_weight',
#                        mjj=df['mjj'][mask],
#                        variation=variation,
#                        weight=(rw*trigw)[mask] 
#                    )
#
#            if cfg.RUN.UNCERTAINTIES.PILEUP_SF and not df['is_data']:
#                rw_nopu = region_weights.partial_weight(exclude=exclude+['pileup'])
#
#                puweights = pileup_sf_variations(df, evaluator, cfg)
#                for puvar, w in puweights.items():
#                    ezfill('mjj_unc',
#                        mjj=df['mjj'][mask],
#                        uncertainty=puvar,
#                        weight=(rw_nopu * w)[mask]
#                    )
#                    for score_type in cfg.NN_MODELS.UNCERTAINTIES:
#                        ezfill(f'{score_type}_unc',
#                            score=df[score_type][:, 1][mask],
#                            uncertainty=puvar,
#                            weight=(rw_nopu * w)[mask]
#                        )
#
#            # Variations in the prefire weight
#            if cfg.RUN.UNCERTAINTIES.PREFIRE_SF and not df['is_data']:
#                rw_nopref = region_weights.partial_weight(exclude=exclude+['prefire'])
#
#                # Fill mjj and score distributions with the variations of the prefire weight
#                try:
#                    pref_weights = {
#                        "prefireNom" : df['PrefireWeight'],
#                        "prefireUp" : df['PrefireWeight_Up'],
#                        "prefireDown" : df['PrefireWeight_Down'],
#                    }
#    
#                    for variation, w in pref_weights.items():
#                        ezfill('mjj_unc',
#                            mjj=df['mjj'][mask],
#                            uncertainty=variation,
#                            weight=(rw_nopref * w)[mask]
#                        )
#                        for score_type in cfg.NN_MODELS.UNCERTAINTIES:
#                            ezfill(f'{score_type}_unc',
#                                score=df[score_type][:, 1][mask],
#                                uncertainty=variation,
#                                weight=(rw_nopref * w)[mask]
#                            )
#                
#                except KeyError:
#                    pass
#
#            if cfg.RUN.UNCERTAINTIES.VETO_WEIGHTS and 'no_veto_all' in region and not df['is_data']:
#                variations = ['nominal', 'tau_id_up', 'tau_id_dn', 
#                    'ele_id_up', 'ele_id_dn', 'ele_reco_up', 'ele_reco_dn',
#                    'muon_id_up', 'muon_id_dn', 'muon_iso_up', 'muon_iso_dn'
#                    ]
#                rw_no_veto = region_weights.partial_weight(exclude=exclude+['veto'])
#                for v in variations:
#                    ezfill('mjj_veto_weight',
#                        mjj=df['mjj'][mask],
#                        variation=v,
#                        weight=(rw_no_veto * veto_weights.partial_weight(include=[v]))[mask]
#                    )
#
#
#            # Photon CR data-driven QCD estimate
#            if df['is_data'] and re.match("cr_g.*", region) and re.match("(SinglePhoton|EGamma).*", dataset):
#                w_imp = photon_impurity_weights(photons[leadphoton_index].pt.max()[mask], df["year"])
#                output['mjj'].fill(
#                                    dataset=data_driven_qcd_dataset(dataset),
#                                    region=region,
#                                    mjj=df["mjj"][mask],
#                                    weight=rweight[mask] * w_imp
#                                )
#                output['recoil'].fill(
#                                    dataset=data_driven_qcd_dataset(dataset),
#                                    region=region,
#                                    recoil=df["recoil_pt"][mask],
#                                    weight=rweight[mask] * w_imp
#                                )
#
#            # Theory uncertainties on V+jets processes
#            if cfg.RUN.UNCERTAINTIES.THEORY:
#                # Scale and PDF uncertainties for Z(vv) / W(lv) and gamma+jets / Z(vv) ratios
#                # Information will be stored in Z(vv) and gamma+jets histograms
#                if df['is_lo_z'] or df['is_nlo_z'] or df['is_lo_z_ewk'] or df['is_lo_g'] or df['is_lo_g_ewk']:
#                    theory_uncs = [x for x in cfg.SF.keys() if x.startswith('unc')]
#                    for unc in theory_uncs:
#                        reweight = evaluator[unc](gen_v_pt)
#                        w = (region_weights.weight() * reweight)[mask]
#                        ezfill(
#                            'mjj_unc',
#                            mjj=df['mjj'][mask],
#                            uncertainty=unc,
#                            weight=w)
#                        
#                        for score_type in cfg.NN_MODELS.UNCERTAINTIES:
#                            ezfill(
#                                f'{score_type}_unc',
#                                score=df[score_type][:, 1][mask],
#                                uncertainty=unc,
#                                weight=w)
#
#                # Distributions without the NLO EWK weights for the V+jets samples
#                # This is used to compute the NLO EWK uncertainty on V+jets transfer factors
#                if df['is_nlo_z'] or df['is_nlo_w'] or df['is_lo_g']:
#                    ewk_weights = get_nlo_ewk_weights(df, evaluator, gen_v_pt)
#                    
#                    # Fill the NN score and mjj distributions without the NLO EWK correction applied
#                    weight_noewk = (region_weights.weight() / ewk_weights)[mask]
#
#                    ezfill('mjj_noewk',    mjj=df['mjj'][mask],     weight=weight_noewk)
#
#                    for score_type in cfg.NN_MODELS.UNCERTAINTIES:
#                        ezfill(f'{score_type}_noewk',   score=df[score_type][:, 1][mask],   weight=weight_noewk)
                            
            # Muons
#            if '_1m_' in region or '_2m_' in region or 'no_veto' in region:
#                w_allmu = weight_shape(muons.pt[mask], rweight[mask])
#                ezfill('muon_pt',   pt=muons.pt[mask].flatten(),    weight=w_allmu )
#                ezfill('muon_pt_abseta',pt=muons.pt[mask].flatten(),abseta=muons.eta[mask].flatten(),    weight=w_allmu )
#                ezfill('muon_mt',   mt=df['MT_mu'][mask],           weight=rweight[mask])
#                ezfill('muon_eta',  eta=muons.eta[mask].flatten(),  weight=w_allmu)
#                ezfill('muon_phi',  phi=muons.phi[mask].flatten(),  weight=w_allmu)

            # Dimuon
            if '_2m_' in region:
                w_dimu = weight_shape(dimuons.pt[mask], rweight[mask])                
                #ezfill('muon_pt0',      pt=dimuons.i0.pt[mask].flatten(),           weight=w_dimu)
                #ezfill('muon_pt1',      pt=dimuons.i1.pt[mask].flatten(),           weight=w_dimu)
                #ezfill('muon_eta0',     eta=dimuons.i0.eta[mask].flatten(),         weight=w_dimu)
                #ezfill('muon_eta1',     eta=dimuons.i1.eta[mask].flatten(),         weight=w_dimu)
                #ezfill('muon_phi0',     phi=dimuons.i0.phi[mask].flatten(),         weight=w_dimu)
                #ezfill('muon_phi1',     phi=dimuons.i1.phi[mask].flatten(),         weight=w_dimu)
                #ezfill('dimuon_pt',     pt=dimuons.pt[mask].flatten(),              weight=w_dimu)
                #ezfill('dimuon_eta',    eta=dimuons.eta[mask].flatten(),            weight=w_dimu)
                #ezfill('dimuon_mass',   dilepton_mass=dimuons.mass[mask].flatten(), weight=w_dimu )

                ezfill("mu1_pt", pt=mu1_pt[mask].flatten(), weight=w_dimu)
                ezfill("mu1_eta", eta=mu1_eta[mask].flatten(), weight=w_dimu)
                ezfill("mu1_phi", phi=mu1_phi[mask].flatten(), weight=w_dimu)
                ezfill("mu2_pt", pt=mu2_pt[mask].flatten(), weight=w_dimu)
                ezfill("mu2_eta", eta=mu2_eta[mask].flatten(), weight=w_dimu)
                ezfill("mu2_phi", phi=mu2_phi[mask].flatten(), weight=w_dimu)
                ezfill("dimuon_mass", dilepton_mass=dimuon_mass[mask].flatten(), weight=w_dimu)
                ezfill("dimuon_pt", pt=dimuon_pt[mask].flatten(), weight=w_dimu)
                ezfill("dimuon_eta", eta=dimuon_eta[mask].flatten(), weight=w_dimu)
                ezfill("dimuon_phi", phi=dimuon_phi[mask].flatten(), weight=w_dimu)
                ezfill("mt_mu1mu2", mt=mt_mu1mu2[mask].flatten(), weight=w_dimu)
                ezfill("mt_mu1met", mt=mt_mu1met[mask].flatten(), weight=w_dimu)
                ezfill("mt_mu2met", mt=mt_mu2met[mask].flatten(), weight=w_dimu)
                ezfill("mt_dimuonmet", mt=mt_mu2met[mask].flatten(), weight=w_dimu)


#            # Electrons
#            if '_1e_' in region or '_2e_' in region or 'no_veto' in region:
#                w_allel = weight_shape(electrons.pt[mask], rweight[mask])
#                ezfill('electron_pt',   pt=electrons.pt[mask].flatten(),    weight=w_allel)
#                ezfill('electron_pt_eta',   pt=electrons.pt[mask].flatten(), eta=electrons.eta[mask].flatten(),    weight=w_allel)
#                ezfill('electron_mt',   mt=df['MT_el'][mask],               weight=rweight[mask])
#                ezfill('electron_eta',  eta=electrons.eta[mask].flatten(),  weight=w_allel)
#                ezfill('electron_phi',  phi=electrons.phi[mask].flatten(),  weight=w_allel)

#            # Dielectron
#            if '_2e_' in region:
#                w_diel = weight_shape(dielectrons.pt[mask], rweight[mask])
#                ezfill('electron_pt0',      pt=dielectrons.i0.pt[mask].flatten(),               weight=w_diel)
#                ezfill('electron_pt1',      pt=dielectrons.i1.pt[mask].flatten(),               weight=w_diel)
#                ezfill('electron_eta0',     eta=dielectrons.i0.eta[mask].flatten(),             weight=w_diel)
#                ezfill('electron_eta1',     eta=dielectrons.i1.eta[mask].flatten(),             weight=w_diel)
#                ezfill('electron_phi0',     phi=dielectrons.i0.phi[mask].flatten(),             weight=w_diel)
#                ezfill('electron_phi1',     phi=dielectrons.i1.phi[mask].flatten(),             weight=w_diel)
#                ezfill('dielectron_pt',     pt=dielectrons.pt[mask].flatten(),                  weight=w_diel)
#                ezfill('dielectron_eta',    eta=dielectrons.eta[mask].flatten(),                weight=w_diel)
#                ezfill('dielectron_mass',   dilepton_mass=dielectrons.mass[mask].flatten(),     weight=w_diel)

#            # Photon
#            if '_g_' in region:
#                w_leading_photon = weight_shape(photons[leadphoton_index].pt[mask],rweight[mask])
#                ezfill('photon_pt0',              pt=photons[leadphoton_index].pt[mask].flatten(),    weight=w_leading_photon)
#                ezfill('photon_eta0',             eta=photons[leadphoton_index].eta[mask].flatten(),  weight=w_leading_photon)
#                ezfill('photon_phi0',             phi=photons[leadphoton_index].phi[mask].flatten(),  weight=w_leading_photon)
#                ezfill('photon_eta_phi',          eta=photons[leadphoton_index].eta[mask].flatten(), phi=photons[leadphoton_index].phi[mask].flatten(),  weight=w_leading_photon)

                # w_drphoton_jet = weight_shape(df['dRPhotonJet'][mask], rweight[mask])

#            # Tau
#            if 'no_veto' in region:
#                w_all_taus = weight_shape(taus.pt[mask], rweight[mask])
#                ezfill("tau_pt", pt=taus.pt[mask].flatten(), weight=w_all_taus)

            # PV
            ezfill('npv', nvtx=df['PV_npvs'][mask], weight=rweight[mask])
            ezfill('npvgood', nvtx=df['PV_npvsGood'][mask], weight=rweight[mask])

            ezfill('npv_nopu', nvtx=df['PV_npvs'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])
            ezfill('npvgood_nopu', nvtx=df['PV_npvsGood'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])

            ezfill('rho_all', rho=df['fixedGridRhoFastjetAll'][mask], weight=region_weights.partial_weight(exclude=exclude)[mask])
            ezfill('rho_central', rho=df['fixedGridRhoFastjetCentral'][mask], weight=region_weights.partial_weight(exclude=exclude)[mask])
            ezfill('rho_all_nopu', rho=df['fixedGridRhoFastjetAll'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])
            ezfill('rho_central_nopu', rho=df['fixedGridRhoFastjetCentral'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])
        return output

    def postprocess(self, accumulator):
        return accumulator
