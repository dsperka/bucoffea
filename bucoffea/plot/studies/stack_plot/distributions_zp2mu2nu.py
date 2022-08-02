#!/usr/bin/env python
import numpy as np
from coffea import hist

Bin = hist.Bin


common_distributions = [
    'mu1_pt',
    'mu1_eta',
    'mu2_pt',
    'mu2_eta',
    'dimuon_mass',
    'dimuon_pt',
]

# Distributions to plot for each region
distributions = {
    'cr_2m_zp2mu2nu'   : common_distributions
} 

binnings = {
    'dimuon_mass': Bin('dimuon_mass', r'$M_{\mu\mu} \ (GeV)$', 300, 0., 300.)
}

ylims = {
}
