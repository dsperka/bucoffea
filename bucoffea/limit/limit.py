#!/usr/bin/env python

import os
import csv
import argparse

from klepto.archives import dir_archive
from datetime import datetime
from typing import Dict
from pprint import pprint

from bucoffea.helpers.git import git_rev_parse, git_diff

pjoin = os.path.join


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', type=str, help='Input path to use.')
    parser.add_argument('--distribution', type=str, help='Distribution to make limit inputs for.', default='cnn_score')
    parser.add_argument('--channel', type=str, help='Channel to make inputs for.', default='vbfhinv')
    parser.add_argument('--unblind', action='store_true', help='Include signal region data')
    parser.add_argument('--years', nargs='*', default=[2017, 2018], help='The years to prepare the limit input for')
    parser.add_argument('--one-fifth-unblind', action='store_true', help='1/5th unblinding: Scale the MC in signal region by 1/5')
    parser.add_argument('--mcscales', type=str, default=None, help='An optional txt file for storing the MC scales per region')
    args = parser.parse_args()

    if not os.path.isdir(args.inpath):
        raise RuntimeError(f"Commandline argument is not a directory: {args.inpath}")

    if args.one_fifth_unblind and args.unblind:
        raise IOError("--one-fifth-unblind and --unblind cannot be specified at the same time.")

    return args


def dump_info(args, outdir) -> None:
    """
    Dump repo version info and command line arguments to an INFO.txt file under outdir.
    """
    infofile = pjoin(outdir, 'INFO.txt')
    with open(infofile, 'w+') as f:
        f.write(f'Limit input creation: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        f.write('Repo information:\n')

        f.write(git_rev_parse()+'\n')
        f.write(git_diff()+'\n')

        f.write('Command line arguments:\n\n')
        cli = vars(args)
        for arg, val in cli.items():
            f.write(f'{arg}: {val}\n')


def get_mc_scales(infile: str) -> Dict[str, float]:
    """
    From the given input file containing MC scales per region,
    write them to a dictionary and return the dict.
    """
    scales = {}
    # If an input file is not provided, just return an empty dict
    if not infile:
        return scales
    
    with open(infile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            scales[row[0]] = float(row[1])

    return scales


def main():
    args = parse_commandline()

    acc = dir_archive(args.inpath, serialized=True, compression=0, memsize=1e3)

    acc.load(args.distribution)
    acc.load('sumw')
    acc.load('sumw_pileup')
    acc.load('nevents')

    outdir = pjoin('./output/',list(filter(lambda x:x,args.inpath.split('/')))[-1])

    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    
    # Handle MC scaling
    mcscales = get_mc_scales(args.mcscales)
    if mcscales:
        print('Will apply the following MC scaling:')
        pprint(mcscales)

    # Store the command line arguments in the INFO.txt file
    dump_info(args, outdir)

    # Create limit input ROOT files per channel specified from the command line
    for channel in args.channel.split(','):
        if channel == 'monojet':
            from legacy_monojet import legacy_limit_input_monojet
            legacy_limit_input_monojet(acc, outdir=outdir, unblind=args.unblind)
        elif channel == 'monov':
            from legacy_monov import legacy_limit_input_monov
            legacy_limit_input_monov(acc, outdir=outdir, unblind=args.unblind)
        elif channel == 'vbfhinv':
            from legacy_vbf import legacy_limit_input_vbf
            legacy_limit_input_vbf(acc,
                    distribution=args.distribution,
                    outdir=outdir, 
                    unblind=args.unblind, 
                    years=args.years, 
                    one_fifth_unblind=args.one_fifth_unblind,
                    mcscales=mcscales,
                )
        else:
            raise ValueError(f'Unknown channel specified: {channel}')


if __name__ == "__main__":
    main()
