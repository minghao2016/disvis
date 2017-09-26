"""
Extract ligand translations and rotations consistent with a specified number
of restraints.
"""

from __future__ import division

from argparse import ArgumentParser
from os.path import abspath, join, splitext
from glob import glob
from sys import exit
import operator

import numpy as np

from disvis.volume import Volume
from disvis.rotations import proportional_orientations
from disvis.helpers import mkdir_p


def parse_args():
    """Parse the command-line arguments."""

    p = ArgumentParser(description=__doc__)
    p.add_argument('consistent_restraints', type=int, metavar='<int>',
                   help="Minimum number of required consistent restraints.")
    p.add_argument('-i', '--input', dest='input', type=abspath, default='.',
                   metavar='<dir>',
                   help="Directory where the input files can be found.")
    p.add_argument('-o', '--output', dest='output', type=abspath, default='.',
                   metavar='<dir>',
                   help="Directory where the output file will be stored.")
    p.add_argument('-e', '--exact', dest='exact', action='store_true',
                   help="Only write moves that are consistent with exactly the"
                        " specified restraints.")

    args = p.parse_args()
    return args


def main():

    args = parse_args()

    ais_fnames = glob(join(args.input, 'ais*.mrc'))
    nrot = len(ais_fnames)
    if nrot == 0:
        raise ValueError("No input files where found in specified directory")
    quats = proportional_orientations(nrot, metric='number')[0]
    nrot_per_job = len(glob(join(args.input, 'ais_0_*.mrc')))
    mkdir_p(args.output)

    print 'Analyzing data ...'
    n = 1
    # Determine the logical operator
    l_operator = operator.ge
    if args.exact:
        l_operator = operator.eq
    fn_out = join(
        args.output, 'consistent_moves_{:d}.txt'
    ).format(args.consistent_restraints)
    with open(fn_out, 'w') as f:
        # Write nsol x y z q0 q1 q2 q3
        line = '{:d}' + ' {:.2f} ' * 3 + ' {:6.4f}' * 4 + '\n'
        for ais_fname in ais_fnames:
            job, ind = [int(x) for x in splitext(ais_fname)[0].split('_')[1:]]
            quat = quats[ind + job * nrot_per_job]
            ais = Volume.fromfile(ais_fname)
            trans = (np.asarray(
                l_operator(ais.array, args.consistent_restraints).nonzero()[::-1]).T
                    * ais.voxelspacing + ais.origin)
            for t in trans:
                data = [n] + list(t) + list(quat)
                f.write(line.format(*data))
                n += 1


if __name__ == '__main__':
    main()
