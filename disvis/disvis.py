"""Quantify and visualize the information content of distance restraints."""

from argparse import ArgumentParser
import time
from itertools import izip
import sys
from glob import glob
import os
from collections import OrderedDict
import logging
from multiprocessing import RawValue, Lock, Process, cpu_count

import numpy as np

from .pdb import PDB
from .volume import Volume, Volumizer
from .spaces import (InteractionSpace, RestraintSpace, Restraint,
                     AccessibleInteractionSpace, OccupancySpace,
                     InterfaceResidueIdentifyer, InteractionAnalyzer)
from .helpers import RestraintParser, DJoiner
from .rotations import proportional_orientations, quat_to_rotmat


logger = logging.getLogger(__name__)


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("receptor", type=str,
                   help="Receptor / fixed chain.")
    p.add_argument("ligand", type=str,
                   help="Ligand / scanning chain.")
    p.add_argument("restraints", type=file,
                   help="File containing restraints.")

    p.add_argument("-vs", "--voxelspacing", default=2, type=float,
                   help="Voxelspacing of grids.")
    p.add_argument("-a", "--angle", default=20, type=float,
                   help="Rotational sampling density in degree.")
    p.add_argument("-ir", "--interaction-radius", type=float, default=3,
                   help="Interaction radius.")
    p.add_argument("-mi", "--minimum-interaction-volume", type=float, default=300,
                   help="Minimum interaction volume required for a complex.")
    p.add_argument("-mc", "--maximum-clash-volume", type=float, default=200,
                   help="Maximum clash volume allowed for a complex.")
    p.add_argument("-oa", "--occupancy-analysis", action="store_true",
                   help="Perform an occupancy analysis.")
    p.add_argument("-ia", "--interaction-analysis", action="store_true",
                   help="Perform an interaction analysis.")
    # TODO implement
    #p.add_argument("-is", "--interaction-selection", default="CA,O3",
    #               help="Atom names that are included in interaction analysis. Type 'all' to include all atoms.")
    p.add_argument("-s", "--save", action="store_true",
                   help="Save entire accessible interaction space to disk.")
    p.add_argument("-d", "--directory", default=DJoiner('.'), type=DJoiner,
                   help="Directory to store the results.")
    p.add_argument("-p", "--nprocessors", type=int, default=None,
                   help="Number of processors to use during search.")
    # TODO implement
    #p.add_argument("-g", "--gpu", action="store_true",
    #               help="Divert calculations to GPU.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")

    args = p.parse_args()
    return args


class DisVisOptions(object):
    minimum_interaction_volume = 300
    maximum_clash_volume = 200
    voxelspacing = 2
    interaction_radius = 3
    save = False
    save_id = 1
    directory = DJoiner()

    @classmethod
    def fromargs(cls, args):
        for key, value in vars(args).iteritems():
            setattr(cls, key, value)
        return cls


class DisVis(object):
    def __init__(self, receptor, ligand, restraints, options, counter=None):
        self.receptor = receptor
        self.ligand = ligand
        self.restraints = restraints
        self.options = options
        self._initialized = False
        self._n = 0
        self._counter = counter

    def initialize(self):

        self._volumizer = Volumizer(
            self.receptor, self.ligand,
            voxelspacing=self.options.voxelspacing,
            interaction_radius=self.options.interaction_radius,
        )

        rcore = self._volumizer.rcore
        rsurface = self._volumizer.rsurface
        lcore = self._volumizer.lcore
        interaction_space = Volume.zeros_like(rcore, dtype=np.int32)
        self._interaction_space_calc = InteractionSpace(
            interaction_space, rcore, rsurface, lcore,
            max_clash=self.options.maximum_clash_volume,
            min_inter=self.options.minimum_interaction_volume,
        )
        restraint_space = Volume.zeros_like(rcore, dtype=np.int32)
        self._restraint_space_calc = RestraintSpace(
            restraint_space, self.restraints, self.ligand.center
        )
        accessible_interaction_space = Volume.zeros_like(rcore, dtype=np.int32)
        self._ais_calc = AccessibleInteractionSpace(
            accessible_interaction_space, self._interaction_space_calc,
            self._restraint_space_calc
        )
        if self.options.occupancy_analysis:
            self._occupancy_space = OccupancySpace(self._ais_calc)

        if self.options.interaction_analysis:
            space = Volume.zeros_like(rcore)
            identifyer = InterfaceResidueIdentifyer(space)
            pruned_receptor = identifyer(self.receptor)
            pruned_receptor.tofile('pruned_receptor.pdb')
            pruned_ligand = identifyer(self.ligand)
            pruned_ligand.tofile('pruned_ligand.pdb')
            self._interaction_analyzer = InteractionAnalyzer(
                pruned_receptor, pruned_ligand, self._ais_calc,
            )
        self._initialized = True

    def __call__(self, rotmat, weight=1):

        if not self._initialized:
            self.initialize()

        self._volumizer.generate_lcore(rotmat)
        self._interaction_space_calc()
        self._restraint_space_calc(rotmat)
        self._ais_calc(weight=weight)
        if self.options.save:
            save_id = self.options.save_id
            fname = self.options.directory(
                'ais{}_{:d}.mrc'.format(save_id, self.self._n)
            )
            self._ais_calc.consistent_space.tofile(fname)
            self._n += 1

        if self.options.occupancy_analysis:
            self._occupancy_space(weight=weight)
        if self.options.interaction_analysis:
            self._interaction_analyzer(rotmat, weight=weight)

        try:
            self._counter.increment()
        except AttributeError:
            pass

    @property
    def consistent_complexes(self):
        return self._ais_calc.consistent_complexes()

    @property
    def violation_matrix(self):
        return self._ais_calc.violation_matrix()

    @property
    def max_consistent(self):
        return self._ais_calc.max_consistent

    def tofile(self, normalize=True, prefix='', suffix=''):
        directory = self.options.directory
        # Write consistent complexes to file
        fname = directory(prefix + 'consistent_complexes' + suffix + '.txt')
        with open(fname, 'w') as f:
            for n, consistent_complexes in enumerate(self.consistent_complexes):
                f.write('{} {:.0f}\n'.format(n, consistent_complexes))

        # Write violation matrix to file
        fname = directory(prefix + 'violation_matrix' + suffix + '.txt')
        violation_matrix = self.violation_matrix
        if normalize:
            consistent_complexes = self.consistent_complexes.reshape(-1, 1)
            violation_matrix /= consistent_complexes
        np.savetxt(fname, violation_matrix, fmt='%.4f')

        fname = directory(prefix + 'consistent_space' + suffix + '.mrc')
        self.max_consistent.tofile(fname)

        # Write number of correlated consistent restraints
        nrestraints = len(self.restraints)
        fname = directory(prefix + 'restraint_correlations' + suffix + '.txt')
        with open(fname, 'w') as f:
            ais = self._ais_calc
            # Only print correlations among restraints consistent with at least N - 10
            start = max(nrestraints - 10, 0)
            for nconsistent_restraints in xrange(start, nrestraints + 1):
                mask = ais._consistent_restraints == nconsistent_restraints
                cons_perm = ais._consistent_permutations[mask]
                consistent_sets = [bin(x) for x in ais._indices[mask]]
                for cset, sperm in izip(consistent_sets, cons_perm):
                    restraint_flags = ' '.join(list(
                        ('{:0>' + str(nrestraints) + 'd}').format(int(cset[2:]))
                    )[::-1])
                    if sperm != 0:
                        f.write('{} {:.0f}\n'.format(restraint_flags, sperm))
                f.write('#' * 30 + '\n')

        # Write out occupancy maps
        if self.options.occupancy_analysis:
            occs = self._occupancy_space
            iterator = izip(occs.nconsistent, occs.spaces)
            for n, space in iterator:
                fname = directory(prefix + 'occ_{:}' + suffix + '.mrc')
                space.tofile(directory(fname.format(n)))

        # Write out interaction analysis
        if self.options.interaction_analysis:
            ia = self._interaction_analyzer
            # Loop over receptor and ligand interactions
            iterator = [[ia._receptor_interactions, ia._receptor_residues, 'receptor_interactions'],
                        [ia._ligand_interactions, ia._ligand_residues, 'ligand_interactions']]
            for interactions, residues, fname in iterator:
                interactions = interactions.T
                # Normalize interactions over the size of the interaction space
                if normalize:
                    consistent_complexes = self.consistent_complexes[1:].reshape(1, -1)
                    interactions /= consistent_complexes
                fname = directory(prefix + fname + suffix + '.txt')
                with open(directory(fname), 'w') as f:
                    line = '{} ' + ' '.join(['{:.3f}'] * nrestraints) + '\n'
                    for resid, ni in izip(residues, interactions):
                        f.write(line.format(resid, *ni))


class _Counter(object):
    """Thread-safe counter object to follow DisVis progress"""

    def __init__(self):
        self.val = RawValue('i', 0)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class MPDisVis(object):

    """Multiprocessor interface to DisVis."""

    def __init__(self, receptor, ligand, restraints, options, rotmats, weights):
        self.receptor = receptor
        self.ligand = ligand
        self.restraints = restraints
        self.nrestraints = len(restraints)
        self.options = options
        self.rotmats = rotmats
        self.weights = weights

    @staticmethod
    def _run_disvis_instance(receptor, ligand, restraints, options, rotmats, weights,
                             counter, job_id):
        options.save_id = job_id
        disvis = DisVis(receptor, ligand, restraints, options, counter=counter)
        # Loop over rotations and weights
        for w, rotmat in izip(weights, rotmats):
            disvis(rotmat, weight=w)
        # Write results to file
        suffix = '_{}'.format(job_id)
        disvis.tofile(normalize=False, prefix='_', suffix=suffix)

    def _combine(self):
        """Combine results of each DisVis instance"""

        # Read in all files and combine them
        directory = self.options.directory
        nrestraints = self.nrestraints
        self.restraint_correlations = OrderedDict()
        self.consistent_complexes = np.zeros(nrestraints + 1)
        self.violation_matrix = np.zeros((nrestraints + 1, nrestraints))
        for n in xrange(self.options.nprocessors):
            fname = directory('_consistent_complexes_{}.txt'.format(n))
            self.consistent_complexes += np.loadtxt(fname, usecols=1)

            fname = directory('_violation_matrix_{}.txt'.format(n))
            self.violation_matrix += np.loadtxt(fname)

            fname = directory('_consistent_space_{}.mrc'.format(n))
            max_consistent = Volume.fromfile(fname)
            try:
                np.maximum(self.max_consistent, max_consistent, self.max_consistent)
            except:
                self.max_consistent = max_consistent

            fname = directory("_restraint_correlations_{}.txt".format(n))
            with open(fname) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    words = line.split()
                    key = ' '.join(words[:-1])
                    value = int(words[-1])
                    try:
                        self.restraint_correlations[key] += value
                    except KeyError:
                        self.restraint_correlations[key] = value

            if self.options.occupancy_analysis:
                self.occupancy_spaces = {}
                fnames = glob(directory('_occ_*_{}.mrc').format(n))
                for fname in fnames:
                    _, _, nconsistent, job_id = os.path.split(fname)[1].split('_')
                    occ_space = Volume.fromfile(fname)
                    try:
                        self.occupancy_spaces[nconsistent].array += occ_space
                    except:
                        self.occupancy_spaces[nconsistent] = occ_space

            if self.options.interaction_analysis:
                fname = directory('_receptor_interactions_{}.txt'.format(n))
                data = np.loadtxt(fname, dtype=np.str_)
                self.receptor_residues = data[:, 0]
                interactions = data[:, 1:].astype(np.float64)
                try:
                    self.receptor_interactions += interactions
                except:
                    self.receptor_interactions = interactions
                fname = directory('_ligand_interactions_{}.txt'.format(n))
                data = np.loadtxt(fname, dtype=np.str_)
                self.ligand_residues = data[:, 0]
                interactions = data[:, 1:].astype(np.float64)
                try:
                    self.ligand_interactions += interactions
                except:
                    self.ligand_interactions = interactions

        # Remove all files
        files_to_remove = glob(directory('_*'))
        for fn in files_to_remove:
            os.remove(fn)

    def run(self):
        processes = []
        # Divide rotations in equal sized blocks to divide over each DisVis instance
        nrots = self.rotmats.shape[0]
        nrots_per_job = nrots // self.options.nprocessors
        counter = _Counter()
        for n in xrange(self.options.nprocessors):
            init_rot = n * nrots_per_job
            end_rot = min(init_rot + nrots_per_job, nrots)
            rotmats = self.rotmats[init_rot:end_rot]
            weights = self.weights[init_rot:end_rot]
            args = (self.receptor, self.ligand, self.restraints, self.options,
                    rotmats, weights, counter, n)
            process = Process(target=self._run_disvis_instance, args=args)
            processes.append(process)

        time0 = time.time()
        for p in processes:
            p.start()

        # Report on progress
        if self.options.verbose:
            line = '{n} / {total}  time passed: {passed:.0f}s  eta: {eta:.0f}s       \r'
            while True:
                n = counter.value()
                percentage = (n + 1) / float(nrots) * 100
                time_passed = time.time() - time0
                eta = (time_passed) / percentage * (100 - percentage)
                sys.stdout.write(line.format(n=n, total=nrots, passed=time_passed, eta=eta))
                sys.stdout.flush()
                if n >= nrots:
                    sys.stdout.write('\n')
                    break
                time.sleep(0.5)

        for p in processes:
            p.join()
        self._combine()

    def tofile(self):
        directory = self.options.directory
        # Write consistent complexes to file
        fname = directory('consistent_complexes.txt')
        with open(fname, 'w') as f:
            for n, consistent_complexes in enumerate(self.consistent_complexes):
                f.write('{} {:.0f}\n'.format(n, consistent_complexes))

        # Write violation matrix to file
        fname = directory('violation_matrix.txt')
        violation_matrix = self.violation_matrix
        consistent_complexes = self.consistent_complexes.reshape(-1, 1)
        violation_matrix /= consistent_complexes
        np.savetxt(fname, violation_matrix, fmt='%.4f')

        # Write max consistent space
        fname = directory('consistent_space.mrc')
        self.max_consistent.tofile(fname)

        fname = directory('restraint_correlations.txt')
        with open(fname, 'w') as f:
            for key, value in self.restraint_correlations.iteritems():
                f.write('{} {}\n'.format(key, value))

        # Write out occupancy maps
        if self.options.occupancy_analysis:
            for n, space in self.occupancy_spaces.iteritems():
                fname = directory('occ_{:}.mrc')
                space.tofile(directory(fname.format(n)))

        # Write out interaction analysis
        if self.options.interaction_analysis:
            # Loop over receptor and ligand interactions
            iterator = [[self.receptor_interactions, self.receptor_residues, 'receptor_interactions'],
                        [self.ligand_interactions, self.ligand_residues, 'ligand_interactions']]
            for interactions, residues, fname in iterator:
                # Normalize interactions over the size of the interaction space
                consistent_complexes = self.consistent_complexes[1:].reshape(1, -1)
                interactions /= consistent_complexes
                fname = directory(fname + '.txt')
                with open(directory(fname), 'w') as f:
                    line = '{} ' + ' '.join(['{:.3f}'] * self.nrestraints) + '\n'
                    for resid, ni in izip(residues, interactions):
                        f.write(line.format(resid, *ni))


def main():
    args = parse_args()
    args.directory.mkdir()

    # Set up logger
    logging_fname = args.directory('disvis.log')
    logging.basicConfig(filename=logging_fname, level=logging.INFO)
    logger.info(' '.join(sys.argv))
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console_out)

    logger.info("Reading receptor and ligand.")
    receptor = PDB.fromfile(args.receptor)
    ligand = PDB.fromfile(args.ligand)

    # Get the restraint
    logger.info("Parsing restraints.")
    restraint_parser = RestraintParser()
    restraints = []
    for line in args.restraints:
        out = restraint_parser.parse_line(line)
        if out is not None:
            rsel, lsel, min_dis, max_dis = out
            receptor_selection = []
            for sel in rsel:
                rpart = receptor
                for key, value in sel:
                    rpart = rpart.select(key, value)
                receptor_selection.append(rpart)
            lpart = ligand
            ligand_selection = []
            for sel in lsel:
                lpart = ligand
                for key, value in sel:
                    lpart = lpart.select(key, value)
                ligand_selection.append(lpart)
            restraints.append(Restraint(
                receptor_selection, ligand_selection, min_dis, max_dis)
            )

    options = DisVisOptions.fromargs(args)

    quat, weights, alpha = proportional_orientations(args.angle)
    rotations = quat_to_rotmat(quat)
    nrot = rotations.shape[0]
    time0 = time.time()
    logger.info("Starting search.")
    if args.nprocessors <= 1:
        disvis = DisVis(receptor, ligand, restraints, options)
        for n, (rotmat, weight) in enumerate(izip(rotations, weights)):
            if args.verbose:
                sys.stdout.write('{:>6d} {:>6d}\r'.format(n, nrot))
                sys.stdout.flush()
            disvis(rotmat, weight=weight)
    else:
        disvis = MPDisVis(receptor, ligand, restraints, options, rotations, weights)
        disvis.run()
    logger.info('Time: {:.2f} s'.format(time.time() - time0))

    disvis.tofile()

