"""Quantify and visualize the information content of distance restraints."""

from argparse import ArgumentParser
import time
from itertools import izip
import sys
import logging
from multiprocessing import RawValue, Lock, Process, cpu_count

import numpy as np

from .pdb import PDB
from .volume import Volume, Volumizer
from .spaces import (InteractionSpace, RestraintSpace, Restraint,
                     AccessibleInteractionSpace, OccupancySpace, InteractionAnalyzer)
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
    save_id = ''
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
            self._interaction_analyzer = InteractionAnalyzer(
                    self.receptor, self.ligand, self._ais_calc, self._volumizer
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
            fname = self.options.directory('ais{}_{:d}.mrc'.format(save_id, self.self._n))
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
        # TODO normalize violation matrix
        np.savetxt(fname, self.violation_matrix, fmt='%.2f')
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
                    restraint_flags = ' '.join(list(('{:0>' + str(nrestraints) + 'd}').format(int(cset[2:])))[::-1])
                    if sperm != 0:
                        f.write('{} {:.0f}\n'.format(restraint_flags, sperm))
                f.write('#' * 30 + '\n')

        # Write out occupancy maps
        if self.options.occupancy_analysis:
            occs = self._occupancy_space
            iterator = izip(occs.nconsistent, occs.spaces)
            for n, space in iterator:
                space.tofile(directory('occ_{:}.mrc'.format(n)))

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
                    line = '{} ' + ' '.join(['{:.3f}'] * (nrestraints - 1)) + '\n'
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
        pass

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
        line = '{n} / {total}  time passed: {passed:.0f}s  eta: {eta:.0f}s       \r'
        while True:
            n = counter.value()
            percentage = (n + 1) / float(nrots) * 100
            time_passed = time.time() - time0
            eta = (time_passed) / percentage * (100 - percentage)
            sys.stdout.write(line.format(n=n, total=nrots, passed=time_passed, eta=eta))
            sys.stdout.flush()
            if n >= nrots:
                break
            time.sleep(0.5)

        for p in processes:
            p.join()
        self._combine()

    def tofile(self):
        pass


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

