# coding=utf-8
"""
This module defines the Symmetry Function descriptor.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
import time

from ase.build import bulk
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from itertools import chain
from collections import Counter
from typing import List
from sklearn.model_selection import ParameterGrid
from os.path import dirname
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_elements_from_kbody_term(kbody_term: str) -> List[str]:
    """
    Extract elements from a k-body term.
    """
    sel = [0]
    for i in range(len(kbody_term)):
        if kbody_term[i].isupper():
            sel.append(i + 1)
        else:
            sel[-1] += 1
    atoms = []
    for i in range(len(sel) - 1):
        atoms.append(kbody_term[sel[i]: sel[i + 1]])
    return atoms


def compute_dimension(all_kbody_terms: List[str], n_etas, n_omegas, n_betas,
                      n_gammas, n_zetas):
    """
    Compute the overall dimension of G and the starting offsets of the k-body
    terms.
    """
    total_dim = 0
    kbody_sizes = []
    for kbody_term in all_kbody_terms:
        if len(get_elements_from_kbody_term(kbody_term)) == 2:
            n = n_etas * n_omegas
        else:
            n = n_gammas * n_betas * n_zetas
        total_dim += n
        kbody_sizes.append(n)
    return total_dim, kbody_sizes


def get_kbody_terms(elements: List[str], angular=False):
    """
    Return ordered k-body terms (k=2 or k=2,3 if angular is True).
    """
    elements = sorted(list(set(elements)))
    n = len(elements)
    kbody_terms = {}
    for i in range(n):
        kbody_term = "{}{}".format(elements[i], elements[i])
        kbody_terms[elements[i]] = [kbody_term]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            kbody_term = "{}{}".format(elements[i], elements[j])
            kbody_terms[elements[i]].append(kbody_term)
    if angular:
        for i in range(n):
            center = elements[i]
            for j in range(n):
                for k in range(j, n):
                    suffix = "".join(sorted([elements[j], elements[k]]))
                    kbody_term = "{}{}".format(center, suffix)
                    kbody_terms[elements[i]].append(kbody_term)
    all_terms = [
        x for x in chain(*[kbody_terms[element] for element in elements])]
    return all_terms, kbody_terms, elements


def get_max_occurs(list_of_atoms: List[Atoms]):
    """
    Find all N_{el}^{max} of the dataset.
    """
    max_occurs = Counter()
    for atoms in list_of_atoms:
        c = Counter(atoms.get_chemical_symbols())
        for e, n in c.items():
            max_occurs[e] = max(n, max_occurs[e])
    return max_occurs


def cosine_cutoff(r: tf.Tensor, rc: float, name=None):
    """
    The cosine cutoff function.
    """
    with tf.name_scope(name):
        r = tf.convert_to_tensor(r, name='r')
        rc = tf.convert_to_tensor(rc, dtype=r.dtype, name="rc")
        z = 0.5 * (tf.cos(tf.minimum(r / rc, 1.0) * np.pi, name='cos') + 1.0)
        return z


class VirtualAtomMap:
    """
    The core of the Virtual-Atom Approach.
    """

    REAL_ATOM_START = 1

    def __init__(self, max_occurs: Counter, symbols: List[str]):
        istart = VirtualAtomMap.REAL_ATOM_START
        self.max_occurs = max_occurs
        self.symbols = symbols
        self.max_vap_n_atoms = sum(max_occurs.values()) + istart
        elements = sorted(max_occurs.keys())
        offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
        offsets = np.insert(offsets, 0, 0)
        delta = Counter()
        index_map = {}
        mask = np.zeros(self.max_vap_n_atoms, dtype=bool)
        for i, symbol in enumerate(symbols):
            i_ele = elements.index(symbol)
            i_old = i + istart
            i_new = offsets[i_ele] + delta[symbol] + istart
            index_map[i_old] = i_new
            delta[symbol] += 1
            mask[i_new] = True
        reverse_map = {v: k - 1 for k, v in index_map.items()}
        index_map[0] = 0
        reverse_map[0] = -1
        self.atom_masks = mask
        self.index_map = index_map
        self.reverse_map = reverse_map
        self.splits = np.array([1, ] + [max_occurs[e] for e in elements])

    def map_array_to_gsl(self, array: np.ndarray):
        """
        Map a local array to its GSL form.
        """
        rank = np.ndim(array)
        if rank == 2:
            array = array[np.newaxis, ...]
        elif rank <= 1 or rank > 3:
            raise ValueError("The rank should be 2 or 3")
        if array.shape[1] == len(self.symbols):
            array = np.insert(
                array, 0, np.asarray(0, dtype=array.dtype), axis=1)
        else:
            shape = (array.shape[0], len(self.symbols), array.shape[2])
            raise ValueError(f"The shape should be {shape}")
        indices = []
        for i in range(self.max_vap_n_atoms):
            indices.append(self.reverse_map.get(i, -1) + self.REAL_ATOM_START)
        output = array[:, indices]
        if rank == 2:
            output = np.squeeze(output, axis=0)
        return output

    def reverse_map_gsl_to_local(self, array: np.ndarray):
        """
        Reverse map a GSL array to its local form.
        """
        rank = np.ndim(array)
        if rank == 2:
            array = array[np.newaxis, ...]
        assert array.shape[1] == self.max_vap_n_atoms
        istart = self.REAL_ATOM_START
        indices = []
        for i in range(istart, istart + len(self.symbols)):
            indices.append(self.index_map[i])
        output = array[:, indices]
        if rank == 2:
            output = np.squeeze(output, axis=0)
        return output


class SymmetryFunction:
    """
    A tensorflow based implementation of Behler-Parinello's SymmetryFunction
    descriptor.
    """
    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc, elements, eta=np.array([0.05, 4.0, 20.0, 80.0]),
                 omega=np.array([0.0]), beta=np.array([0.005, ]),
                 gamma=np.array([1.0, -1.0]), zeta=np.array([1.0, 4.0]),
                 angular=True, periodic=True):
        """
        Initialization method.
        """
        all_kbody_terms, kbody_terms, elements = \
            get_kbody_terms(elements, angular=angular)
        ndim, kbody_sizes = compute_dimension(
            all_kbody_terms, len(eta), len(omega), len(beta), len(gamma),
            len(zeta))
        self.rc = rc
        self.all_kbody_terms = all_kbody_terms
        self.kbody_terms = kbody_terms
        self.elements = elements
        self.n_elements = len(elements)
        self.periodic = periodic
        self.angular = angular
        self.kbody_sizes = kbody_sizes
        self.ndim = ndim
        self.kbody_index = {kbody_term: self.all_kbody_terms.index(kbody_term)
                            for kbody_term in self.all_kbody_terms}
        self.offsets = np.insert(np.cumsum(kbody_sizes), 0, 0)
        self.radial_indices_grid = ParameterGrid({
            'eta': np.arange(len(eta), dtype=int),
            'omega': np.arange(len(omega), dtype=int)})
        self.angular_indices_grid = ParameterGrid({
            'beta': np.arange(len(beta), dtype=int),
            'gamma': np.arange(len(gamma), dtype=int),
            'zeta': np.arange(len(zeta), dtype=int)})
        self.initial_values = {'eta': eta, 'omega': omega, 'gamma': gamma,
                               'beta': beta, 'zeta': zeta}

    @staticmethod
    def get_pbc_displacements(shift, cells, dtype=tf.float32):
        """
        Compute D_{pbc}.
        """
        return tf.matmul(shift, cells, name='displacements')

    def get_rij(self, positions, cells, ilist, jlist, shift, name):
        """
        Return r^{GSL} and D^{GSL} of Fig(3).
        """
        with tf.name_scope(name):
            dtype = positions.dtype
            Ri = self.gather_fn(positions, ilist, 'Ri')
            Rj = self.gather_fn(positions, jlist, 'Rj')
            Dij = tf.subtract(Rj, Ri, name='Dij')
            if self.periodic:
                pbc = self.get_pbc_displacements(shift, cells, dtype=dtype)
                Dij = tf.add(Dij, pbc, name='pbc')
            # By adding `eps` to the reduced sum NaN can be eliminated.
            with tf.name_scope("safe_norm"):
                eps = tf.constant(1e-8, dtype=dtype, name='eps')
                rij = tf.sqrt(tf.reduce_sum(
                    tf.square(Dij, name='Dij2'), axis=-1) + eps)
                return rij, Dij

    def get_v2g_map(self, features: dict, prefix: str):
        """
        Return the base v2g map.
        """
        return tf.identity(features[f"{prefix}.v2g_map"], name='v2g_map')

    @staticmethod
    def get_v2g_map_delta(tau):
        """
        Return the tau-th `v2g_map`.
        """
        return tf.constant([0, tau], dtype=tf.int32, name='delta')

    def get_g_shape(self, features: dict):
        """
        Return the shape of the descriptor matrix.
        """
        return [features['n_atoms_vap'], self.ndim]

    def get_g2_op_for_tau(self, shape, tau, r, rc2, fc_r, base_v2g_map):
        """
        Return the Op to compute G2(tau) using tau-th (`eta`, `omega`) pair.
        """
        with tf.name_scope(f"Grid{tau}"):
            grid = self.radial_indices_grid[tau]
            etai = grid['eta']
            omegai = grid['omega']
            eta = tf.constant(self.initial_values['eta'][etai], r.dtype)
            omega = tf.constant(self.initial_values['omega'][omegai], r.dtype)
            delta = self.get_v2g_map_delta(tau)
            r2c = tf.math.truediv(tf.square(r - omega), rc2, name='r2c')
            v = tf.exp(-tf.multiply(eta, r2c, 'eta_r2c')) * fc_r
            v2g_map_tau = tf.add(base_v2g_map, delta, f'v2g_map_{tau}')
            return tf.scatter_nd(v2g_map_tau, v, shape, f"g{tau}")

    def get_g2_op(self, features: dict):
        """
        The implementation of Behler's G2 symmetry function.
        """
        with tf.variable_scope("G2"):
            r = self.get_rij(features['positions'],
                             features['cells'],
                             features['g2.ilist'],
                             features['g2.jlist'],
                             features['g2.shift'],
                             name='rij')[0]
            rc2 = tf.constant(self.rc**2, dtype=r.dtype, name='rc2')
            fc_r = cosine_cutoff(r, rc=self.rc, name='fc_r')
            base_v2g_map = self.get_v2g_map(features, prefix='g2')
            shape = self.get_g_shape(features)
            values = []
            for tau in range(len(self.radial_indices_grid)):
                values.append(
                    self.get_g2_op_for_tau(
                        shape, tau, r, rc2, fc_r, base_v2g_map))
            return tf.add_n(values, name='g')

    def get_g4_op_for_tau(self, shape, tau: int, cos_theta, r2c, fc_r,
                          base_v2g_map):
        """
        Return the Op to compute G4(tau) using tau-th (`beta`, `gamma`, `zeta`)
        combination.
        """
        with tf.name_scope(f"Grid{tau}"):
            grid = self.angular_indices_grid[tau]
            betai = grid['beta']
            gammai = grid['gamma']
            zetai = grid['zeta']
            beta = tf.constant(self.initial_values['beta'][betai], r2c.dtype)
            gamma = tf.constant(self.initial_values['gamma'][gammai], r2c.dtype)
            zeta = tf.constant(self.initial_values['zeta'][zetai], r2c.dtype)
            delta = self.get_v2g_map_delta(tau)
            one = tf.constant(1.0, dtype=r2c.dtype, name='one')
            two = tf.constant(2.0, dtype=r2c.dtype, name='two')
            gt = tf.math.multiply(gamma, cos_theta, name='gt')
            gt1 = tf.add(gt, one, name='gt1')
            gt1z = tf.pow(gt1, zeta)
            z1 = tf.math.subtract(one, zeta, name='z1')
            z12 = tf.pow(two, z1)
            c = tf.math.multiply(gt1z, z12, name='c')
            v = tf.multiply(c * tf.exp(-beta * r2c), fc_r, f'v_{tau}')
            v2g_map_tau = tf.add(base_v2g_map, delta, name=f'v2g_map_{tau}')
            return tf.scatter_nd(v2g_map_tau, v, shape, f'g{tau}')

    def get_g4_op(self, features: dict):
        """
        The implementation of Behler's angular symmetry function.
        """
        with tf.variable_scope("G4"):
            rij = self.get_rij(features['positions'],
                               features['cells'],
                               features['g4.ij.ilist'],
                               features['g4.ij.jlist'],
                               features['g4.shift.ij'],
                               name='rij')[0]
            rik = self.get_rij(features['positions'],
                               features['cells'],
                               features['g4.ik.ilist'],
                               features['g4.ik.klist'],
                               features['g4.shift.ik'],
                               name='rik')[0]
            rjk = self.get_rij(features['positions'],
                               features['cells'],
                               features['g4.jk.jlist'],
                               features['g4.jk.klist'],
                               features['g4.shift.jk'],
                               name='rjk')[0]
            rij2 = tf.square(rij, name='rij2')
            rik2 = tf.square(rik, name='rik2')
            rjk2 = tf.square(rjk, name='rjk2')
            rc2 = tf.constant(self.rc**2, dtype=rij.dtype, name='rc2')
            r2 = tf.add_n([rij2, rik2, rjk2], name='r2')
            r2c = tf.math.truediv(r2, rc2, name='r2_rc2')
            with tf.name_scope("CosTheta"):
                upper = tf.subtract(rij2 + rik2, rjk2, name='upper')
                lower = tf.multiply(2.0 * rij, rik, name='lower')
                cos_theta = tf.math.truediv(upper, lower, name='theta')
            with tf.name_scope("Cutoff"):
                fc_rij = cosine_cutoff(rij, rc=self.rc, name='fc_rij')
                fc_rik = cosine_cutoff(rik, rc=self.rc, name='fc_rik')
                fc_rjk = cosine_cutoff(rjk, rc=self.rc, name='fc_rjk')
                fc_r = tf.multiply(fc_rij, fc_rik * fc_rjk, 'fc_r')
            base_v2g_map = self.get_v2g_map(features, prefix='g4')
            shape = self.get_g_shape(features)
            values = []
            for tau in range(len(self.angular_indices_grid)):
                values.append(
                    self.get_g4_op_for_tau(
                        shape, tau, cos_theta, r2c, fc_r, base_v2g_map))
            return tf.add_n(values, name='g')

    def get_row_split_sizes(self, features: dict):
        """
        Return the sizes of the row-wise splitted subsets of `descriptors`.
        """
        return features['row_splits']

    @staticmethod
    def get_row_split_axis():
        """
        Return the axis to row-wise split `descriptors`.
        """
        return 0

    def get_column_split_sizes(self):
        """
        Return the sizes of the column-wise splitted subsets of `descriptors`.
        """
        column_splits = {}
        for i, element in enumerate(self.elements):
            column_splits[element] = [len(self.elements), i]
        return column_splits

    @staticmethod
    def get_column_split_axis():
        """
        Return the axis to column-wise split `g`.
        """
        return 1

    def split_descriptors(self, descriptors, features: dict):
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            row_split_sizes = self.get_row_split_sizes(features)
            row_split_axis = self.get_row_split_axis()
            column_split_sizes = self.get_column_split_sizes()
            column_split_axis = self.get_column_split_axis()
            splits = tf.split(
                descriptors, row_split_sizes, axis=row_split_axis,
                name='rows')[1:]
            atom_masks = tf.split(
                features['atom_masks'], row_split_sizes, axis=row_split_axis,
                name='atom_masks')[1:]
            if len(self.elements) > 1:
                # Further split the element arrays to remove redundant zeros
                blocks = []
                for i in range(len(splits)):
                    element = self.elements[i]
                    size_splits, idx = column_split_sizes[element]
                    block = tf.split(splits[i],
                                     size_splits,
                                     axis=column_split_axis,
                                     name='{}_block'.format(element))[idx]
                    blocks.append(block)
            else:
                blocks = splits
            return dict(zip(self.elements, zip(blocks, atom_masks)))

    def build_graph(self, features: dict):
        """
        Get the tensorflow based computation graph of the Symmetry Function.
        """
        with tf.variable_scope("Behler"):
            descriptors = self.get_g2_op(features)
            if self.angular:
                descriptors += self.get_g4_op(features)
        return self.split_descriptors(descriptors, features)


class BatchSymmetryFunction(SymmetryFunction):
    """
    A special implementation of Behler-Parinello's Symmetry Function for batch
    training and evaluations.
    """

    gather_fn = staticmethod(tf.batch_gather)

    def __init__(self, rc, max_occurs: Counter, nij_max: int, nijk_max: int,
                 batch_size: int, eta=np.array([0.05, 4.0, 20.0, 80.0]),
                 omega=np.array([0.0]), beta=np.array([0.005, ]),
                 gamma=np.array([1.0, -1.0]), zeta=np.array([1.0, 4.0]),
                 angular=True, periodic=True):
        """
        Initialization method.
        """
        elements = sorted(list(max_occurs.keys()))

        super(BatchSymmetryFunction, self).__init__(
            rc=rc, elements=elements, eta=eta, beta=beta, gamma=gamma,
            zeta=zeta, omega=omega, angular=angular, periodic=periodic)
        self._max_occurs = max_occurs
        self._max_n_atoms = sum(max_occurs.values())
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._batch_size = batch_size

    @staticmethod
    def get_pbc_displacements(shift, cells, dtype=tf.float32):
        """
        Compute r^{GSL} and D^{GSL} in the training phase.
        """
        with tf.name_scope("Einsum"):
            shift = tf.convert_to_tensor(shift, dtype=dtype, name='shift')
            cells = tf.convert_to_tensor(cells, dtype=dtype, name='cells')
            return tf.einsum('ijk,ikl->ijl', shift, cells, name='displacements')

    def get_g_shape(self, _):
        """
        Return the shape of the descriptors.
        """
        n_atoms_vap = self._max_n_atoms + 1
        return [self._batch_size, n_atoms_vap, self.ndim]

    def get_v2g_map_batch_indexing_matrix(self, prefix='g2'):
        """
        Return an `int32` matrix of shape `[batch_size, ndim, 3]` to rebuild the
        batch indexing of a `v2g_map`.
        """
        if prefix == 'g2':
            ndim = self._nij_max
        else:
            ndim = self._nijk_max
        indexing_matrix = np.zeros((self._batch_size, ndim, 3), dtype=np.int32)
        for i in range(self._batch_size):
            indexing_matrix[i] += [i, 0, 0]
        return indexing_matrix

    @staticmethod
    def get_v2g_map_delta(tau):
        return tf.constant([0, 0, tau], tf.int32, name='delta')

    def get_v2g_map(self, features: dict, prefix="g2"):
        """
        Return the Op to get `v2g_map`. In the batch implementation, `v2g_map`
        has a shape of `[batch_size, ndim, 3]` and the first axis represents the
        local batch indices.
        """
        indexing = self.get_v2g_map_batch_indexing_matrix(prefix=prefix)
        return tf.add(features[f"{prefix}.v2g_map"], indexing, name='v2g_map')

    def get_row_split_sizes(self, _):
        row_splits = [1, ]
        for i, element in enumerate(self.elements):
            row_splits.append(self._max_occurs[element])
        return row_splits

    @staticmethod
    def get_row_split_axis():
        return 1

    @staticmethod
    def get_column_split_axis():
        return 2


def get_g2_map(atoms: Atoms,
               rc: float,
               nij_max: int,
               interactions: list,
               vap: VirtualAtomMap,
               offsets: np.ndarray,
               for_prediction=False,
               print_time=False):
    if for_prediction:
        iaxis = 0
    else:
        iaxis = 1
    g2_map = np.zeros((nij_max, iaxis + 2), dtype=np.int32)
    tlist = np.zeros(nij_max, dtype=np.int32)
    symbols = atoms.get_chemical_symbols()
    tic = time.time()
    ilist, jlist, n1 = neighbor_list('ijS', atoms, rc)
    if print_time:
        print(f"* ASE neighbor time: {time.time() - tic}")
    nij = len(ilist)
    tlist.fill(0)
    for i in range(nij):
        symboli = symbols[ilist[i]]
        symbolj = symbols[jlist[i]]
        tlist[i] = interactions.index('{}{}'.format(symboli, symbolj))
    ilist = np.pad(ilist + 1, (0, nij_max - nij), 'constant')
    jlist = np.pad(jlist + 1, (0, nij_max - nij), 'constant')
    n1 = np.pad(n1, ((0, nij_max - nij), (0, 0)), 'constant')
    n1 = n1.astype(np.float32)
    for count in range(len(ilist)):
        if ilist[count] == 0:
            break
        ilist[count] = vap.index_map[ilist[count]]
        jlist[count] = vap.index_map[jlist[count]]
    g2_map[:, iaxis + 0] = ilist
    g2_map[:, iaxis + 1] = offsets[tlist]
    return {"g2.v2g_map": g2_map, "g2.ilist": ilist, "g2.jlist": jlist,
            "g2.shift": n1}


def get_g4_map(atoms: Atoms,
               g2_map: dict,
               interactions: list,
               offsets: np.ndarray,
               vap: VirtualAtomMap,
               nijk_max: int,
               for_prediction=True):
    if for_prediction:
        iaxis = 0
    else:
        iaxis = 1
    g4_map = np.zeros((nijk_max, iaxis + 2), dtype=np.int32)
    ijk = np.zeros((nijk_max, 3), dtype=np.int32)
    n1 = np.zeros((nijk_max, 3), dtype=np.float32)
    n2 = np.zeros((nijk_max, 3), dtype=np.float32)
    n3 = np.zeros((nijk_max, 3), dtype=np.float32)
    symbols = atoms.get_chemical_symbols()
    indices = {}
    vectors = {}
    for i, atom_gsl_i in enumerate(g2_map["g2.ilist"]):
        if atom_gsl_i == 0:
            break
        if atom_gsl_i not in indices:
            indices[atom_gsl_i] = []
            vectors[atom_gsl_i] = []
        indices[atom_gsl_i].append(g2_map["g2.jlist"][i])
        vectors[atom_gsl_i].append(g2_map["g2.shift"][i])
    count = 0
    for atom_gsl_i, nl in indices.items():
        atom_local_i = vap.reverse_map[atom_gsl_i]
        symboli = symbols[atom_local_i]
        prefix = '{}'.format(symboli)
        for j in range(len(nl)):
            atom_vap_j = nl[j]
            atom_local_j = vap.reverse_map[atom_vap_j]
            symbolj = symbols[atom_local_j]
            for k in range(j + 1, len(nl)):
                atom_vap_k = nl[k]
                atom_local_k = vap.reverse_map[atom_vap_k]
                symbolk = symbols[atom_local_k]
                interaction = '{}{}'.format(
                    prefix, ''.join(sorted([symbolj, symbolk])))
                ijk[count] = atom_gsl_i, atom_vap_j, atom_vap_k
                n1[count] = vectors[atom_gsl_i][j]
                n2[count] = vectors[atom_gsl_i][k]
                n3[count] = vectors[atom_gsl_i][k] - vectors[atom_gsl_i][j]
                index = interactions.index(interaction)
                g4_map[count, iaxis + 0] = atom_gsl_i
                g4_map[count, iaxis + 1] = offsets[index]
                count += 1
    return {"g4.v2g_map": g4_map,
            "g4.ij.ilist": ijk[:, 0], "g4.ij.jlist": ijk[:, 1],
            "g4.ik.ilist": ijk[:, 0], "g4.ik.klist": ijk[:, 2],
            "g4.jk.jlist": ijk[:, 1], "g4.jk.klist": ijk[:, 2],
            "g4.shift.ij": n1, "g4.shift.ik": n2, "g4.shift.jk": n3}


def get_nij_and_nijk(atoms, rc, angular=False):
    ilist, jlist = neighbor_list('ij', atoms, cutoff=rc)
    nij = len(ilist)
    if angular:
        nl = {}
        for i, atomi in enumerate(ilist):
            if atomi not in nl:
                nl[atomi] = []
            nl[atomi].append(jlist[i])
        nijk = 0
        for atomi, nlist in nl.items():
            n = len(nlist)
            nijk += (n - 1 + 1) * (n - 1) // 2
    else:
        nijk = 0
    return nij, nijk


def bytes_feature(value):
    """
    Convert the `value` to Protobuf bytes.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """
    Convert the `value` to Protobuf float32.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int32_feature(value):
    """
    Convert the `value` to Protobuf int64.
    """
    return bytes_feature(np.int32(value).tostring())


def test_cache():
    atoms = bulk('Ni', crystalstructure='fcc', cubic=True)
    symbols = atoms.get_chemical_symbols()
    rc = 4.6
    nij_max = 4678
    max_occurs = Counter({'Ni': 108, "Mo": 54})
    angular = False
    eta = np.array([0.05, 4.0, 20.0, 80.0])
    omega = np.array([0.0])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])
    params = {'eta': eta, 'omega': omega, 'gamma': gamma, 'zeta': zeta,
              'beta': beta}

    with tf.Graph().as_default():
        vap = VirtualAtomMap(max_occurs, symbols)
        sf = BatchSymmetryFunction(rc, max_occurs, nij_max, 0, batch_size=1,
                                   angular=angular, **params)

        positions = vap.map_array_to_gsl(atoms.positions).astype(np.float32)
        cells = atoms.cell.array.astype(np.float32)
        atom_masks = vap.atom_masks.astype(np.float32)
        g2_map = get_g2_map(atoms, rc, nij_max, sf.all_kbody_terms,
                            vap, sf.offsets, for_prediction=False)
        composition = np.zeros(len(sf.elements), dtype=np.float32)
        for element, count in Counter(symbols).items():
            composition[sf.elements.index(element)] = np.float32(count)

        cache_dict = {
            'positions': bytes_feature(positions.tostring()),
            'cells': bytes_feature(cells.tostring()),
            'n_atoms': int32_feature(len(atoms)),
            'volume': float_feature(np.float32(atoms.get_volume())),
            'y_true': float_feature(np.float32(0.0)),
            'mask': bytes_feature(atom_masks.tostring()),
            'composition': bytes_feature(composition.tostring()),
            'pulay': float_feature(np.float32(0.0)),
        }
        for key, value in g2_map.items():
            cache_dict[key] = bytes_feature(value.tostring())

        forces = vap.map_array_to_gsl(np.zeros((4, 3))).astype(np.float32)
        cache_dict['f_true'] = bytes_feature(forces.tostring())
        virial = np.zeros(6, np.float32)
        cache_dict['stress'] = bytes_feature(virial.tostring())

        with tf.python_io.TFRecordWriter("test.tfrecords") as writer:
            example = tf.train.Example(
                features=tf.train.Features(feature=cache_dict))
            writer.write(example.SerializeToString())


def test_single():
    atoms = bulk('Ni', crystalstructure='fcc', cubic=True)
    atoms.set_chemical_symbols(['Ni', 'Mo', 'Ni', 'Ni'])

    symbols = atoms.get_chemical_symbols()
    rc = 6.5
    elements = sorted(list(set(symbols)))
    angular = False
    dtype = np.float32

    with tf.Graph().as_default():
        vap = VirtualAtomMap(Counter(symbols), symbols)
        nij_max, nijk_max = get_nij_and_nijk(atoms, rc, angular=angular)
        sf = SymmetryFunction(rc, elements, angular=angular, periodic=True)

        inputs = {
            "positions": vap.map_array_to_gsl(atoms.positions).astype(dtype),
            "cells": atoms.cell.array.astype(dtype),
            "atom_masks": vap.atom_masks.astype(dtype),
            "n_atoms_vap": np.int32(len(atoms) + 1),
            "row_splits": vap.splits.astype(np.int32)
        }

        g2_map = get_g2_map(atoms, rc, nij_max, sf.all_kbody_terms,
                            vap, sf.offsets, for_prediction=True)
        inputs.update(g2_map)

        if angular:
            g4_map = get_g4_map(atoms, g2_map, sf.all_kbody_terms, sf.offsets,
                                vap, nijk_max, for_prediction=True)
            inputs.update(g4_map)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print(sess.run(sf.build_graph(inputs)))


def test_batch():
    atoms = bulk('Ni', crystalstructure='fcc', cubic=True)
    symbols = atoms.get_chemical_symbols()
    rc = 4.6
    nij_max = 4678
    max_occurs = Counter({'Ni': 108, "Mo": 54})
    angular = False
    eta = np.array([0.05, 4.0, 20.0, 80.0])
    omega = np.array([0.0])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])
    params = {'eta': eta, 'omega': omega, 'gamma': gamma, 'zeta': zeta,
              'beta': beta}
    dtype = np.float32

    with tf.Graph().as_default():
        vap = VirtualAtomMap(max_occurs, symbols)
        sf = BatchSymmetryFunction(rc, max_occurs, nij_max, 0, batch_size=1,
                                   angular=angular, **params)
        inputs = {
            "positions": vap.map_array_to_gsl(atoms.positions).astype(dtype),
            "cells": atoms.cell.array.astype(dtype),
            "atom_masks": vap.atom_masks.astype(dtype),
        }
        g2_map = get_g2_map(atoms, rc, nij_max, sf.all_kbody_terms,
                            vap, sf.offsets, for_prediction=False)
        inputs.update(g2_map)

        for key, val in inputs.items():
            inputs[key] = np.expand_dims(val, 0)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print(sess.run(sf.build_graph(inputs)))


class TensorAlloyCalculator(Calculator):
    """
    ASE-Calculator for TensorAlloy graph models.
    """

    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {}
    nolabel = True

    def __init__(self, graph_model_path: str, atoms=None):
        """
        Initialization method.

        Parameters
        ----------
        graph_model_path : str
            The exported model to load.
        atoms : Atoms
            The target `Atoms` object.
        """
        super(TensorAlloyCalculator, self).__init__(
            restart=None, ignore_bad_restart_file=False, label=None,
            atoms=atoms)

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = graph_pb2.GraphDef()
            with open(graph_model_path, "rb") as fp:
                output_graph_def.ParseFromString(fp.read())
                importer.import_graph_def(output_graph_def, name="")
            self._graph_model_path = graph_model_path
            self._model_dir = dirname(graph_model_path)
            self._sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True), graph=graph)
            self._graph = graph
            self._ops = self._get_ops()
            self._params = self._get_params()
            self._placeholders = self._get_placeholder_ops()
            self._feed_dict = None
            self.implemented_properties = self._predict_properties
            self.vap_cache = {}

    def _get_ops(self):
        """
        Recover prediction Ops from the graph model.
        """
        graph = self._graph
        props_and_names = {
            'energy': 'Output/Energy/energy:0',
            'forces': 'Output/Forces/forces:0',
            'stress': 'Output/Stress/Voigt/stress:0',
        }
        ops = {}
        for prop, name in props_and_names.items():
            try:
                ops[prop] = graph.get_tensor_by_name(name)
            except KeyError:
                continue
        self._predict_properties = list(ops.keys())
        return ops

    def _get_params(self):
        """
        Recover symmetry function parameters from the graph model.
        """
        return json.loads(self._sess.run(
            self._graph.get_tensor_by_name('Transformer/params:0')))

    def _get_placeholder_ops(self):
        """
        Recover placeholders from the graph model.
        """
        keys = ['positions', 'cells', 'n_atoms_plus_virt', 'volume',
                'pulay_stress', 'mask', 'composition', 'row_splits',
                'g2.ilist', 'g2.jlist', 'g2.shift', 'g2.v2g_map']
        if self._params['angular']:
            keys += ['g4.v2g_map', 'g4.ij.ilist', 'g4.ij.jlist',
                     'g4.ik.ilist', 'g4.ik.klist', 'g4.jk.jlist', 'g4.jk.klist',
                     'g4.shift.ij', 'g4.shift.ik', 'g4.shift.jk']
        placeholders = {}
        for key in keys:
            placeholders[key] = self._graph.get_tensor_by_name(
                f"Placeholders/{key}:0")
        return placeholders

    def get_feed_dict(self, atoms: Atoms, print_time=False):
        """
        Return the feed dict.
        """
        rc = self._params['rc']
        angular = self._params['angular']
        elements = self._params['elements']
        all_kbody_terms = get_kbody_terms(elements, angular)[0]
        kbody_sizes = compute_dimension(
            all_kbody_terms, len(self._params['eta']),
            len(self._params['omega']), len(self._params['beta']),
            len(self._params['gamma']), len(self._params['zeta']))[1]
        offsets = np.insert(np.cumsum(kbody_sizes), 0, 0)
        symbols = atoms.get_chemical_symbols()
        c = Counter(symbols)
        max_occurs = Counter({el: max(1, c[el]) for el in elements})
        vap = VirtualAtomMap(max_occurs, symbols)
        formula = atoms.get_chemical_formula('reduce')
        if formula not in self.vap_cache:
            self.vap_cache[formula] = vap
        nij_max, nijk_max = get_nij_and_nijk(atoms, self._params['rc'],
                                             angular=self._params['angular'])
        g2_map = get_g2_map(atoms, rc, nij_max, all_kbody_terms, vap, offsets,
                            for_prediction=True, print_time=print_time)
        positions = vap.map_array_to_gsl(atoms.positions).astype(np.float32)
        cell = np.asarray(atoms.cell).astype(np.float32)
        mask = vap.atom_masks.astype(np.float32)
        n_atoms_vap = np.int32(vap.max_vap_n_atoms)
        volume = np.float32(atoms.get_volume())
        pulay_stress = np.float32(0.0)
        row_splits = vap.splits.astype(np.int32)
        n_elements = len(elements)
        composition = np.zeros(n_elements, dtype=np.float32)
        for element, count in Counter(symbols).items():
            composition[elements.index(element)] = np.float32(count)
        feed_dict = {
            self._placeholders['positions']: positions,
            self._placeholders['cells']: cell,
            self._placeholders['n_atoms_plus_virt']: n_atoms_vap,
            self._placeholders['mask']: mask,
            self._placeholders['composition']: composition,
            self._placeholders['row_splits']: row_splits,
            self._placeholders['volume']: volume,
            self._placeholders['pulay_stress']: pulay_stress,
        }
        for key, value in g2_map.items():
            feed_dict[self._placeholders[key]] = value
        if angular:
            g4_map = get_g4_map(atoms, g2_map, all_kbody_terms, offsets, vap,
                                nijk_max, for_prediction=True)
            for key, value in g4_map.items():
                feed_dict[self._placeholders[key]] = value
        return feed_dict

    def get_forces(self, atoms=None):
        """
        Return the calculated atomic forces.
        """
        atoms = atoms or self.atoms
        forces = np.insert(self.get_property('forces', atoms), 0, 0, 0)
        formula = atoms.get_chemical_formula('reduce')
        vap = self.vap_cache[formula]
        return vap.reverse_map_gsl_to_local(forces)

    def calculate(self, atoms=None, properties=('energy', 'forces', 'stress'),
                  system_changes=all_changes, print_time=False):
        """
        Calculate the total energy and other properties (1body, kbody, atomic).
        """
        Calculator.calculate(self, atoms, properties)
        with self._graph.as_default():
            ops = {target: self._ops[target] for target in properties}
            tic = time.time()
            feed_dict = self.get_feed_dict(atoms, print_time=print_time)
            if print_time:
                print(f'* Python feed dict time: {time.time() - tic}')
            tic = time.time()
            self.results = self._sess.run(ops, feed_dict=feed_dict)
            if print_time:
                print(f'* TensorFlow execution time: {time.time() - tic}')


def prediction_phase_benchmark():
    """
    Benchmark tests of the prediction phase.
    """
    calc = TensorAlloyCalculator('NiMo.pb')
    ref = bulk('Mo', crystalstructure='bcc', cubic=True)
    ref.set_chemical_symbols(['Ni', 'Mo'])

    # Warm up
    calc.calculate(ref)

    for n in [10, 15, 20, 25, 30, 35, 40]:
        atoms = ref * [n, n, n]
        print(f"Number of atoms: {len(atoms)}")
        calc.calculate(atoms, print_time=True)


if __name__ == "__main__":
    test_single()
    test_batch()
    test_cache()
    prediction_phase_benchmark()
