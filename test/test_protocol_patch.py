"""Patch-restricted and self-adjoint HB+OR updates along the O-REX/NEO-REX protocols."""

import unittest
from types import SimpleNamespace

import torch

import neoqcd.sun_utils as sun
from neoqcd.mcmc import HBOR, DefectMCMCAdapter
from neoqcd.utils import Defect, create_mask


def _defect_plaquette_links(defect):
    """Links of every plaquette whose weight depends on bc, as a (D, T, L, L, L) bool tensor."""
    D, T, L = defect.D, defect.T, defect.L
    kappa = defect.create_defect_mask()[1].to(torch.bool)  # (D, T, L, L, L): links with kappa < 1
    links = torch.zeros_like(kappa)
    for mu in range(D):
        for nu in range(D):
            if nu == mu:
                continue
            # Plaquette (mu, nu) at x: U_mu(x), U_nu(x + mu), U_mu(x + nu), U_nu(x); roll by -1 reads x + e.
            members = [
                (mu, (0, 0)), (nu, (-1, mu)), (mu, (-1, nu)), (nu, (0, 0)),
            ]
            depends = torch.zeros((T,) + (L,) * (D - 1), dtype=torch.bool)
            for direction, (shift, axis) in members:
                field = kappa[direction]
                depends |= torch.roll(field, shift, dims=axis) if shift else field
            for direction, (shift, axis) in members:
                # The member link at x + e is marked by rolling the plaquette flag back by +1.
                flag = torch.roll(depends, -shift, dims=axis) if shift else depends
                links[direction] |= flag
    return links


class ProtocolPatchMaskTest(unittest.TestCase):
    def test_half_size_zero_is_the_bounding_box_of_the_bc_dependent_plaquette_links(self):
        for dsize, time_slice, space_slice in ((2, 4, 3), (1, 2, 1), (3, 3, 2)):
            defect = Defect(D=4, T=8, L=8, dsize=dsize, time_slice=time_slice, space_slice=space_slice)
            patch = defect.protocol_patch_mask(0)[0, ..., 0, 0].to(torch.bool)
            reference = _defect_plaquette_links(defect)
            # Every link of a bc-dependent plaquette is in the box ...
            self.assertFalse(bool((reference & ~patch).any()), (dsize, time_slice, space_slice))
            # ... and the box is their bounding box: same occupied sites along every axis.
            patch_sites, reference_sites = patch.any(dim=0), reference.any(dim=0)
            for axis in range(4):
                others = tuple(a for a in range(4) if a != axis)
                self.assertTrue(torch.equal(patch_sites.any(dim=others), reference_sites.any(dim=others)), axis)
            # The box holds every link at its sites, the strict set fewer: 512 against 104 for dsize 2.
            self.assertEqual(int(patch.sum()), 4 * 2 * (dsize + 2) ** 3)
            if dsize == 2:
                self.assertEqual(int(reference.sum()), 104)

    def test_box_grows_by_one_site_per_unit_and_wraps(self):
        defect = Defect(D=4, T=8, L=8, dsize=2, time_slice=0, space_slice=7)
        for half in range(3):
            patch = defect.protocol_patch_mask(half)
            self.assertEqual(tuple(patch.shape), (1, 4, 8, 8, 8, 8, 1, 1))
            sites = patch[0, 0, ..., 0, 0]
            self.assertEqual(int(sites.sum()), (2 * half + 2) * (2 + 2 * half + 2) ** 3)
            times = sites.flatten(1).any(dim=1).nonzero().squeeze(-1).tolist()
            self.assertEqual(sorted(times), sorted({(t) % 8 for t in range(-half, 2 + half)}))
            xs = sites.movedim(1, 0).flatten(1).any(dim=1).nonzero().squeeze(-1).tolist()
            self.assertEqual(sorted(xs), sorted({(7 - 1 - half + k) % 8 for k in range(2 + 2 * half + 2)}))
        self.assertTrue(bool(defect.protocol_patch_mask(10).all()))
        with self.assertRaises(ValueError):
            defect.protocol_patch_mask(-1)


class _RecordingHBOR(HBOR):
    """Records the order of the elementary updates and leaves the links unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.hb_update = self._record_hb
        self.over_update = self._record_over

    def _record_hb(self, flat, links, m, rand, defect, cfgs):
        self.calls.append(("hb", m))
        return flat.index_select(1, links.links[m])

    def _record_over(self, flat, links, m, defect, cfgs):
        self.calls.append(("or", m))
        return flat.index_select(1, links.links[m])


class _OverrelaxationOnlyHBOR(HBOR):
    """The heat bath leaves the links unchanged, so the sweep is the deterministic OR sequence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hb_update = lambda flat, links, m, rand, defect, cfgs: flat.index_select(1, links.links[m])


def _flow_pars(D=4, T=4, L=4, orsteps=2):
    return SimpleNamespace(N=3, D=D, orsteps=orsteps, device=torch.device("cpu"), mask=create_mask(D, T, L), defect=None)


class ReversedSweepTest(unittest.TestCase):
    def test_reverse_runs_the_same_elementary_updates_backwards(self):
        pars = _flow_pars(orsteps=2)
        hbor = _RecordingHBOR(pars, beta=6.0)
        cfgs = sun.SUN_haar((1, 4, 4, 4, 4, 4, 3), dtype=torch.cdouble)
        hbor(cfgs, pars.mask, defect=None)
        forward = list(hbor.calls)
        hbor.calls.clear()
        hbor(cfgs, pars.mask, defect=None, reverse=True)
        self.assertEqual(forward, [("hb", m) for m in range(8)] + [("or", m) for m in range(8)] * 2)
        self.assertEqual(hbor.calls, list(reversed(forward)))

    def test_reversed_overrelaxation_sweep_inverts_the_forward_one(self):
        # The OR sweep is a deterministic map that amplifies rounding by ~1e3 per sweep on a hot
        # configuration, so the round trip is checked over one sweep (4e-12 measured) and per block.
        torch.manual_seed(0)
        pars = _flow_pars(orsteps=1)
        hbor = _OverrelaxationOnlyHBOR(pars, beta=6.0)
        cfgs = sun.SUN_haar((2, 4, 4, 4, 4, 4, 3), dtype=torch.cdouble)
        forward = hbor(cfgs, pars.mask, defect=None)
        self.assertGreater((forward - cfgs).abs().max().item(), 1e-3)
        back = hbor(forward, pars.mask, defect=None, reverse=True)
        self.assertLess((back - cfgs).abs().max().item(), 1e-10)
        # The plain sweep applied twice is not the identity: the order matters.
        twice = hbor(forward, pars.mask, defect=None)
        self.assertGreater((twice - cfgs).abs().max().item(), 1e-3)

    def test_one_block_update_is_undone_only_with_the_subgroups_reversed(self):
        torch.manual_seed(0)
        pars = _flow_pars(orsteps=1)
        hbor = _OverrelaxationOnlyHBOR(pars, beta=6.0)
        cfgs = sun.SUN_haar((2, 4, 4, 4, 4, 4, 3), dtype=torch.cdouble)
        links = hbor._links_for_mask(pars.mask)
        flat = sun.flat_links(cfgs, 4)
        once = flat.index_copy(1, links.links[0], hbor.over_update(flat, links, 0, None, cfgs))
        hbor._subgroup_order = (2, 1, 0)
        undone = once.index_copy(1, links.links[0], hbor.over_update(once, links, 0, None, cfgs))
        self.assertLess((undone - flat).abs().max().item(), 1e-13)
        hbor._subgroup_order = (0, 1, 2)
        same_order = once.index_copy(1, links.links[0], hbor.over_update(once, links, 0, None, cfgs))
        self.assertGreater((same_order - flat).abs().max().item(), 1e-2)


class _ReverseRecordingHBOR(HBOR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverses = []

    def __call__(self, cfgs, mask, defect, rand=None, reverse=False):
        self.reverses.append(bool(reverse))
        return super().__call__(cfgs, mask, defect, rand=rand, reverse=reverse)


class AdapterPatchTest(unittest.TestCase):
    def test_patch_restricts_the_updates_and_the_coin_picks_both_orders(self):
        torch.manual_seed(1)
        D, T, L = 4, 6, 6
        pars = SimpleNamespace(N=3, D=D, orsteps=1, device=torch.device("cpu"), mask=create_mask(D, T, L), defect=None)
        defect = Defect(D, T, L, dsize=1, time_slice=2, space_slice=2)
        patch = defect.protocol_patch_mask(0)
        adapter = DefectMCMCAdapter(pars, beta=6.0, defect_mask=defect.create_defect_mask(), batch_size=2,
                                    patch_mask=patch, symmetric_updates=True)
        adapter._batch_hbor = _ReverseRecordingHBOR(pars, beta=6.0)
        adapter.set_parameter(torch.tensor([0.3, 0.7]))
        cfgs = sun.SUN_haar((2, D, T, L, L, L, 3), dtype=torch.cdouble)
        out = cfgs
        for _ in range(30):
            out = adapter(out, nsteps=1)
        inside = patch[0, ..., 0, 0].to(torch.bool).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(out)
        self.assertLess((out - cfgs)[~inside].abs().max().item(), 1e-12)
        self.assertGreater((out - cfgs)[inside].abs().max().item(), 1e-3)
        reverses = adapter._batch_hbor.reverses
        self.assertEqual(len(reverses), 30)
        self.assertIn(True, reverses)
        self.assertIn(False, reverses)

    def test_without_patch_the_whole_lattice_moves(self):
        torch.manual_seed(2)
        D, T, L = 4, 4, 4
        pars = SimpleNamespace(N=3, D=D, orsteps=1, device=torch.device("cpu"), mask=create_mask(D, T, L), defect=None)
        defect = Defect(D, T, L, dsize=1, time_slice=1, space_slice=1)
        adapter = DefectMCMCAdapter(pars, beta=6.0, defect_mask=defect.create_defect_mask(), batch_size=1)
        adapter.set_parameter(torch.tensor([0.5]))
        cfgs = sun.SUN_haar((1, D, T, L, L, L, 3), dtype=torch.cdouble)
        out = adapter(cfgs, nsteps=1)
        moved = (out - cfgs).abs().amax(dim=(-1, -2)) > 1e-6
        self.assertGreater(moved.float().mean().item(), 0.9)


if __name__ == "__main__":
    unittest.main()
