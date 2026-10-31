import torch
import torch.utils.checkpoint as checkpoint
import neoqcd.sun_utils as sun
from neoqcd.theory import (
    Wilson_action,
    build_nf_halo_plaquette_masks,
    local_nf_action_change as theory_local_nf_action_change,
)
from typing import Optional
from neoqcd.utils import Defect, DefectBox


_CONSTANTS = {}


def quiesce_for_capture(device):
    """Make a CUDA graph capture safe next to NCCL.

    torch.cuda.graph captures in global mode, where any CUDA call from another thread aborts the
    process. ProcessGroupNCCL's cleanup thread polls the events of pending collectives about once
    a second (torch 2.0 has no thread-local capture mode), so the device is synchronized and the
    thread given time to find its queue empty. Jobs should also export
    NCCL_ASYNC_ERROR_HANDLING=0, which keeps that thread from being started.
    """
    import time
    torch.cuda.synchronize(device)
    try:
        import torch.distributed as dist
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    except Exception:  # noqa: BLE001
        distributed = False
    if distributed:
        time.sleep(1.5)


def _constant(key, build):
    """A small device tensor built once per (key, device, dtype): a tensor made from Python data
    inside a CUDA graph capture is a synchronous host-to-device copy, which the capture forbids,
    so the constants of the sweep are created before it (the warm-up runs fill this cache)."""
    value = _CONSTANTS.get(key)
    if value is None:
        value = build()
        _CONSTANTS[key] = value
    return value


def _su2_sign(view_ndim, device, dtype):
    sign = _constant(("su2_sign", str(device), dtype), lambda: torch.tensor([1, -1], device=device, dtype=dtype))
    return sign.view([1] * view_ndim + [2])


def _su2_haar_from_uniform(rand):
    eps = torch.finfo(rand.dtype).tiny
    u0, u1 = rand[:, 0].clamp_min(eps), rand[:, 1]
    u2, u3 = rand[:, 2].clamp_min(eps), rand[:, 3]
    r0, r1 = torch.sqrt(-2.0 * torch.log(u0)), torch.sqrt(-2.0 * torch.log(u2))
    g = torch.stack((r0 * torch.cos(2.0 * torch.pi * u1), r0 * torch.sin(2.0 * torch.pi * u1),
                     r1 * torch.cos(2.0 * torch.pi * u3), r1 * torch.sin(2.0 * torch.pi * u3)), dim=-1)
    g = g / torch.linalg.vector_norm(g, dim=-1, keepdim=True)
    return torch.complex(g[..., 0], g[..., 1]), torch.complex(g[..., 2], g[..., 3])


class NEMCMC_update(torch.nn.Module):
    def __init__(self, flow_pars, beta, defect_mask=None, defect_par=1.0):
        super().__init__()
        self.action = Wilson_action(flow_pars, beta, defect_par)
        self.update = HBOR(flow_pars, beta, defect_par)
        self.beta = beta
        self.D = flow_pars.D
        self.device = flow_pars.device
        self.nupdates = flow_pars.updates_per_layer
        self.hierarchical_levels = flow_pars.hierarchical_levels

        self.defect_mask = defect_mask
        if self.hierarchical_levels > 0:
            self.hierarchical_rectangles = flow_pars.hierarchical_rectangles
            self.hierarchical_nupdates = flow_pars.hierarchical_updates

    def set_parameter(self, parameter: torch.Tensor, parameter_name: str = "bc"):
        if str(parameter_name) != "bc":
            raise ValueError(
                f"NEMCMC_update supports only parameter_name='bc', got {parameter_name!r}"
            )
        p = torch.as_tensor(parameter, dtype=torch.float64, device=self.device).reshape(-1)
        if p.numel() == 0:
            raise ValueError("parameter cannot be empty")
        p0 = float(p[0].item())
        if not bool(torch.allclose(p, torch.full_like(p, p0))):
            raise ValueError(
                "NEMCMC_update expects a scalar/shared parameter across the local batch. "
                "Use DefectMCMCAdapter for per-sample parameters."
            )
        self.action.defect_par = p0
        self.update.set_defect_parameter(p0)

    def forward(self, cfgs, flow_pars, rho_root=None, plaquettes=None, return_plaquettes=False):
        if plaquettes is None:
            plaquettes = sun.plaquette_fields(cfgs, self.D)
        s_old = self.action.from_plaquettes(plaquettes, self.defect_mask)

        for u in range(self.nupdates):
            cfgs = self.update(cfgs, flow_pars.mask, self.defect_mask)
            if self.hierarchical_levels > 0:
                cfgs = self.hierarchical(cfgs, 0, flow_pars)
        
        new_plaquettes = sun.plaquette_fields(cfgs, self.D)
        s_new = self.action.from_plaquettes(new_plaquettes, self.defect_mask)
        if return_plaquettes:
            return cfgs, s_new - s_old, 0, new_plaquettes, sun.mean_plaquette(new_plaquettes, self.action.N)
        return cfgs, s_new - s_old, 0

    def hierarchical(self, cfgs, level, flow_pars):
        buffers = [int(v) for v in self.hierarchical_rectangles]
        if level != 0:
            raise ValueError("hierarchical updates must start at level 0")
        if len(buffers) < self.hierarchical_levels:
            raise ValueError("one hierarchical rectangle is required per level")
        if len(self.hierarchical_nupdates) < self.hierarchical_levels:
            raise ValueError("one hierarchical update count is required per level")
        buffers = buffers[:self.hierarchical_levels]
        if any(buffer <= 0 for buffer in buffers):
            raise ValueError("hierarchical rectangle buffers must be positive")
        if flow_pars.defect is None:
            raise ValueError("hierarchical updates require flow_pars.defect geometry")

        levels = sorted(
            zip(buffers, self.hierarchical_nupdates[:self.hierarchical_levels]),
            key=lambda item: item[0],
            reverse=True,
        )
        boxes = self._hierarchical_boxes_for(flow_pars, tuple(buffer for buffer, _ in levels))
        largest = boxes[0]

        # Work on one cut of the largest box. Each level updates every link whose
        # staples stay inside its box, then copies only those links into its parent.
        patch = largest.cut(cfgs)
        for (_, nupdates), box in zip(levels, boxes):
            for _ in range(nupdates):
                if box is largest:
                    patch = self.update(patch, box.links, box.defect_mask)
                else:
                    local = box.cut(patch)
                    local = self.update(local, box.links, box.defect_mask)
                    patch = box.embed(local, patch)
        return largest.embed(patch, cfgs)

    def _hierarchical_boxes_for(self, flow_pars, buffers):
        defect = flow_pars.defect
        mask = flow_pars.mask
        defect_mask = self.defect_mask
        geometry = (
            int(defect.D), int(defect.T), int(defect.L), int(defect.dsize),
            int(defect.time_slice), int(defect.space_slice),
        )
        signature = (
            geometry,
            tuple(buffers),
            mask._version,
            None if defect_mask is None else defect_mask._version,
        )
        entry = getattr(flow_pars, "_hierarchical_boxes_cache", None)
        if (entry is not None and entry[0] == signature and entry[1] is defect
                and entry[2] is mask and entry[3] is defect_mask):
            return entry[4]

        largest = DefectBox(defect, mask, defect_mask, buffers[0])
        boxes = [largest]
        boxes.extend(
            DefectBox(defect, mask, defect_mask, buffer, outer=largest)
            for buffer in buffers[1:]
        )
        try:
            flow_pars._hierarchical_boxes_cache = (
                signature, defect, mask, defect_mask, boxes
            )
        except (AttributeError, TypeError):
            pass
        return boxes


class HBOR():
    def __init__(self, flow_pars, beta, defect_par=1.0, hierarchical=False):
        self.N = flow_pars.N
        self.D = flow_pars.D
        self.orsteps = flow_pars.orsteps
        self.device = flow_pars.device
        self._masked_links_cache = sun.masked_links_cache_for(flow_pars)

        self.beta = beta
        self._subgroup_order = (0, 1, 2)  # SU(2) subgroups of each SU(3) link update
        # CUDA graphs of whole sweeps, keyed by shapes, mask, branch and sweep direction: see
        # _graphed_sweep. Off by default; main_pt_obc.py turns it on with --cuda-graphs.
        self.use_cuda_graphs = False
        self._graphs = {}

        if self.N==2:
            self.hb_update=self._su2_selected_hb
            self.haar_hb_update=self.SU2_link_haar_hb
            self.over_update=self._su2_selected_over
        elif self.N==3:
            self.hb_update=self._su3_selected_hb
            self.haar_hb_update=self.SU3_link_haar_hb
            self.over_update=self._su3_selected_over

        self.set_defect_parameter(defect_par)

    def _defect_parameter_uses_defect(self) -> bool:
        # Decided on the host in set_defect_parameter and cached: inside a CUDA graph capture a
        # device read is not allowed, and the branch is part of the graph key anyway.
        return self._uses_defect

    def _compute_uses_defect(self) -> bool:
        if torch.is_tensor(self.defect_par):
            return bool((self.defect_par < 1.0).any().item())
        return bool(self.defect_par < 1.0)

    def set_defect_parameter(self, defect_par):
        if torch.is_tensor(defect_par):
            self.defect_par = defect_par.detach().to(device=self.device, dtype=torch.float64).reshape(-1)
        else:
            self.defect_par = float(defect_par)
        self._uses_defect = self._compute_uses_defect()
        if self._uses_defect:
            self.sum_of_staples = self.sum_of_staples_defect
            self.heatbath = self.heatbath_defect
        else:
            self.sum_of_staples = self.sum_of_staples_standard
            self.heatbath = self.heatbath_standard


    def __call__(self, cfgs, mask, defect, rand=None, reverse=False):
        """One HB sweep followed by `orsteps` OR sweeps over the links selected by `mask`.

        With reverse=True the same elementary updates run in the opposite order (OR sweeps, then
        HB, each over the checkerboard blocks backwards). Every elementary update is self-adjoint,
        so this is the adjoint kernel of the plain sweep: choosing either with probability 1/2
        gives a self-adjoint update, as the detailed balance of the O-REX swap requires.
        """

        if (self.use_cuda_graphs and cfgs.is_cuda and not torch.is_grad_enabled()
                and not isinstance(mask, sun.MaskedLinks)):
            return self._graphed_sweep(cfgs, mask, defect, rand, reverse)
        if rand is None:
            rand = torch.rand(self._rand_shape(cfgs), device=cfgs.device, dtype=cfgs.real.dtype)
        return self._sweep(cfgs, mask, defect, rand, reverse)

    def _rand_shape(self, cfgs):
        rn_shape = list(cfgs.shape[:-2])
        rn_shape.insert(1, 18 if self.N == 3 else 6)
        return tuple(rn_shape)

    def _graph_key(self, cfgs, mask, defect, reverse):
        parameter = (
            ("batched",) if torch.is_tensor(self.defect_par) and self.defect_par.numel() > 1
            else ("scalar", float(self.defect_par.reshape(-1)[0]) if torch.is_tensor(self.defect_par) else float(self.defect_par))
        )
        return (tuple(cfgs.shape), str(cfgs.dtype), id(mask), int(mask._version), id(defect),
                self._defect_parameter_uses_defect(), bool(reverse), parameter)

    def _graphed_sweep(self, cfgs, mask, defect, rand, reverse):
        """Replay the CUDA graph of this sweep, capturing it on first use.

        Static buffers hold the configurations, the heat-bath random numbers and the per-chain
        defect coupling; the graph is keyed by the shapes, the mask, the "uses defect" branch,
        the sweep direction and (for a scalar coupling) its value, all of which are decided on the
        host before the capture. If the capture fails the sweep runs eagerly from then on.
        """
        key = self._graph_key(cfgs, mask, defect, reverse)
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture_sweep(cfgs, mask, defect, reverse)
            except Exception as err:  # noqa: BLE001 - any capture failure means eager from now on
                import warnings
                warnings.warn(f"CUDA graph capture of the HBOR sweep failed, running eagerly: {type(err).__name__}: {err}")
                self.use_cuda_graphs = False
                if rand is None:
                    rand = torch.rand(self._rand_shape(cfgs), device=cfgs.device, dtype=cfgs.real.dtype)
                return self._sweep(cfgs, mask, defect, rand, reverse)
            self._graphs[key] = entry
        graph, static_cfgs, static_rand, static_par, static_out, _links = entry
        static_cfgs.copy_(cfgs)
        if rand is None:
            static_rand.uniform_()
        else:
            static_rand.copy_(rand)
        if static_par is not None:
            static_par.copy_(self.defect_par)
        graph.replay()
        return static_out.clone()

    def _capture_sweep(self, cfgs, mask, defect, reverse):
        static_cfgs = cfgs.clone()
        static_rand = torch.rand(self._rand_shape(cfgs), device=cfgs.device, dtype=cfgs.real.dtype)
        # The index tables are read by the captured kernels: keep them alive with the graph.
        links = self._links_for_mask(mask)
        static_par = None
        saved_par = self.defect_par
        if torch.is_tensor(self.defect_par) and self.defect_par.numel() > 1:
            static_par = self.defect_par.clone()
            self.defect_par = static_par  # the captured kernels read this buffer
        try:
            # Warm-up on a side stream, as the CUDA graph documentation prescribes, then capture.
            quiesce_for_capture(cfgs.device)
            stream = torch.cuda.Stream(device=cfgs.device)
            stream.wait_stream(torch.cuda.current_stream(device=cfgs.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._sweep(static_cfgs, links, defect, static_rand, reverse)
            torch.cuda.current_stream(device=cfgs.device).wait_stream(stream)
            # Each graph keeps its own memory pool: they are replayed in any order.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._sweep(static_cfgs, links, defect, static_rand, reverse)
        finally:
            self.defect_par = saved_par
        return graph, static_cfgs, static_rand, static_par, static_out, links

    def _sweep(self, cfgs, mask, defect, rand, reverse):
        links = mask if isinstance(mask, sun.MaskedLinks) else self._links_for_mask(mask)
        flat = sun.flat_links(cfgs, self.D)
        blocks = list(range(2 * self.D))
        if reverse:
            # Blocks and, inside each SU(3) link update, the SU(2) subgroups run backwards.
            blocks.reverse()
            self._subgroup_order = (2, 1, 0)
            try:
                flat = self._overrelaxation_sweeps(flat, links, blocks, defect, cfgs)
                flat = self._heatbath_sweep(flat, links, blocks, rand, defect, cfgs)
            finally:
                self._subgroup_order = (0, 1, 2)
        else:
            flat = self._heatbath_sweep(flat, links, blocks, rand, defect, cfgs)
            flat = self._overrelaxation_sweeps(flat, links, blocks, defect, cfgs)
        cfgs = flat.reshape(cfgs.shape)
        return sun.SU3_unitarize(cfgs) if self.N == 3 else cfgs

    def _heatbath_sweep(self, flat, links, blocks, rand, defect, cfgs):
        # Gather only the selected links and the links needed by their staples.
        for m in blocks:
            mu = m // 2
            rand_m = rand[:, :, mu].flatten(2).index_select(2, links.sites[m])
            flat = flat.index_copy(1, links.links[m], self.hb_update(flat, links, m, rand_m, defect, cfgs))
        return flat

    def _overrelaxation_sweeps(self, flat, links, blocks, defect, cfgs):
        for _ in range(self.orsteps):
            for m in blocks:
                flat = flat.index_copy(1, links.links[m], self.over_update(flat, links, m, defect, cfgs))
        return flat

    def _links_for_mask(self, mask):
        return self._masked_links_cache.get(mask, self.D)

    def haar(self, cfgs):
        return sun.SUN_haar(cfgs.shape[:-1], device=cfgs.device, dtype=cfgs.dtype)
    
    def _batch_kappa(self, cfgs, defect):
        defect = defect.to(device=cfgs.device, dtype=cfgs.real.dtype)
        if torch.is_tensor(self.defect_par):
            p = self.defect_par.to(device=cfgs.device, dtype=cfgs.real.dtype)
            if p.numel() == 1:
                return defect[0, :] + defect[1, :] * p.reshape(())
            if p.numel() != int(cfgs.shape[0]):
                raise ValueError(
                    f"defect_par has shape ({p.numel()},), expected batch size {int(cfgs.shape[0])}"
                )
            view_shape = (int(cfgs.shape[0]),) + (1,) * (defect.dim() - 1)
            return defect[0, :].unsqueeze(0) + defect[1, :].unsqueeze(0) * p.view(view_shape)
        return defect[0, :] + defect[1, :] * float(self.defect_par)

    def _kappa_link(self, kappa, mu):
        if kappa.dim() == self.D + 2:
            return kappa[:, mu]
        return kappa[mu]

    def _beta_for_staples(self, k):
        if torch.is_tensor(self.beta):
            beta = self.beta.to(device=k.device, dtype=k.dtype)
        else:
            beta = _constant(("beta", float(self.beta), str(k.device), k.dtype),
                             lambda: torch.tensor(float(self.beta), device=k.device, dtype=k.dtype))
        if beta.dim() == 0:
            return beta
        beta = beta.reshape(-1)
        if beta.numel() == 1:
            return beta.reshape(())
        if beta.numel() != int(k.shape[0]):
            raise ValueError(f"beta has shape ({beta.numel()},), expected batch size {int(k.shape[0])}")
        return beta.view((int(k.shape[0]),) + (1,) * (k.dim() - 1))

    def sum_of_staples_defect(self, cfgs, mu, defect):
        staple = torch.zeros(cfgs[:,0].shape, dtype=cfgs.dtype, device=cfgs.device)
        kappa = self._batch_kappa(cfgs, defect)

        for nu in range(self.D):
            if nu != mu:
                kappamunu = (
                    self._kappa_link(kappa, mu)
                    * self._kappa_link(torch.roll(kappa, 1, dims=(-self.D + mu)), nu)
                    * self._kappa_link(torch.roll(kappa, 1, dims=(-self.D + nu)), mu)
                    * self._kappa_link(kappa, nu)
                )
                staple += kappamunu.unsqueeze(-1).unsqueeze(-1) * sun.pstaple(cfgs, mu, nu, self.D)
                
                kappamunu = (
                    self._kappa_link(kappa, mu)
                    * self._kappa_link(torch.roll(kappa, (1,-1), dims=(-self.D + mu,-self.D + nu)), nu)
                    * self._kappa_link(torch.roll(kappa, -1, dims=(-self.D + nu)), mu)
                    * self._kappa_link(torch.roll(kappa, -1, dims=(-self.D + nu)), nu)
                )
                staple += kappamunu.unsqueeze(-1).unsqueeze(-1) * sun.nstaple(cfgs, mu, nu, self.D)

        return staple

    def sum_of_staples_standard(self, cfgs, mu, defect=None):
        staple = torch.zeros(cfgs[:,0].shape, dtype=cfgs.dtype, device=cfgs.device)

        for nu in range(self.D):
            if nu != mu:
                staple += sun.pstaple(cfgs, mu, nu, self.D)
                staple += sun.nstaple(cfgs, mu, nu, self.D)

        return staple

    def _gathered_staples(self, cfgs, flat, links, m, defect):
        mu = m // 2
        batch = int(cfgs.shape[0])
        n = int(links.sites[m].numel())
        staples = torch.zeros((batch, n, self.N, self.N), dtype=cfgs.dtype, device=cfgs.device)
        use_defect = defect is not None and self._defect_parameter_uses_defect()
        if use_defect:
            kappa = self._batch_kappa(cfgs, defect)
            if kappa.dim() == self.D + 1:
                kappa = kappa.reshape(1, -1)
            else:
                kappa = kappa.flatten(1)

            def kappa_at(position):
                values = kappa.index_select(-1, position)
                return values.expand(batch, -1) if values.shape[0] == 1 and batch != 1 else values

            kappa_mu = kappa_at(links.links[m])

        for j, indices in enumerate(links.staple_links[m]):
            upper, lower = sun.staples_at(flat, indices)
            if use_defect:
                upper_weight = kappa_mu * kappa_at(indices[0]) * kappa_at(indices[1]) * kappa_at(indices[2])
                lower_weight = kappa_mu * kappa_at(indices[3]) * kappa_at(indices[4]) * kappa_at(indices[5])
                upper = upper * upper_weight.unsqueeze(-1).unsqueeze(-1)
                lower = lower * lower_weight.unsqueeze(-1).unsqueeze(-1)
            staples = staples + upper + lower
        return staples

    def _su2_selected_hb(self, flat, links, m, rand, defect, cfgs):
        staple = self._gathered_staples(cfgs, flat, links, m, defect)
        old = flat.index_select(1, links.links[m])
        proposal, acc = self.heatbath(staple, rand[:, :6], self.N / 2.0)
        return torch.where(acc, proposal, old)

    def _su2_selected_over(self, flat, links, m, defect, cfgs):
        staple = self._gathered_staples(cfgs, flat, links, m, defect)
        old = flat.index_select(1, links.links[m])
        k = torch.sqrt(sun.det_small(staple).abs())
        good = k > 0
        safe = torch.where(good, k, torch.ones_like(k))
        V = sun.SUN_dagger(staple / safe.unsqueeze(-1).unsqueeze(-1))
        proposal = sun.SUN_mul(sun.SUN_mul(V, sun.SUN_dagger(old)), V)
        return torch.where(good.unsqueeze(-1).unsqueeze(-1), proposal, old)

    def _su3_selected_hb(self, flat, links, m, rand, defect, cfgs):
        staple = self._gathered_staples(cfgs, flat, links, m, defect)
        new = flat.index_select(1, links.links[m])
        colors = torch.arange(3, device=cfgs.device)
        for phb in self._subgroup_order:
            Rk = SU3_compute_RK(new, staple, colors, phb)
            subgroup, acc = self.heatbath(Rk, rand[:, 6 * phb:6 * (phb + 1)], self.N / 2.0)
            proposal = SU2_left_multiply_SU3(new, subgroup, phb)
            new = torch.where(acc, proposal, new)
        return new

    def _su3_selected_over(self, flat, links, m, defect, cfgs):
        staple = self._gathered_staples(cfgs, flat, links, m, defect)
        new = flat.index_select(1, links.links[m])
        colors = torch.arange(3, device=cfgs.device)
        for phb in self._subgroup_order:
            Rk = SU3_compute_RK(new, staple, colors, phb)
            k = torch.sqrt(sun.det_small(Rk).abs())
            good = k > 0
            safe = torch.where(good, k, torch.ones_like(k))
            V = sun.SUN_dagger(Rk / safe.unsqueeze(-1).unsqueeze(-1))
            subgroup = sun.SUN_mul(V, V)
            proposal = SU2_left_multiply_SU3(new, subgroup, phb)
            new = torch.where(good.unsqueeze(-1).unsqueeze(-1), proposal, new)
        return new

    def SU2toSU3(self, Ak, phb):
        idx0 = torch.arange(phb, device=Ak.device)
        idx1 = torch.arange(phb, 2, device=Ak.device)
        A = torch.zeros(Ak.shape[:-2] + (2, 1), dtype=Ak.dtype, device=Ak.device)
        B = torch.cat((torch.index_select(Ak, -1, idx0), A, torch.index_select(Ak, -1, idx1)), -1)
        A = torch.zeros(Ak.shape[:-2] + (1, 2), dtype=Ak.dtype, device=Ak.device)
        C = torch.ones(Ak.shape[:-2] + (1, 1), dtype=Ak.dtype, device=Ak.device)
        AA = torch.cat((torch.index_select(A, -1, idx0), C, torch.index_select(A, -1, idx1)), -1)
        return torch.cat((torch.index_select(B, -2, idx0), AA, torch.index_select(B, -2, idx1)), -2)

    def heatbath_standard(self, staples, rand, factor):
        k = torch.sqrt(sun.det_small(staples).abs())
        k0 = k > 0
        safe_k = torch.where(k0, k, torch.ones_like(k))
        xi = (self.N / 2.0) / (self._beta_for_staples(safe_k) * safe_k)

        #WARNING: if rand[:,0] and one of the two other rand are simultaneously 0 a NaN occurs in the gradient of mod. Probably very unlikely
        lam2 = (- 0.5 * xi * (torch.log(1.0 - rand[:, 0]) + torch.cos(2.0 * torch.pi * (1.0 - rand[:, 1]))**2 * torch.log(1.0 - rand[:, 2])))
        acc = (1.0 - lam2 >= rand[:, 3]**2)
        x0 = (1.0 - 2.0 * lam2) * acc

        #x0=x0.detach() # This detach prevents (unlikely) NaN in the gradient of the square root mod
        stheta = torch.sqrt(1.0 -  (1.0 - 2.0 * rand[:, 4])**2)
        mod = torch.sqrt(1.0 - x0**2)

        A = torch.complex(x0, mod * (1.0 - 2.0 * rand[:, 4])).unsqueeze(-1)
        B = (stheta*mod*torch.complex(torch.cos(2.0 * torch.pi * rand[:, 5]), torch.sin(2.0 * torch.pi * rand[:, 5]))).unsqueeze(-1)
        haar_a, haar_b = _su2_haar_from_uniform(rand)
        A = torch.where(k0, A.squeeze(-1), haar_a).unsqueeze(-1)
        B = torch.where(k0, B.squeeze(-1), haar_b).unsqueeze(-1)
        X = torch.cat((A, B), -1)
        v1 = torch.conj((X * _su2_sign(len(X.shape) - 1, X.device, X.real.dtype)).roll(1, -1))
        Xmat = torch.cat((X.unsqueeze(-2), v1.unsqueeze(-2)), -2)
        proposal = sun.SUN_mul(Xmat, sun.SUN_dagger(staples / safe_k.unsqueeze(-1).unsqueeze(-1)))
        proposal = torch.where(k0.unsqueeze(-1).unsqueeze(-1), proposal, Xmat)
        acc = (acc | ~k0).unsqueeze(-1).unsqueeze(-1)
        return proposal, acc

    def heatbath_defect(self, staples, rand, factor):
        return self.heatbath_standard(staples, rand, factor)

    def SU2_link_update_hb(self, cfgs, mu, rand, defect):
        staples = self.sum_of_staples(cfgs, mu, defect)
        
        Xmat, acc = self.heatbath(staples, rand[:,0:6,mu,:], 2.0)

        new_cfgs = Xmat * acc + cfgs[:, mu] * ~acc
        return new_cfgs.unsqueeze(1)

    def SU2_link_update_over(self, cfgs, mu, defect):
        staples = self.sum_of_staples(cfgs, mu, defect)
        k = torch.sqrt(sun.det_small(staples).abs())

        #this guarantees that links whose sum of staples has zero determinant do not generate nans
        k0 = k > 0
        k = k * k0 + torch.ones_like(k) * ~k0
        k0 = k0.unsqueeze(-1).unsqueeze(-1)

        staples = sun.SUN_dagger(staples / k.unsqueeze(-1).unsqueeze(-1))
        new_cfgs = sun.SUN_mul(sun.SUN_mul(staples, sun.SUN_dagger(cfgs[:, mu])), staples) * k0 + cfgs[:, mu] * ~k0
        return new_cfgs.unsqueeze(1)

    def _sum_of_staples_for_update(self, cfgs, mu, defect):
        if torch.is_grad_enabled() and cfgs.requires_grad:
            return checkpoint.checkpoint(
                self.sum_of_staples,
                cfgs,
                mu,
                defect,
                use_reentrant=False,
            )
        return self.sum_of_staples(cfgs, mu, defect)

    def SU3_link_update_hb(self, cfgs, mu, rand, defect):
        staples = self._sum_of_staples_for_update(cfgs, mu, defect)
        phb_steps = 3
        new_cfgs = cfgs[:,mu]
        sel = torch.tensor([0, 1, 2], device=new_cfgs.device)
        for phb in range(phb_steps):
            Rk = SU3_compute_RK(new_cfgs, staples, sel, phb)
            Ak, acc = self.heatbath(Rk, rand[:,6*phb:6*(phb+1),mu,:], 3.0 / 2.0)
            Ak = self.SU2toSU3(Ak, phb)
            new_cfgs = sun.SUN_mul(Ak, new_cfgs) * acc + new_cfgs * ~acc
        return new_cfgs.unsqueeze(1)

    def SU3_link_update_over(self, cfgs, mu, defect):
        staples = self._sum_of_staples_for_update(cfgs, mu, defect)
        phb_steps = 3
        new_cfgs = cfgs[:,mu]
        sel = torch.tensor([0, 1, 2], device=new_cfgs.device)
        for phb in range(phb_steps):
            Rk = SU3_compute_RK(new_cfgs, staples, sel, phb)
            k = torch.sqrt(sun.det_small(Rk).abs())

            #this guarantees that links whose sum of staples has zero determinant do not generate nans
            k0 = k > 0
            k = k * k0 + torch.ones_like(k) * ~k0

            Rk = Rk / k.unsqueeze(-1).unsqueeze(-1)
            i0 = torch.tensor([0], device=Rk.device)
            i1 = torch.tensor([1], device=Rk.device)
            a = (SU2_element(Rk,i1,i1) * SU2_element(Rk,i1,i1) + SU2_element(Rk,i0,i1) * SU2_element(Rk,i1,i0)).unsqueeze(-1)
            b = (-SU2_element(Rk,i1,i1) * SU2_element(Rk,i0,i1) - SU2_element(Rk,i0,i1) * SU2_element(Rk,i0,i0)).unsqueeze(-1)
            v0 = torch.cat((a,b),-1)
            v1 = torch.conj((v0 * _su2_sign(len(v0.shape) - 1, v0.device, v0.real.dtype)).roll(1,-1))
            Ak = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
            #bring Ak back to SU(3)
            Ak = self.SU2toSU3(Ak, phb)

            #new link is Ak*old_link
            #k=0 cases are left untouched
            k0 = k0.unsqueeze(-1).unsqueeze(-1)
            new_cfgs = sun.SUN_mul(Ak, new_cfgs) * k0 + new_cfgs * ~k0
        return new_cfgs.unsqueeze(1)

    #haar measure sampling (UNTESTED)
    def haar_heatbath(self, rand, cfgs_shape):
        x0 = rand[:,0,:]
        acc = (1.0 - x0**2 >= rand[:,3,:]**2)

        x0 = x0 * acc
        stheta = torch.sqrt(1.0 - rand[:,4,:]**2)
        mod = torch.sqrt(1.0 - x0**2)

        A = torch.complex(x0, mod*rand[:,4,:]).unsqueeze(-1)
        B = (stheta*mod*torch.complex(torch.cos(rand[:,5,:]), torch.sin(rand[:,5,:]))).unsqueeze(-1)
        Xa = torch.ones(cfgs_shape[:-2] + (1,), dtype=A.dtype, device=A.device) * A
        Xb = torch.ones(cfgs_shape[:-2] + (1,), dtype=B.dtype, device=B.device) * B
        v0 = torch.cat((Xa,Xb),-1)

        v1 = torch.conj((v0 * _su2_sign(len(v0.shape) - 1, v0.device, v0.real.dtype)).roll(1,-1))
        Xmat = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
        acc = acc.unsqueeze(-1).unsqueeze(-1)

        return Xmat, acc

    def SU2_link_haar_hb(self, cfgs, rand):
        Xmat, acc = self.haar_heatbath(rand[:,0:6], cfgs.shape)
        new_cfgs = Xmat * acc + cfgs * ~acc
        return new_cfgs

    def SU3_link_haar_hb(self, cfgs, rand):
        phb_steps = 3
        new_cfgs = cfgs
        for phb in range(phb_steps):
            Xmat, acc = self.haar_heatbath(rand[:,6*phb:6*(phb+1),:], cfgs.shape)
            Ak = sun.SU2toSU3(Xmat, phb)
            new_cfgs = sun.SUN_mul(Ak, new_cfgs) * acc + new_cfgs * ~acc

        return new_cfgs

def SU2_element(mat, i, j):
    return torch.index_select(torch.index_select(mat,-2,i),-1,j).squeeze(-1).squeeze(-1)

def SU3_compute_RK(new_cfgs, staples, sel, phb):
    i, j = (color for color in range(3) if color != phb)

    def element(row, col):
        return (new_cfgs[..., row, :] * staples[..., :, col]).sum(-1)

    # Only four entries of new @ staple are needed for the SU(2) subgroup matrix.
    a = 0.5 * (element(i, i) + element(j, j).conj())
    b = 0.5 * (element(i, j) - element(j, i).conj())
    first = torch.stack((a, b), dim=-1)
    second = torch.stack((-b.conj(), a.conj()), dim=-1)
    return torch.stack((first, second), dim=-2)


def SU2_left_multiply_SU3(links, subgroup, phb):
    """Apply an SU(2) subgroup update to just the two affected SU(3) rows."""
    pair = _constant(("pair", int(phb), str(links.device)),
                     lambda: torch.tensor([i for i in range(3) if i != phb], device=links.device))
    rows = links.index_select(-2, pair)
    updated_rows = torch.matmul(subgroup, rows)
    return links.index_copy(-2, pair, updated_rows)


class DefectMCMCAdapter:
    """
    Runtime-parameterized defect MCMC/action adapter.

    This exposes the same interface used by O-REX and NEO-REX:
    - set_parameter(parameter, parameter_name=...)
    - action(x, parameter, parameter_name=...)
    - __call__(x, nsteps=..., ...)

    `patch_mask` (Defect.protocol_patch_mask) restricts the HB+OR updates to a box around the
    defect; `symmetric_updates` applies the sweep or its reverse with probability 1/2 (one coin
    per sweep, shared by the batch), so that each update kernel is self-adjoint. Both follow the
    protocol updates of the O-REX paper; the equilibrium sweeps between swaps are not affected.
    """

    def __init__(
        self,
        flow_pars,
        beta: float,
        defect_mask: torch.Tensor,
        batch_size: int,
        parameter_name: str = "bc",
        default_parameter: float = 1.0,
        patch_mask: Optional[torch.Tensor] = None,
        symmetric_updates: bool = False,
    ):
        self.flow_pars = flow_pars
        self.beta = float(beta)
        self.defect_mask = defect_mask
        self.batch_size = int(batch_size)
        self.parameter_name = str(parameter_name)
        self.symmetric_updates = bool(symmetric_updates)
        self.patch_mask = patch_mask
        # One tensor, kept for the whole run, so the masked-link index cache hits on its identity.
        self._update_mask = (
            self.flow_pars.mask
            if patch_mask is None
            else (self.flow_pars.mask * patch_mask.to(device=self.flow_pars.mask.device)).contiguous()
        )
        self._current_parameter: Optional[torch.Tensor] = torch.full(
            (self.batch_size,),
            float(default_parameter),
            dtype=torch.float64,
        )
        self._hbor_cache: dict[float, HBOR] = {}
        self._action_cache: dict[float, Wilson_action] = {}
        self._batch_hbor = HBOR(
            self.flow_pars,
            beta=self.beta,
            defect_par=float(default_parameter),
        )
        self._nf_halo_plaquette_masks = None
        if getattr(self.flow_pars, "defect", None) is not None:
            self._nf_halo_plaquette_masks = build_nf_halo_plaquette_masks(
                self.flow_pars,
                self.defect_mask,
                device=self.flow_pars.device,
            )
        self._defect_plaquettes = None  # built on first use: see _defect_plaquette_tables

    def _defect_plaquette_tables(self, device):
        """Flat link indices of the plaquettes whose weight depends on bc: one (4, n) table per
        orientation (mu, nu), rows U_mu(x), U_nu(x+mu), U_mu(x+nu), U_nu(x), x+mu = roll(+1) as in
        pstaple. Only these plaquettes contribute to S(x; bc') - S(x; bc)."""
        if self._defect_plaquettes is not None and self._defect_plaquettes[0] == str(device):
            return self._defect_plaquettes[1]
        D = int(self.flow_pars.D)
        defect = self.defect_mask.to(device=device)[1].to(torch.bool)  # (D, T, L, ...): links with kappa < 1
        lattice = tuple(defect.shape[1:])
        V = int(defect[0].numel())
        ids = torch.arange(V, device=device).reshape(lattice)
        shift = lambda axis: torch.roll(ids, 1, dims=axis).flatten()
        tables = []
        for nu in range(1, D):
            for mu in range(0, nu):
                depends = (defect[mu] | torch.roll(defect[nu], 1, dims=mu) | torch.roll(defect[mu], 1, dims=nu) | defect[nu])
                sites = depends.flatten().nonzero().squeeze(-1)
                table = torch.stack((mu * V + sites, nu * V + shift(mu)[sites], mu * V + shift(nu)[sites], nu * V + sites))
                tables.append((mu, nu, table))
        self._defect_plaquettes = (str(device), tables)
        return tables

    def _normalize_parameter(
        self,
        parameter: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        expected_batch_size: Optional[int] = None,
        allow_any_batch: bool = False,
    ) -> torch.Tensor:
        p = torch.as_tensor(parameter, dtype=torch.float64)
        expected = self.batch_size if expected_batch_size is None else int(expected_batch_size)
        if p.dim() == 0:
            p = p.expand(expected)
        if p.dim() != 1:
            raise ValueError(
                f"parameter must be a scalar or 1D tensor, got {tuple(p.shape)}"
            )
        if not allow_any_batch and int(p.shape[0]) != expected:
            raise ValueError(
                f"parameter must have shape ({expected},), got {tuple(p.shape)}"
            )
        if device is not None:
            p = p.to(device=device)
        return p.detach()

    def _get_hbor(self, defect_par: float) -> HBOR:
        if defect_par not in self._hbor_cache:
            self._hbor_cache[defect_par] = HBOR(
                self.flow_pars,
                beta=self.beta,
                defect_par=defect_par,
            )
        return self._hbor_cache[defect_par]

    def _get_action(self, defect_par: float) -> Wilson_action:
        if defect_par not in self._action_cache:
            self._action_cache[defect_par] = Wilson_action(
                self.flow_pars,
                beta=self.beta,
                defect_par=defect_par,
            )
        return self._action_cache[defect_par]

    def set_parameter(self, parameter: torch.Tensor, parameter_name: str = "bc"):
        if str(parameter_name) != self.parameter_name:
            raise ValueError(
                f"Unsupported parameter_name={parameter_name!r}, expected {self.parameter_name!r}"
            )
        self._current_parameter = self._normalize_parameter(parameter, allow_any_batch=True)

    def action(
        self,
        x: torch.Tensor,
        parameter: torch.Tensor,
        parameter_name: str = "bc",
    ) -> torch.Tensor:
        if str(parameter_name) != self.parameter_name:
            raise ValueError(
                f"Unsupported parameter_name={parameter_name!r}, expected {self.parameter_name!r}"
            )
        p = self._normalize_parameter(
            parameter,
            device=x.device,
            expected_batch_size=int(x.shape[0]),
        )
        out = torch.empty(x.shape[0], device=x.device, dtype=torch.float64)
        defect_mask = self.defect_mask.to(device=x.device)
        for p_val in torch.unique(p):
            sel = torch.nonzero(p == p_val, as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                continue
            act = self._get_action(float(p_val.item()))
            vals = act(x.index_select(0, sel), defect_mask).to(dtype=out.dtype)
            out.index_copy_(0, sel, vals)
        return out

    def _kappa_for_parameter(self, parameter: torch.Tensor, defect_mask: torch.Tensor) -> torch.Tensor:
        defect = defect_mask.to(device=parameter.device, dtype=parameter.dtype)
        view_shape = (int(parameter.numel()),) + (1,) * (defect.dim() - 1)
        p = parameter.view(view_shape)
        return defect[0].unsqueeze(0) + defect[1].unsqueeze(0) * p

    def action_difference(
        self,
        x: torch.Tensor,
        parameter_old: torch.Tensor,
        parameter_new: torch.Tensor,
        parameter_name: str = "bc",
        full_lattice: bool = False,
    ) -> torch.Tensor:
        """S(x; parameter_new) - S(x; parameter_old) per chain. By default only the plaquettes whose
        weight depends on bc are evaluated (exact: the others cancel); full_lattice=True multiplies
        every plaquette, the reference implementation."""
        if str(parameter_name) != self.parameter_name:
            raise ValueError(
                f"Unsupported parameter_name={parameter_name!r}, expected {self.parameter_name!r}"
            )
        p_old = self._normalize_parameter(
            parameter_old,
            device=x.device,
            expected_batch_size=int(x.shape[0]),
        )
        p_new = self._normalize_parameter(
            parameter_new,
            device=x.device,
            expected_batch_size=int(x.shape[0]),
        )
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        p_old = p_old.to(dtype=work_dtype)
        p_new = p_new.to(dtype=work_dtype)
        defect_mask = self.defect_mask.to(device=x.device, dtype=work_dtype)
        kappa_old = self._kappa_for_parameter(p_old, defect_mask)
        kappa_new = self._kappa_for_parameter(p_new, defect_mask)
        if not full_lattice:
            # Only the plaquettes with a defect link change weight: gather their four links and
            # the four couplings per chain instead of multiplying every plaquette of the lattice.
            flat = sun.flat_links(x, self.flow_pars.D)
            dk = (kappa_old - kappa_new).flatten(1)  # (B, D * V)
            delta = torch.zeros(x.shape[0], device=x.device, dtype=work_dtype)
            for mu, nu, table in self._defect_plaquette_tables(x.device):
                if table.shape[1] == 0:
                    continue
                U = flat.index_select(1, table.flatten()).unflatten(1, table.shape)
                U_mu, U_nu_xpmu, U_mu_xpnu, U_nu = U.unbind(1)
                plaq = sun.SUN_mul(sun.SUN_mul(U_mu, U_nu_xpmu), sun.SUN_dagger(sun.SUN_mul(U_nu, U_mu_xpnu)))
                k_old = kappa_old.flatten(1).index_select(1, table.flatten()).unflatten(1, table.shape).prod(dim=1)
                k_new = kappa_new.flatten(1).index_select(1, table.flatten()).unflatten(1, table.shape).prod(dim=1)
                delta = delta + ((k_old - k_new) * sun.SUN_trace(plaq).real / float(self.flow_pars.N)).sum(dim=1)
            del dk
            return delta * float(self.beta)
        delta = torch.zeros(x.shape[0], device=x.device, dtype=work_dtype)
        dims = tuple(range(1, self.flow_pars.D + 1))

        for nu in range(1, self.flow_pars.D):
            for mu in range(0, nu):
                plaq = sun.SUN_mul(sun.pstaple(x, mu, nu, self.flow_pars.D), x[:, mu])
                retr = sun.SUN_trace(plaq).real
                weight_old = (
                    kappa_old[:, mu]
                    * torch.roll(kappa_old, 1, dims=(-self.flow_pars.D + mu))[:, nu]
                    * torch.roll(kappa_old, 1, dims=(-self.flow_pars.D + nu))[:, mu]
                    * kappa_old[:, nu]
                )
                weight_new = (
                    kappa_new[:, mu]
                    * torch.roll(kappa_new, 1, dims=(-self.flow_pars.D + mu))[:, nu]
                    * torch.roll(kappa_new, 1, dims=(-self.flow_pars.D + nu))[:, mu]
                    * kappa_new[:, nu]
                )
                delta = delta + torch.sum((weight_old - weight_new) * retr / float(self.flow_pars.N), dims)

        return delta * float(self.beta)

    def local_nf_action_change(
        self,
        x_old: torch.Tensor,
        x_new: torch.Tensor,
        parameter_old: torch.Tensor,
        parameter_new: torch.Tensor,
        parameter_name: str = "bc",
    ) -> torch.Tensor:
        if str(parameter_name) != self.parameter_name:
            raise ValueError(
                f"Unsupported parameter_name={parameter_name!r}, expected {self.parameter_name!r}"
            )
        p_old = self._normalize_parameter(
            parameter_old,
            device=x_old.device,
            expected_batch_size=int(x_old.shape[0]),
        )
        p_new = self._normalize_parameter(
            parameter_new,
            device=x_old.device,
            expected_batch_size=int(x_old.shape[0]),
        )
        if self._nf_halo_plaquette_masks is None:
            raise RuntimeError("local_nf_action_change requires flow_pars.defect")
        return theory_local_nf_action_change(
            x_old=x_old,
            x_new=x_new,
            defect_mask=self.defect_mask,
            beta=self.beta,
            parameter_old=p_old,
            parameter_new=p_new,
            plaquette_masks=self._nf_halo_plaquette_masks,
            D=self.flow_pars.D,
            N=self.flow_pars.N,
        )

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        nsteps: int = 1,
        ensembles: bool = False,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        del ensembles, use_tqdm
        if self._current_parameter is None:
            raise RuntimeError("Call set_parameter(...) before using DefectMCMCAdapter.")

        p = self._normalize_parameter(
            self._current_parameter,
            device=x.device,
            expected_batch_size=int(x.shape[0]),
        )
        out = x
        mask = self._update_mask.to(device=x.device)
        defect_mask = self.defect_mask.to(device=x.device)
        for _ in range(int(nsteps)):
            self._batch_hbor.set_defect_parameter(p)
            reverse = self.symmetric_updates and bool(torch.rand((), device=x.device) < 0.5)
            out = self._batch_hbor(out, mask, defect_mask, reverse=reverse)
        return out
