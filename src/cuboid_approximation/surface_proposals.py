"""Large observed surface panels as a complement to maximal-volume proposals.

Each orientation partitions the cloud into thin projection layers once. Spatial
components and bounded rectangle splitting avoid joining disconnected sheets or
filling a concave patch's bounding rectangle. Candidates retain the same whole
volume certificate and later full-cloud union checks as every other proposal.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .approximation import spatial_weights
from .cuboids import certify_box
from .selection import adaptive_face_samples


class SurfacePanelProposer:
    def __init__(self, points, frames, tolerance, solid, config):
        self.tree = cKDTree(points)
        self.frames = np.asarray(frames)[:8]
        self.tolerance, self.solid, self.config = tolerance, solid, config
        self.budget = max(16, min(96, config.seeds_per_frame * 4))

    def _components(self, projected, ids, axis):
        tangent = [a for a in range(3) if a != axis]
        width = max(2 * self.tolerance, self.solid["voxel_size"])
        cells, inverse = np.unique(
            np.floor(projected[ids][:, tangent] / width).astype(np.int64),
            axis=0,
            return_inverse=True,
        )
        pairs = cKDTree(cells).query_pairs(np.sqrt(2) + 1e-8, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells))
        )
        count, labels = connected_components(graph, directed=False)
        labels = labels[inverse]
        order = np.argsort(labels, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
        return [ids[order[a:b]] for a, b in zip(starts[:-1], starts[1:]) if b - a >= 3]

    def __call__(self, points):
        if len(points) < 3:
            return []
        points = np.asarray(points)
        origin = points.min(axis=0)
        weights = spatial_weights(points, self.tolerance)
        minimum = self.solid["voxel_size"] * 1e-4
        per_frame = max(2, self.budget // max(1, len(self.frames)))
        output, seen = [], set()
        # The layer width bounds the panel's thickness, independent of acquisition
        # density or the number of points already covered by previous boxes.
        width = self.tolerance
        for frame in self.frames:
            projected = (points - origin) @ frame
            layers, layer_keys = [], set()
            for axis in range(3):
                for phase in (0.0, 0.5):
                    bins = np.floor(projected[:, axis] / width - phase).astype(np.int64)
                    order = np.argsort(bins, kind="stable")
                    _, starts, counts = np.unique(
                        bins[order], return_index=True, return_counts=True
                    )
                    mass = np.add.reduceat(weights[order], starts)
                    for position in np.argsort(-mass, kind="stable")[:4]:
                        ids = order[starts[position] : starts[position] + counts[position]]
                        if len(ids) < 3:
                            continue
                        identity = (
                            axis,
                            float(projected[ids, axis].min()),
                            float(projected[ids, axis].max()),
                        )
                        if identity not in layer_keys:
                            layer_keys.add(identity)
                            layers.append((float(mass[position]), axis, ids))
            accepted = 0
            for _, axis, ids in sorted(layers, key=lambda item: (-item[0], item[1]))[:8]:
                components = self._components(projected, ids, axis)
                components.sort(key=lambda members: -float(weights[members].sum()))
                pending = [(members, 0) for members in components[:8]]
                while pending and accepted < per_frame:
                    members, depth = pending.pop(0)
                    local = projected[members]
                    low, high = local.min(axis=0), local.max(axis=0)
                    dimensions = np.maximum(high - low, minimum)
                    if np.count_nonzero(high - low > minimum) < 2:
                        continue
                    box = dict(
                        center=origin + ((low + high) / 2) @ frame.T,
                        dimensions=dimensions,
                        rotation=frame.copy(),
                        volume=float(np.prod(dimensions)),
                        proposal_kind="surface_panel",
                    )
                    identity = tuple(np.round(np.r_[box["center"], dimensions, frame.ravel()], 10))
                    if identity in seen:
                        continue
                    certified = certify_box(box, self.solid)
                    supported = False
                    if certified:
                        samples, area = adaptive_face_samples(box, self.tolerance, budget=256)
                        distances = self.tree.query(samples)[0]
                        supported = np.average(distances <= self.tolerance, weights=area) >= max(
                            0.85, self.config.target_surface_support - 0.03
                        )
                    if certified and supported:
                        seen.add(identity)
                        output.append(box)
                        accepted += 1
                        continue
                    if depth >= 3 or len(members) < 6:
                        continue
                    tangent = [a for a in range(3) if a != axis]
                    split_axis = tangent[int(np.argmax(dimensions[tangent]))]
                    order = np.argsort(projected[members, split_axis], kind="stable")
                    cumulative = np.cumsum(weights[members[order]])
                    middle = int(np.searchsorted(cumulative, cumulative[-1] / 2))
                    cut = projected[members[order[min(middle, len(order) - 2)]], split_axis]
                    left = projected[members, split_axis] <= cut
                    if left.all() or not left.any():
                        continue
                    pending.extend(
                        (part, depth + 1)
                        for part in (members[left], members[~left])
                        if len(part) >= 3
                    )
                if accepted >= per_frame:
                    break
        return output
