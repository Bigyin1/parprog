import numpy as np

from mpi4py import MPI
from config import Config


class Grid:
    def __init__(
        self,
        cfg: Config,
        mpi: MPI.Intracomm,
    ):
        self.mpi = mpi
        self.cfg = cfg

        self.M_i = cfg.M // mpi.Get_size()
        self.offset = mpi.Get_rank() * self.M_i

        self.npGrid = np.empty((cfg.K, self.M_i), dtype=np.float64)

        self._leftmostColumn = np.empty(
            cfg.K, dtype=np.float64
        )  # rightmost column of prev node
        self._nextLeftMostIdx = 0

        self._applyBoundaryCond()
        self._applyInitCond()

    def __getitem__(self, idx: tuple[int, int]):
        if idx[1] != -1:
            return self.npGrid[idx]

        if idx[0] < self._nextLeftMostIdx:
            return self._leftmostColumn[idx[0]]

        res = self._leftmostColumn[self._nextLeftMostIdx] = self.mpi.recv(
            source=self.mpi.Get_rank() - 1, tag=idx[0]
        )
        self._nextLeftMostIdx += 1

        return res

    def __setitem__(self, idx: tuple[int, int], val: np.float64):
        if self.mpi.Get_rank() != self.mpi.Get_size() - 1:
            if idx[1] == self.M_i - 1:
                self.mpi.isend(val, self.mpi.Get_rank() + 1, tag=idx[0])

        self.npGrid[idx] = val

    def _applyInitCond(self):
        for m in range(self.M_i):
            self[0, m] = self.cfg.init_cond(self.offset, m)

    def _applyBoundaryCond(self):
        for k in range(self.cfg.K):
            self[k, 0] = self.cfg.boundary_cond(k)
