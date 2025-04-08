from mpi4py import MPI
import numpy as np
import time

comp_per_process = 50


def computePiSeriesPart(start: np.uint, end: np.uint) -> float:

    pi_part: float = 0

    for k in range(start, end):
        r1 = 4 / (8 * k + 1)
        r2 = -2 / (8 * k + 4)
        r3 = -1 / (8 * k + 5)
        r4 = -1 / (8 * k + 6)

        pi_part += (r1 + r2 + r3 + r4) / (16**k)

    return pi_part


rank = np.uint(MPI.COMM_WORLD.Get_rank())
worldSize = MPI.COMM_WORLD.Get_size()

start_time = time.time()

if rank == worldSize - 1:
    status = MPI.Status()

    pi: float = 0
    for _ in range(0, worldSize - 1):
        MPI.COMM_WORLD.Probe(status=status)
        pi += MPI.COMM_WORLD.recv(source=status.Get_source())

    print(f"pi: {pi} in {time.time() - start_time} seconds")

else:
    start = rank * comp_per_process
    end = start + comp_per_process
    pi_part = computePiSeriesPart(start, end)

    MPI.COMM_WORLD.send(pi_part, worldSize - 1)
