from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import time


rank = MPI.COMM_WORLD.Get_rank()


def send(array: npt.NDArray[np.int32]):
    MPI.COMM_WORLD.send(array, dest=1)


def recieve(sz: int) -> npt.NDArray[np.int32]:
    return MPI.COMM_WORLD.recv(source=0)


send_arr = np.array([1, 2, 3, 5, 6])

start_time = time.time()

if rank == 0:
    for _ in range(100000):
        send(send_arr)
else:
    for _ in range(100000):
        recieve(len(send_arr))

print(f"{rank}: {time.time() - start_time} seconds")
