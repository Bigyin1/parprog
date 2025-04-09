import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy.typing as npt
from config import Config


def DrawSolution(res: npt.NDArray[np.float64], cfg: Config):

    X = np.linspace(cfg.x_0, cfg.x_1, cfg.M)
    T = np.linspace(cfg.t_0, cfg.t_1, cfg.K)
    x, t = np.meshgrid(X, T)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    # ax.view_init(50, 15)
    ax.plot_surface(x, t, res, vmin=res.min(), cmap=cm.jet)

    ax.set_xlabel("x")

    ax.set_ylabel("t")
    ax.set_zlabel("z")

    plt.savefig("res.png")
