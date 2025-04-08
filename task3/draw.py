from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy.typing as npt
from config import Config


def DrawSolution(res: npt.NDArray[np.float64], cfg: Config):

    X = np.linspace(cfg.x_0, cfg.x_1, cfg.M)
    T = np.linspace(cfg.t_0, cfg.t_1, cfg.K)
    x, t = np.meshgrid(X, T)
    print(res)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    # ax.view_init(30, 100)
    ax.plot_surface(x, t, res.transpose(), vmin=res.min() * 2, cmap=cm.coolwarm)

    ax.set_xlabel("x")

    ax.set_ylabel("t")
    ax.set_zlabel("z")

    plt.savefig("res.png")
