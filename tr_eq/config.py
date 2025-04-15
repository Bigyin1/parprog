import numpy as np


class Config:

    K = 40  # time steps
    M = 40  # coord steps

    t_0 = np.float64(0.0)
    t_1 = np.float64(5.0)
    x_0 = np.float64(0.0)
    x_1 = np.float64(5.0)

    tau = (t_1 - t_0) / (K - 1)
    h = (x_1 - x_0) / (M - 1)

    coeff_a = np.float64(2.0)

    courant = coeff_a * tau / h

    @staticmethod
    def _mToFloat(m: int):
        return Config.x_0 + Config.h * (m)

    @staticmethod
    def _kToFloat(k: int):
        return Config.t_0 + Config.tau * (k)

    @staticmethod
    def f_x_t(offset: int, m: int, k: int):
        x = Config._mToFloat(offset + m)
        t = Config._kToFloat(k)

        return x + t

    @staticmethod
    def boundary_cond(k: int):
        t = Config._kToFloat(k)

        return np.exp(-t)

    @staticmethod
    def init_cond(offset: int, m: int):
        x = Config._mToFloat(offset + m)

        return np.cos(np.pi * x)
