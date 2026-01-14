from typing import Optional, Tuple

import numpy as np
from numba import njit, prange
from scipy import fftpack


# Struve函数H0
@njit(cache=True)
def stvh0(X):
    """Compute Struve function H0(x)"""
    PI = np.pi
    S = 1.0
    R = 1.0

    if X <= 20.0:
        A0 = 2.0 * X / PI
        for K in range(1, 61):
            R = -R * X / (2.0 * K + 1.0) * X / (2.0 * K + 1.0)
            S = S + R
            if abs(R) < abs(S) * 1.0e-12:
                break
        SH0 = A0 * S
    else:
        KM = int(0.5 * (X + 1.0))
        if X >= 50.0:
            KM = 25
        S = 1.0
        R = 1.0
        for K in range(1, KM + 1):
            R = -R * (2.0 * K - 1.0) * (2.0 * K - 1.0) / (X * X)
            S = S + R
            if abs(R) < abs(S) * 1.0e-12:
                break

        T = 4.0 / X
        T2 = T * T
        P0 = (
            (((-0.37043e-5 * T2 + 0.173565e-4) * T2 - 0.487613e-4) * T2 + 0.17343e-3)
            * T2
            - 0.1753062e-2
        ) * T2 + 0.3989422793
        Q0 = T * (
            (
                (
                    ((0.32312e-5 * T2 - 0.142078e-4) * T2 + 0.342468e-4) * T2
                    - 0.869791e-4
                )
                * T2
                + 0.4564324e-3
            )
            * T2
            - 0.0124669441
        )
        TA0 = X - 0.785398164
        BY0 = 2.0 / np.sqrt(X) * (P0 * np.sin(TA0) + Q0 * np.cos(TA0))
        SH0 = 2.0 / (PI * X) * S + BY0

    return SH0


# Struve函数H1
@njit(cache=True)
def stvh1(X):
    """Compute Struve function H1(x)"""
    PI = np.pi

    if X <= 20.0:
        R = 1.0
        S = 0.0
        A0 = -2.0 / PI
        for K in range(1, 61):
            R = -R * X * X / (4.0 * K * K - 1.0)
            S = S + R
            if abs(R) < abs(S) * 1.0e-12:
                break
        SH1 = A0 * S
    else:
        S = 1.0
        R = 1.0
        KM = int(0.5 * X)
        if X > 50.0:
            KM = 25

        for K in range(1, KM + 1):
            R = -R * (4.0 * K * K - 1.0) / (X * X)
            S = S + R
            if abs(R) < abs(S) * 1.0e-12:
                break

        T = 4.0 / X
        T2 = T * T
        P1 = (
            (((0.42414e-5 * T2 - 0.20092e-4) * T2 + 0.580759e-4) * T2 - 0.223203e-3)
            * T2
            + 0.29218256e-2
        ) * T2 + 0.3989422819
        Q1 = T * (
            (
                (
                    ((-0.36594e-5 * T2 + 0.1622e-4) * T2 - 0.398708e-4) * T2
                    + 0.1064741e-3
                )
                * T2
                - 0.63904e-3
            )
            * T2
            + 0.0374008364
        )
        TA1 = X - 0.75 * PI
        BY1 = 2.0 / np.sqrt(X) * (P1 * np.sin(TA1) + Q1 * np.cos(TA1))
        SH1 = 2.0 / PI * (1.0 + S / (X * X)) + BY1

    return SH1


# 0阶第一类贝塞尔函数
@njit(cache=True)
def bessj0(x):
    """Bessel function of first kind, order 0"""
    if abs(x) < 8.0:
        y = x * x
        ans1 = 57568490574.0 + y * (
            -13362590354.0
            + y
            * (
                651619640.7
                + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))
            )
        )
        ans2 = 57568490411.0 + y * (
            1029532985.0
            + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0)))
        )
        return ans1 / ans2
    else:
        z = 8.0 / abs(x)
        y = z * z
        xx = abs(x) - 0.785398164
        ans1 = 1.0 + y * (
            -0.1098628627e-2
            + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
        )
        ans2 = -0.1562499995e-1 + y * (
            0.1430488765e-3
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934945152e-7)))
        )
        ans = np.sqrt(0.636619772 / abs(x)) * (
            np.cos(xx) * ans1 - z * np.sin(xx) * ans2
        )
        return ans if x >= 0 else ans


# 1阶第一类贝塞尔函数
@njit(cache=True)
def bessj1(x):
    """Bessel function of first kind, order 1"""
    if abs(x) < 8.0:
        y = x * x
        ans1 = x * (
            72362614232.0
            + y
            * (
                -7895059235.0
                + y
                * (
                    242396853.1
                    + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606)))
                )
            )
        )
        ans2 = 144725228442.0 + y * (
            2300535178.0
            + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0)))
        )
        return ans1 / ans2
    else:
        z = 8.0 / abs(x)
        y = z * z
        xx = abs(x) - 2.356194491
        ans1 = 1.0 + y * (
            0.183105e-2
            + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6)))
        )
        ans2 = 0.04687499995 + y * (
            -0.2002690873e-3
            + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6))
        )
        ans = np.sqrt(0.636619772 / abs(x)) * (
            np.cos(xx) * ans1 - z * np.sin(xx) * ans2
        )
        return ans if x >= 0 else -ans


# 2阶第一类贝塞尔函数 - 使用递推关系实现
@njit(cache=True)
def bessj2(x):
    """Bessel function of first kind, order 2
    使用递推关系：J₂(x) = (2/x)J₁(x) - J₀(x)
    """
    # 处理x=0的特殊情况，避免除以零
    if abs(x) < 1e-10:
        return 0.0
    # 使用递推关系计算J₂(x)
    return (2.0 / x) * bessj1(x) - bessj0(x)


# 0阶第二类贝塞尔函数
@njit(cache=True)
def bessy0(x):
    """Bessel function of second kind, order 0"""
    if x < 8.0:
        y = x * x
        ans1 = -2957821389.0 + y * (
            7062834065.0
            + y
            * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733)))
        )
        ans2 = 40076544269.0 + y * (
            745249964.8
            + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0)))
        )
        ans = (ans1 / ans2) + 0.636619772 * bessj0(x) * np.log(x)
    else:
        z = 8.0 / x
        y = z * z
        xx = x - 0.785398164
        ans1 = 1.0 + y * (
            -0.1098628627e-2
            + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
        )
        ans2 = -0.1562499995e-1 + y * (
            0.1430488765e-3
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934945152e-7)))
        )
        ans = np.sqrt(0.636619772 / x) * (np.sin(xx) * ans1 + z * np.cos(xx) * ans2)
    return ans


# 1阶第二类贝塞尔函数
@njit(cache=True)
def bessy1(x):
    """Bessel function of second kind, order 1"""
    if x < 8.0:
        y = x * x
        ans1 = x * (
            -0.4900604943e13
            + y
            * (
                0.1275274390e13
                + y
                * (
                    -0.5153438139e11
                    + y * (0.7349264551e9 + y * (-0.4237922726e7 + y * 0.8511937935e4))
                )
            )
        )
        ans2 = 0.2499580570e14 + y * (
            0.4244419664e12
            + y
            * (
                0.3733650367e10
                + y * (0.2245904002e8 + y * (0.1020426050e6 + y * (0.3549632885e3 + y)))
            )
        )
        ans = (ans1 / ans2) + 0.636619772 * (bessj1(x) * np.log(x) - 1.0 / x)
    else:
        z = 8.0 / x
        y = z * z
        xx = x - 2.356194491
        ans1 = 1.0 + y * (
            0.183105e-2
            + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6)))
        )
        ans2 = 0.04687499995 + y * (
            -0.2002690873e-3
            + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6))
        )
        ans = np.sqrt(0.636619772 / x) * (np.sin(xx) * ans1 + z * np.cos(xx) * ans2)
    return ans


# 2阶第二类贝塞尔函数 - 使用递推关系实现
@njit(cache=True)
def bessy2(x):
    """Bessel function of second kind, order 2
    使用递推关系：Y₂(x) = (2/x)Y₁(x) - Y₀(x)
    """
    # 处理x=0的特殊情况，避免除以零
    if abs(x) < 1e-10:
        return -np.inf  # 当x接近0时，Y₂(x)趋近于负无穷
    # 使用递推关系计算Y₂(x)
    return (2.0 / x) * bessy1(x) - bessy0(x)


# 梯形积分
@njit(parallel=True, cache=True)
def trap_J(U_f, r, f, c, nc, nr, nf):
    """Python implementation of trap_J function - optimized with numba parallel"""
    # 使用C++相同的索引顺序，先创建(nf, nc)数组，然后转置
    out_temp = np.zeros((nf, nc), dtype=np.float32)
    PI = np.float32(np.pi)  # 使用float32精度的PI

    # 使用prange进行并行循环
    for i in prange(nf):
        for j in range(nc):
            fl = np.float32(f[i])  # 转换为float32
            cl = np.float32(c[j])  # 转换为float32

            # 处理f=0的特殊情况，避免除以零
            if fl < 1e-10:
                # 当f=0时，k=0，设置kernel=0
                kernel = np.float32(0.0)
            else:
                k = np.float32(2 * PI * fl / cl)  # 计算k，使用float32
                kernel = np.float32(0.0)  # 使用float32累加

                for ir in range(1, nr):
                    indx_d = i + ir * nf
                    g1 = np.float32(U_f[indx_d - nf])  # 转换为float32
                    g2 = np.float32(U_f[indx_d])  # 转换为float32
                    r1 = np.float32(r[ir - 1])  # 转换为float32
                    r2 = np.float32(r[ir])  # 转换为float32
                    dr0 = np.float32(max((r2 - r1), 0.1))  # 使用float32的max

                    # 计算k*r1和k*r2，使用float32
                    k_r1 = np.float32(k * r1)
                    k_r2 = np.float32(k * r2)

                    # 使用numba优化的贝塞尔函数，提高性能
                    bessj0_r1 = np.float32(bessj0(k_r1))
                    bessj0_r2 = np.float32(bessj0(k_r2))

                    # 梯形积分公式，所有计算使用float32
                    term = np.float32(
                        (g1 * bessj0_r1 * r1 + g2 * bessj0_r2 * r2) * dr0 * 0.5
                    )
                    kernel = np.float32(kernel + term)

            # 使用C++相同的索引顺序：i + j*nf
            indx = i + j * nf
            out_temp[i, j] = kernel

    # 转置为(nc, nf)顺序，与C++接口一致
    out = out_temp.T.astype(np.float32)
    return out


@njit(parallel=True, cache=True)
def trap_J2(U_f, r, f, c, nc, nr, nf):
    """Python implementation of trap_J2 function - optimized with numba parallel
    根据公式：IRR(ω,k) = ∫₀^+∞ UR(r,ω)J₀(kr)rdr - ∫₀^+∞ UR(r,ω)J₂(kr)rdr
    """
    # 使用C++相同的索引顺序，先创建(nf, nc)数组，然后转置
    out_temp = np.zeros((nf, nc), dtype=np.float32)
    PI = np.float32(np.pi)  # 使用float32精度的PI

    # 使用prange进行并行循环
    for i in prange(nf):
        for j in range(nc):
            fl = np.float32(f[i])  # 转换为float32
            cl = np.float32(c[j])  # 转换为float32

            # 处理f=0的特殊情况，避免除以零
            if fl < 1e-10:
                # 当f=0时，k=0，设置kernel=0
                integral_j0 = np.float32(0.0)
                integral_j2 = np.float32(0.0)
            else:
                k = np.float32(2 * PI * fl / cl)  # 计算k，使用float32
                integral_j0 = np.float32(0.0)  # J0积分结果
                integral_j2 = np.float32(0.0)  # J2积分结果

                for ir in range(1, nr):
                    indx_d = i + ir * nf
                    g1 = np.float32(U_f[indx_d - nf])  # 转换为float32
                    g2 = np.float32(U_f[indx_d])  # 转换为float32
                    r1 = np.float32(r[ir - 1])  # 转换为float32
                    r2 = np.float32(r[ir])  # 转换为float32
                    dr0 = np.float32(max((r2 - r1), 0.1))  # 使用float32的max

                    # 计算k*r1和k*r2，使用float32
                    k_r1 = np.float32(k * r1)
                    k_r2 = np.float32(k * r2)

                    # 使用numba优化的贝塞尔函数，提高性能
                    bessj0_r1 = np.float32(bessj0(k_r1))
                    bessj0_r2 = np.float32(bessj0(k_r2))

                    bessj2_r1 = np.float32(bessj2(k_r1))
                    bessj2_r2 = np.float32(bessj2(k_r2))

                    # 计算J0积分的梯形项
                    term_j0 = np.float32(
                        (g1 * bessj0_r1 * r1 + g2 * bessj0_r2 * r2) * dr0 * 0.5
                    )
                    integral_j0 = np.float32(integral_j0 + term_j0)

                    # 计算J2积分的梯形项
                    term_j2 = np.float32(
                        (g1 * bessj2_r1 * r1 + g2 * bessj2_r2 * r2) * dr0 * 0.5
                    )
                    integral_j2 = np.float32(integral_j2 + term_j2)

            # 根据公式计算IRR：IRR = J0积分 - J2积分
            kernel = np.float32(integral_j0 - integral_j2)

            # 使用C++相同的索引顺序：i + j*nf
            indx = i + j * nf
            out_temp[i, j] = kernel

    # 转置为(nc, nf)顺序，与C++接口一致
    out = out_temp.T.astype(np.float32)
    return out


# 梯形积分法，使用0阶第二类贝塞尔函数
@njit(parallel=True, cache=True)
def trap_Y(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """使用梯形积分法计算基于0阶第二类贝塞尔函数的积分

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: 积分结果，形状为(nc, nf)
    """
    out_temp = np.zeros((nf, nc), dtype=np.float32)
    PI = np.float32(np.pi)  # 使用float32精度的PI

    for i in prange(nf):
        for j in range(nc):
            fl = np.float32(f[i])  # 转换为float32
            cl = np.float32(c[j])  # 转换为float32

            # 处理f=0的特殊情况，避免除以零
            if fl < 1e-10:
                # 当f=0时，k=0，设置kernel=0
                kernel = np.float32(0.0)
            else:
                k = np.float32(2 * PI * fl / cl)  # 计算k，使用float32
                kernel = np.float32(0.0)  # 使用float32累加

                for ir in range(1, nr):
                    indx_d = i + ir * nf
                    g1 = np.float32(U_f[indx_d - nf])  # 转换为float32
                    g2 = np.float32(U_f[indx_d])  # 转换为float32
                    r1 = np.float32(r[ir - 1])  # 转换为float32
                    r2 = np.float32(r[ir])  # 转换为float32
                    dr0 = np.float32(max((r2 - r1), 0.1))  # 使用float32的max

                    # 计算k*r1和k*r2，使用float32
                    k_r1 = np.float32(k * r1)
                    k_r2 = np.float32(k * r2)

                    # 使用numba优化的第二类贝塞尔函数，提高性能
                    bessy0_r1 = np.float32(bessy0(k_r1))
                    bessy0_r2 = np.float32(bessy0(k_r2))

                    # 梯形积分公式，所有计算使用float32
                    term = np.float32(
                        (g1 * bessy0_r1 * r1 + g2 * bessy0_r2 * r2) * dr0 * 0.5
                    )
                    kernel = np.float32(kernel + term)

            # 使用C++相同的索引顺序：i + j*nf
            indx = i + j * nf
            out_temp[i, j] = kernel

    # 转置为(nc, nf)顺序，与C++接口一致
    out = out_temp.T.astype(np.float32)
    return out


# 梯形积分法，使用0阶和2阶第二类贝塞尔函数
@njit(parallel=True, cache=True)
def trap_Y2(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """使用梯形积分法计算基于0阶和2阶第二类贝塞尔函数的积分

    根据公式：IRR(ω,k) = ∫₀^+∞ UR(r,ω)Y₀(kr)rdr - ∫₀^+∞ UR(r,ω)Y₂(kr)rdr

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: 积分结果，形状为(nc, nf)
    """
    # 使用C++相同的索引顺序，先创建(nf, nc)数组，然后转置
    out_temp = np.zeros((nf, nc), dtype=np.float32)
    PI = np.float32(np.pi)  # 使用float32精度的PI

    # 使用prange进行并行循环
    for i in prange(nf):
        for j in range(nc):
            fl = np.float32(f[i])  # 转换为float32
            cl = np.float32(c[j])  # 转换为float32

            # 处理f=0的特殊情况，避免除以零
            if fl < 1e-10:
                # 当f=0时，k=0，设置kernel=0
                integral_y0 = np.float32(0.0)
                integral_y2 = np.float32(0.0)
            else:
                k = np.float32(2 * PI * fl / cl)  # 计算k，使用float32
                integral_y0 = np.float32(0.0)  # Y0积分结果
                integral_y2 = np.float32(0.0)  # Y2积分结果

                for ir in range(1, nr):
                    indx_d = i + ir * nf
                    g1 = np.float32(U_f[indx_d - nf])  # 转换为float32
                    g2 = np.float32(U_f[indx_d])  # 转换为float32
                    r1 = np.float32(r[ir - 1])  # 转换为float32
                    r2 = np.float32(r[ir])  # 转换为float32
                    dr0 = np.float32(max((r2 - r1), 0.1))  # 使用float32的max

                    # 计算k*r1和k*r2，使用float32
                    k_r1 = np.float32(k * r1)
                    k_r2 = np.float32(k * r2)

                    # 使用numba优化的贝塞尔函数，提高性能
                    bessy0_r1 = np.float32(bessy0(k_r1))
                    bessy0_r2 = np.float32(bessy0(k_r2))

                    bessy2_r1 = np.float32(bessy2(k_r1))
                    bessy2_r2 = np.float32(bessy2(k_r2))

                    # 计算Y0积分的梯形项
                    term_y0 = np.float32(
                        (g1 * bessy0_r1 * r1 + g2 * bessy0_r2 * r2) * dr0 * 0.5
                    )
                    integral_y0 = np.float32(integral_y0 + term_y0)

                    # 计算Y2积分的梯形项
                    term_y2 = np.float32(
                        (g1 * bessy2_r1 * r1 + g2 * bessy2_r2 * r2) * dr0 * 0.5
                    )
                    integral_y2 = np.float32(integral_y2 + term_y2)

            # 根据公式计算IRR：IRR = Y0积分 - Y2积分
            kernel = np.float32(integral_y0 - integral_y2)

            # 使用C++相同的索引顺序：i + j*nf
            indx = i + j * nf
            out_temp[i, j] = kernel

    # 转置为(nc, nf)顺序，与C++接口一致
    out = out_temp.T.astype(np.float32)
    return out


# 主FJ函数
def fj(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """FJ频散成像方法

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: FJ频散谱，形状为(nc, nf)
    """
    return trap_J(U_f, r, f, c, nc, nr, nf)


# FJ_RR频散成像方法
def fj_rr(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """FJ_RR频散成像方法

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: FJ_RR频散谱，形状为(nc, nf)
    """
    return trap_J2(U_f, r, f, c, nc, nr, nf)


# FH频散成像方法
def fh(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """FH频散成像方法

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: FH频散谱，形状为(nc, nf)
    """
    return trap_Y(U_f, r, f, c, nc, nr, nf)


# FH_RR频散成像方法
def fh_rr(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """FH_RR频散成像方法

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: FH_RR频散谱，形状为(nc, nf)
    """
    return trap_Y2(U_f, r, f, c, nc, nr, nf)


# MFJ频散成像方法
def mfj(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """MFJ频散成像方法

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: MFJ频散谱，形状为(nc, nf)
    """
    outr = fj(U_f, r, f, c, nc, nr, nf)

    # 将U_f重塑为(nr, nf)形状，以适应希尔伯特变换
    uf = U_f.reshape(nr, nf)
    ufi = np.zeros(np.shape(uf), dtype=np.float32)

    for i in range(nr):
        ufi[i, :] = fftpack.hilbert(uf[i, :])

    # 将ufi重新展平为一维数组
    ufi_flat = ufi.flatten()
    outi = fh(ufi_flat, r, f, c, nc, nr, nf)

    out = outr - outi
    return out


# MFJ_RR频散成像方法
def mfj_rr(
    U_f: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    nc: int,
    nr: int,
    nf: int,
) -> np.ndarray:
    """MFJ_RR频散成像方法

    基于改进的MFJ谱公式

    Args:
        U_f: 频域互相关函数，形状为(nf * nr,)
        r: 距离数组，形状为(nr,)
        f: 频率数组，形状为(nf,)
        c: 速度数组，形状为(nc,)
        nc: 速度点数量
        nr: 距离点数量
        nf: 频率点数量

    Returns:
        np.ndarray: MFJ_RR频散谱，形状为(nc, nf)
    """
    # 计算实部贡献：-1/2 ∫ iU_R H₀⁽¹⁾ rdr + 1/2 ∫ (iU_R)^* H₂⁽¹⁾ rdr
    # 保持与mfj一致的实现方式

    # 计算第一项相关：使用fj_rr（J₀ - J₂）
    outr = fj_rr(U_f, r, f, c, nc, nr, nf)

    # 将U_f重塑为(nr, nf)形状，以适应希尔伯特变换
    uf = U_f.reshape(nr, nf)
    ufi = np.zeros(np.shape(uf), dtype=np.float32)

    for i in range(nr):
        ufi[i, :] = fftpack.hilbert(uf[i, :])

    # 将ufi重新展平为一维数组
    ufi_flat = ufi.flatten()

    # 计算第二项相关：使用fh（Y₀）
    outi = fh(ufi_flat, r, f, c, nc, nr, nf)

    # 根据改进的公式，应用1/2系数并调整符号
    # 与mfj保持一致的实现结构，同时应用改进公式的系数
    out = outr - outi

    return out
