# seismocorr/core/subarray.py

"""
SPFI Subarray Building Module

负责从节点（或通道）坐标生成 subarray（索引列表）,支持：
- geometry="2d": 随机 Voronoi（平面坐标，单位：米）
- geometry="1d": 随机滑窗 + 窗内随机抽取通道（沿线坐标，单位：米）
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from matplotlib.path import Path
from scipy.spatial import Voronoi
from seismocorr.config.default import SUPPORTED_GEOMETRY

Subarray = List[np.ndarray]


class SubarrayBuilder(ABC):
    """
    子阵列构建策略抽象基类。
    两种子阵列生成策略需继承并实现 subarray 方法。
    """

    @abstractmethod
    def subarray(
        self,
        sensor_xy: np.ndarray,
        *,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Subarray:
        """
        根据节点（或通道）坐标生成子阵列索引列表。

        Args:
            sensor_xy:
                - geometry="2d": (n,2) 平面坐标（米）
                - geometry="1d": (n,) 或 (n,1) 沿线坐标（米）
            random_state: 随机种子（保证可复现）
            **kwargs: 由具体实现决定

        Returns:
            Subarray: 子阵列索引列表
        """
        raise NotImplementedError

    def __call__(self, sensor_xy: np.ndarray, *, random_state: Optional[int] = None, **kwargs) -> Subarray:
        return self.subarray(sensor_xy, random_state=random_state, **kwargs)


class _Voronoi2DBuilder(SubarrayBuilder):
    """2D（适配节点地震仪数据或者非线性缆DAS数据）：随机 Voronoi 内节点构成子阵列。"""

    def subarray(
        self,
        sensor_xy: np.ndarray,
        *,
        n_realizations: int = 100,
        kmin: int = 20,
        kmax: int = 40,
        min_sensors: int = 6,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Subarray:

        if isinstance(n_realizations, bool) or not isinstance(n_realizations, int):
            raise TypeError(f"n_realizations 类型应为 int，当前为 {type(n_realizations).__name__}: {n_realizations!r}")
        if n_realizations <= 0:
            raise ValueError(f"n_realizations 应 > 0，当前为: {n_realizations!r}")
        for name, v in [("kmin", kmin), ("kmax", kmax), ("min_sensors", min_sensors)]:
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name} 类型应为 int，当前为 {type(v).__name__}: {v!r}")
        if kmin < 1:
            raise ValueError(f"kmin 应 >= 1，当前为: {kmin!r}")
        if kmax < kmin:
            raise ValueError(f"kmax 应 >= kmin，当前为: kmax={kmax!r}, kmin={kmin!r}")
        if min_sensors < 2:
            raise ValueError(f"min_sensors 应 >= 2，当前为: {min_sensors!r}")

        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, int):
                raise TypeError(
                    f"random_state 类型应为 int 或 None，当前为 {type(random_state).__name__}: {random_state!r}")

        xy = _validate_sensor_xy(sensor_xy, geometry="2d")
        rng = np.random.default_rng(random_state)

        # 计算节点包络矩形范围，用于生成 Voronoi 种子点
        minx, maxx = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
        miny, maxy = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))

        out: Subarray = []

        # 多次随机划分（n_realizations 次），每次可产生多个子阵列
        for _ in range(int(n_realizations)):
            # 每次随机选择一个 k，决定 Voronoi 分区数量
            k = int(rng.integers(int(kmin), int(kmax) + 1))

            # 在包络框内生成 k 个随机种子点 (k,2)
            seeds = rng.random((k, 2))
            seeds[:, 0] = seeds[:, 0] * (maxx - minx) + minx
            seeds[:, 1] = seeds[:, 1] * (maxy - miny) + miny

            # 生成 Voronoi 结构
            vor = Voronoi(seeds)

            # 遍历每个种子点对应的 Voronoi region
            for j in range(k):
                region = vor.regions[vor.point_region[j]]

                # region 为空或无界区域，跳过
                if (not region) or (-1 in region):
                    continue

                polygon = vor.vertices[np.asarray(region, dtype=np.int64)]
                if polygon.shape[0] < 3:
                    continue

                # 判断每个传感器是否在 polygon 内
                inside = _points_in_polygon(xy, polygon)
                idx = np.flatnonzero(inside)

                if idx.size < int(min_sensors):
                    continue

                idx = np.unique(idx)
                if idx.size >= 2:
                    out.append(idx)

        if not out:
            raise ValueError("未生成任何子阵列，请检查 n_realizations/kmin/kmax/min_sensors。")

        return out


class _RandomWindow1DBuilder(SubarrayBuilder):
    """1D（适配线性缆DAS数据）：随机滑窗 + 窗内随机抽取通道构成子阵列。"""

    def subarray(
        self,
        sensor_xy: np.ndarray,
        *,
        n_realizations: int = 1000,
        window_length: float = 50.0,
        kmin: int = 5,
        kmax: int = 10,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Subarray:
        if isinstance(n_realizations, bool) or not isinstance(n_realizations, int):
            raise TypeError(f"n_realizations 类型应为 int，当前为 {type(n_realizations).__name__}: {n_realizations!r}")
        if isinstance(window_length, bool) or not isinstance(window_length, (int, float)):
            raise TypeError(f"window_length 类型有误，当前为 {type(window_length).__name__}: {window_length!r}")
        if isinstance(kmin, bool) or not isinstance(kmin, int):
            raise TypeError(f"kmin 类型应为 int，当前为 {type(kmin).__name__}: {kmin!r}")
        if isinstance(kmax, bool) or not isinstance(kmax, int):
            raise TypeError(f"kmax 类型应为 int，当前为 {type(kmax).__name__}: {kmax!r}")

        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, int):
                raise TypeError(
                    f"random_state 必须是 int 或 None，当前为 {type(random_state).__name__}: {random_state!r}")

        s = _validate_sensor_xy(sensor_xy, geometry="1d")
        rng = np.random.default_rng(random_state)

        n_realizations = int(n_realizations)
        if n_realizations <= 0:
            raise ValueError("n_realizations 必须为正整数。")

        L = float(window_length)
        if L <= 0:
            raise ValueError("window_length 必须为正数。")

        kmin = int(kmin)
        kmax = int(kmax)
        if kmin < 2:
            raise ValueError("kmin 至少为 2。")
        if kmax < kmin:
            raise ValueError("kmax 必须 >= kmin。")

        # 为了窗口筛选快，按坐标排序扫描
        order = np.argsort(s)
        s_sorted = s[order]
        s_min = float(s_sorted[0])
        s_max = float(s_sorted[-1])

        if s_max - s_min < L:
            raise ValueError("window_length 大于沿线总长度，无法生成窗口。")

        out: Subarray = []
        seen = set()

        for _ in range(n_realizations):
            left = float(rng.uniform(s_min, s_max - L))
            right = left + L

            mask = (s_sorted >= left) & (s_sorted <= right)
            idx_sorted = np.flatnonzero(mask)
            n_in = int(idx_sorted.size)
            if n_in < kmin:
                continue

            k = int(rng.integers(kmin, min(kmax, n_in) + 1))
            pick_sorted = rng.choice(idx_sorted, size=k, replace=False)

            idx = np.unique(order[pick_sorted].astype(np.int64))
            if idx.size < 2:
                continue

            key = tuple(idx.tolist())
            if key in seen:
                continue

            seen.add(key)
            out.append(idx)

        if not out:
            raise ValueError("未生成任何子阵列，请检查 n_realizations/window_length/kmin/kmax。")

        return out


# ====================
# 工厂函数
# ====================
_SUBARRAY_MAP = {
    "2d": _Voronoi2DBuilder,
    "1d": _RandomWindow1DBuilder,
}


def get_subarray(geometry: str) -> SubarrayBuilder:
    """根据 geometry 返回子阵列构建器实例。"""
    if not isinstance(geometry, str):
        raise TypeError(f"geometry 类型应为 str，当前为 {type(geometry).__name__}: {geometry!r}")
    if not geometry.strip():
        raise ValueError("geometry 不能为空字符串")

    geometry = geometry.strip().lower()
    if geometry not in SUPPORTED_GEOMETRY:
        raise ValueError(f"geometry={geometry} 不支持，应为 {SUPPORTED_GEOMETRY}")
    return _SUBARRAY_MAP[geometry]()


# =====================
# 辅助函数
# =====================

def _validate_sensor_xy(sensor_xy: np.ndarray, *, geometry: str) -> np.ndarray:
    """
    输入合法性检查：
    - geometry="2d": (n,2) 平面坐标（米）
    - geometry="1d": (n,) 或 (n,1) 沿线坐标（米）
    """
    if geometry not in ("1d", "2d"):
        raise ValueError('geometry 只能为 "1d" 或 "2d"。')

    if geometry == "2d":
        xy = np.asarray(sensor_xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("2d 情况下 sensor_xy 必须为 (n_sensors, 2) 的平面坐标（米）。")
        if xy.shape[0] < 2:
            raise ValueError("2d 情况下 sensor_xy 至少包含 2 个传感器。")
        if not np.all(np.isfinite(xy)):
            raise ValueError("2d 情况下 sensor_xy 不能包含 NaN/Inf。")
        return xy

    # geometry == "1d"
    s = np.asarray(sensor_xy, dtype=np.float64)
    if s.ndim == 1:
        s = s.reshape(-1)
    elif s.ndim == 2 and s.shape[1] == 1:
        s = s[:, 0].reshape(-1)
    else:
        raise ValueError("1d 情况下 sensor_xy 必须为 (n,) 或 (n,1) 的沿线坐标（米）。")

    if s.size < 2:
        raise ValueError("1d 情况下 sensor_xy 至少需要 2 个通道。")
    if not np.all(np.isfinite(s)):
        raise ValueError("1d 情况下 sensor_xy 不能包含 NaN/Inf。")
    if float(np.max(s) - np.min(s)) <= 0.0:
        raise ValueError("1d 情况下 sensor_xy 不能全部相等（需要有沿线展开）。")

    return s


def _points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    """判断 points_xy 中每个点是否在 polygon_xy 内（含边界）。"""
    path = Path(polygon_xy)
    return path.contains_points(points_xy, radius=-1e-10)
