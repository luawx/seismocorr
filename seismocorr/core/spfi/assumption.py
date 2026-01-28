# seismocorr/core/assumption.py

"""
SPFI assumption Building Module

仅负责由子阵列 subarray -> 构建 A 矩阵（csr_matrix），支持两种假设：
- assumption="station_avg"：子阵列相速度为台站平均
- assumption="ray_avg"：子阵列慢度为射线路径慢度平均（1d 情况下自动降级为 station_avg）
"""


import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple
from scipy.sparse import csr_matrix
from seismocorr.config.default import SUPPORTED_ASSUMPTION, SUPPORTED_GEOMETRY

# 优化库导入
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    def njit(func=None, **kwargs):
        """Numba不可用时的回退装饰器"""
        if func is None:
            return lambda f: f
        return func
    
    def prange(n):
        """Numba不可用时的回退prange"""
        return range(n)


Subarray = List[np.ndarray]
MatrixLike = csr_matrix

class DesignMatrixBuilder(ABC):
    """
    设计矩阵 A 构建器策略抽象基类。
    具体策略需继承并实现 assumption 方法
    """

    @abstractmethod
    def assumption(
        self,
        subarray: Sequence[Sequence[int]],
        sensor_xy: np.ndarray,
        geometry: str,
        grid_x: Optional[np.ndarray] = None,
        grid_y: Optional[np.ndarray] = None,
        pair_sampling: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> MatrixLike:
        """
        Args:
            subarray: 子阵列索引列表（每个元素是一组台站索引）
            sensor_xy: 传感器坐标或沿线坐标：
                - geometry="2d": (n,2) 的平面坐标（米）
                - geometry="1d": (n,) 或 (n,1) 的沿线坐标（米）
            geometry: "1d" 或 "2d"
            grid_x, grid_y: 2d 网格中心点 x/y（ray_avg 必需）
            pair_sampling: 射线对抽样：None=所有射线对；整数=抽样数量
            random_state: 抽样随机种子

        Returns:
            csr_matrix: 设计矩阵 A
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> MatrixLike:
        return self.assumption(*args, **kwargs)


class StationAvgBuilder(DesignMatrixBuilder):
    """A0：子阵列相速度为其内部台站平均相速度假设。"""

    def assumption(
        self,
        subarray: Sequence[Sequence[int]],
        sensor_xy: np.ndarray,
        geometry: str,
        grid_x: Optional[np.ndarray] = None,
        grid_y: Optional[np.ndarray] = None,
        pair_sampling: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> MatrixLike:
        if not isinstance(subarray, (list, tuple)):
            raise TypeError(f"subarray 类型应为 list/tuple，当前为 {type(subarray).__name__}")
        if not isinstance(sensor_xy, np.ndarray):
            raise TypeError(f"sensor_xy 类型应为 np.ndarray，当前为 {type(sensor_xy).__name__}")
        if not isinstance(geometry, str):
            raise TypeError(f"geometry 类型应为 str，当前为 {type(geometry).__name__}: {geometry!r}")
        if pair_sampling is not None:
            if isinstance(pair_sampling, bool) or not isinstance(pair_sampling, int):
                raise TypeError(
                    f"pair_sampling 类型应为 int 或 None，当前为 {type(pair_sampling).__name__}: {pair_sampling!r}"
                )
            if pair_sampling < 0:
                raise ValueError(f"pair_sampling 应 >= 0 或 None，当前为: {pair_sampling!r}")

        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, int):
                raise TypeError(
                    f"random_state 类型应为 int 或 None，当前为 {type(random_state).__name__}: {random_state!r}"
                )

        _validate_geometry(geometry)
        n_sensors = _infer_n_sensors(sensor_xy, geometry=geometry)

        subs = _normalize_subarray(subarray, n_sensors=n_sensors)
        n_rows = len(subs)
        if n_rows == 0:
            raise ValueError("subarray 为空或无有效子阵列（每个子阵列至少 2 个传感器）。")

        nnz = int(sum(int(idx.size) for idx in subs))
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=np.float64)

        p = 0
        for r, idx in enumerate(subs):
            m = int(idx.size)
            w = 1.0 / float(m)
            rows[p : p + m] = r
            cols[p : p + m] = idx
            data[p : p + m] = w
            p += m

        A = csr_matrix((data, (rows, cols)), shape=(n_rows, n_sensors))
        if not isinstance(A, csr_matrix):
            raise RuntimeError(f"返回值必须是 csr_matrix，当前为 {type(A).__name__}")
        if A.data.size and (not np.all(np.isfinite(A.data))):
            raise RuntimeError("输出矩阵 A.data 包含 NaN/Inf")

        return A


class RayAvgBuilder(DesignMatrixBuilder):
    """
    A1：子阵列相慢度为其内部台站对射线路径的平均慢度假设。
    - geometry="1d" 自动降级为 station_avg
    """

    def assumption(
        self,
        subarray: Sequence[Sequence[int]],
        sensor_xy: np.ndarray,
        geometry: str,
        grid_x: Optional[np.ndarray] = None,
        grid_y: Optional[np.ndarray] = None,
        pair_sampling: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> MatrixLike:
        if not isinstance(subarray, (list, tuple)):
            raise TypeError(f"subarray 类型应为 list/tuple，当前为 {type(subarray).__name__}")
        if not isinstance(sensor_xy, np.ndarray):
            raise TypeError(f"sensor_xy 类型应为 np.ndarray，当前为 {type(sensor_xy).__name__}")
        if not isinstance(geometry, str):
            raise TypeError(f"geometry 类型应为 str，当前为 {type(geometry).__name__}: {geometry!r}")
        if pair_sampling is not None:
            if isinstance(pair_sampling, bool) or not isinstance(pair_sampling, int):
                raise TypeError(
                    f"pair_sampling 类型应为 int 或 None，当前为 {type(pair_sampling).__name__}: {pair_sampling!r}"
                )
            if pair_sampling < 0:
                raise ValueError(f"pair_sampling 应 >= 0 或 None，当前为: {pair_sampling!r}")

        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, int):
                raise TypeError(
                    f"random_state 类型应为 int 或 None，当前为 {type(random_state).__name__}: {random_state!r}"
                )

        _validate_geometry(geometry)

        if geometry == "1d":
            print('[SPFI] Ray_Avg在1d情况下将自动降级为Sta_Avg')
            return StationAvgBuilder().assumption(
                subarray=subarray,
                sensor_xy=sensor_xy,
                geometry=geometry,
                grid_x=grid_x,
                grid_y=grid_y,
                pair_sampling=pair_sampling,
                random_state=random_state,
            )

        # geometry == "2d"
        n_sensors = _infer_n_sensors(sensor_xy, geometry="2d")
        subs = _normalize_subarray(subarray, n_sensors=n_sensors)
        if len(subs) == 0:
            raise ValueError("subarray 为空或无有效子阵列（每个子阵列至少 2 个传感器）。")

        if grid_x is None or grid_y is None:
            raise ValueError("ray_avg & geometry=2d 必须提供 grid_x/grid_y（网格中心点数组）。")
        if not isinstance(grid_x, np.ndarray):
            raise TypeError(f"grid_x 必须是 np.ndarray，当前为 {type(grid_x).__name__}")
        if not isinstance(grid_y, np.ndarray):
            raise TypeError(f"grid_y 必须是 np.ndarray，当前为 {type(grid_y).__name__}")

        if grid_x.ndim != 1:
            raise ValueError(f"grid_x 必须是一维数组（中心点），当前 shape={grid_x.shape}")
        if grid_y.ndim != 1:
            raise ValueError(f"grid_y 必须是一维数组（中心点），当前 shape={grid_y.shape}")

        if grid_x.size < 2 or grid_y.size < 2:
            raise ValueError("grid_x/grid_y 长度必须 >= 2（中心点数组）")

        if not np.issubdtype(grid_x.dtype, np.number):
            raise TypeError(f"grid_x dtype 必须是数值类型，当前为 {grid_x.dtype}")
        if not np.issubdtype(grid_y.dtype, np.number):
            raise TypeError(f"grid_y dtype 必须是数值类型，当前为 {grid_y.dtype}")

        if not np.all(np.isfinite(grid_x)):
            raise ValueError("grid_x 包含 NaN/Inf")
        if not np.all(np.isfinite(grid_y)):
            raise ValueError("grid_y 包含 NaN/Inf")

        x_edges = _ensure_edges_1d(np.asarray(grid_x, dtype=np.float64))
        y_edges = _ensure_edges_1d(np.asarray(grid_y, dtype=np.float64))

        rng = np.random.default_rng(random_state)

        nx = int(x_edges.size - 1)
        ny = int(y_edges.size - 1)
        n_cells = int(nx * ny)

        rows, cols, data = _rows_ray_2d(
            subarray=subs,
            sensor_xy=np.asarray(sensor_xy, dtype=np.float64),
            x_edges=x_edges,
            y_edges=y_edges,
            nx=nx,
            ny=ny,
            pair_sampling=pair_sampling,
            rng=rng,
        )

        A = csr_matrix((data, (rows, cols)), shape=(len(subs), n_cells))
        if not isinstance(A, csr_matrix):
            raise RuntimeError(f"返回值必须是 csr_matrix，当前为 {type(A).__name__}")
        if A.data.size and (not np.all(np.isfinite(A.data))):
            raise RuntimeError("输出矩阵 A.data 包含 NaN/Inf")

        return A


# ====================
# 工厂函数
# ====================
_ASSUMPTION_MAP = {
    "station_avg": StationAvgBuilder,
    "ray_avg": RayAvgBuilder,
}


def get_assumption(assumption: str) -> DesignMatrixBuilder:
    """根据 assumption 名称返回矩阵 A 构建器实例。"""

    if not isinstance(assumption, str):
        raise TypeError(f"assumption 类型应为 str，当前为 {type(assumption).__name__}: {assumption!r}")
    if not assumption.strip():
        raise ValueError("assumption 不能为空字符串")

    if assumption not in SUPPORTED_ASSUMPTION:
        raise ValueError(f"assumption={assumption} 不支持，应为 {SUPPORTED_ASSUMPTION}")
    return _ASSUMPTION_MAP[assumption]()


# =====================
# 辅助函数
# =====================
def _validate_geometry(geometry: str) -> None:
    if geometry not in SUPPORTED_GEOMETRY:
        raise ValueError(f"geometry={geometry} 不支持，应为 {SUPPORTED_GEOMETRY}")


def _infer_n_sensors(sensor_xy: np.ndarray, *, geometry: str) -> int:
    """
    检查 n_sensors 格式。
    - 2d: n = sensor_xy.shape[0]
    - 1d: n = sensor_xy.size（支持 (n,) / (n,1)）
    """
    s = np.asarray(sensor_xy)

    if not np.issubdtype(s.dtype, np.number):
        raise TypeError(f"sensor_xy dtype 应为数值类型，当前为 {s.dtype}")
    if s.size > 0 and (not np.all(np.isfinite(s))):
        raise ValueError("sensor_xy 包含 NaN/Inf")

    if geometry == "2d":
        if s.ndim != 2 or s.shape[1] != 2:
            raise ValueError(f"geometry='2d' 时 sensor_xy 形状应为 (n,2)，当前为 {s.shape}")
        return int(s.shape[0])

    if s.ndim == 1:
        return int(s.size)
    if s.ndim == 2 and s.shape[1] == 1:
        return int(s.shape[0])

    raise ValueError(f"geometry='1d' 时 sensor_xy 形状应为 (n,) 或 (n,1)，当前为 {s.shape}")


def _normalize_subarray(subarray: Sequence[Sequence[int]], n_sensors: int) -> Subarray:
    """
    规范化子阵列索引：
    - 转 int64
    - size<2 的子阵列舍弃
    - 越界报错
    - 子阵列内部去重
    """
    out: Subarray = []
    for s in subarray:
        if s is None or isinstance(s, (int, np.integer, str, bytes)):
            raise TypeError(f"subarray 的每个元素应为索引序列，当前为: {s!r}")
        idx = np.asarray(list(s), dtype=np.int64).reshape(-1)
        if idx.size < 2:
            continue
        if np.any(idx < 0) or np.any(idx >= n_sensors):
            raise ValueError("subarray 中存在越界索引。")
        out.append(np.unique(idx))
    return out


def _ensure_edges_1d(centers: np.ndarray) -> np.ndarray:
    """
    输入：严格递增的中心点数组 c (n,)
    输出：边界 edges (n+1,)
    """
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    if c.size < 2 or not np.all(np.diff(c) > 0):
        raise ValueError("grid_x/grid_y 必须严格递增且长度>=2（中心点数组）。")

    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - (edges[1] - c[0])
    edges[-1] = c[-1] + (c[-1] - edges[-2])
    return edges


def _pairs_from_indices(
    idx: np.ndarray,
    pair_sampling: Optional[int],
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """从一个子阵列索引 idx 生成台站对 (i,j)，可选随机抽样。"""
    n = int(idx.size)
    if n < 2:
        return []

    pairs = [(int(idx[a]), int(idx[b])) for a in range(n) for b in range(a + 1, n)]
    if pair_sampling is None or pair_sampling <= 0 or pair_sampling >= len(pairs):
        return pairs

    sel = rng.choice(len(pairs), size=int(pair_sampling), replace=False)
    return [pairs[int(i)] for i in sel]


def _rows_ray_2d(
    subarray: Subarray,
    sensor_xy: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    nx: int,
    ny: int,
    pair_sampling: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 ray_avg 的稀疏三元组 (rows, cols, data)。"""
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for r, idx in enumerate(subarray):
        pairs = _pairs_from_indices(idx, pair_sampling, rng)
        if not pairs:
            continue

        w = np.zeros(nx * ny, dtype=np.float64)

        for i, j in pairs:
            x1, y1 = float(sensor_xy[i, 0]), float(sensor_xy[i, 1])
            x2, y2 = float(sensor_xy[j, 0]), float(sensor_xy[j, 1])
            if x1 == x2 and y1 == y2:
                continue

            _, seg_len, cell_ij = _ray_grid_intersections_2d(x_edges, y_edges, x1, y1, x2, y2)
            if seg_len.size == 0:
                continue

            # 累计权重
            for k in range(seg_len.size):
                ix = cell_ij[k, 0]
                iy = cell_ij[k, 1]
                cell_idx = iy * nx + ix
                w[cell_idx] += seg_len[k]

        s = float(np.sum(w))
        if s <= 0:
            continue
        w /= s

        nz = np.flatnonzero(w > 0)
        rows.extend([r] * int(nz.size))
        cols.extend(nz.tolist())
        data.extend(w[nz].tolist())

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
    )


@njit(cache=True, fastmath=True)
def _ray_grid_intersections_2d(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    线段 (x1,y1)->(x2,y2) 与网格边界的交点分段计算。

    Returns:
        pts: (n_pts,2) 分段端点（含起终点）
        seg_len: (n_pts-1,) 每段长度
        cell_ij: (n_pts-1,2) 每段中点所在 cell 的 (ix,iy)
    """
    dx = x2 - x1
    dy = y2 - y1

    # 使用数组而不是列表来存储ts，提高Numba性能
    ts_size = 2 + (x_edges.size if abs(dx) > eps else 0) + (y_edges.size if abs(dy) > eps else 0)
    ts = np.zeros(ts_size, dtype=np.float64)
    ts[0] = 0.0
    ts[1] = 1.0
    idx = 2
    
    if abs(dx) > eps:
        for xe in x_edges:
            ts[idx] = (xe - x1) / dx
            idx += 1
    if abs(dy) > eps:
        for ye in y_edges:
            ts[idx] = (ye - y1) / dy
            idx += 1
    
    # 裁剪到有效范围
    t = ts[(ts >= -eps) & (ts <= 1.0 + eps)]
    t = np.clip(t, 0.0, 1.0)
    
    # 去重和排序
    # 使用Numba支持的方式进行round操作
    t_rounded = np.zeros(t.size, dtype=np.float64)
    for i in range(t.size):
        t_rounded[i] = eps * np.round(t[i] / eps)
    
    t = np.unique(t_rounded)
    t = np.clip(t, 0.0, 1.0)
    t = np.sort(t)

    # 计算端点
    n_pts = t.size
    pts = np.empty((n_pts, 2), dtype=np.float64)
    for i in range(n_pts):
        pts[i, 0] = x1 + t[i] * dx
        pts[i, 1] = y1 + t[i] * dy
    
    if n_pts < 2:
        # Numba在nopython模式下无法处理空数组，返回具有最小有效尺寸的数组
        return pts, np.array([0.0], dtype=np.float64)[:0], np.array([[0, 0]], dtype=np.int64)[:0]

    # 计算分段长度
    n_segs = n_pts - 1
    seg_len = np.empty(n_segs, dtype=np.float64)
    mid = np.empty((n_segs, 2), dtype=np.float64)
    
    for i in range(n_segs):
        dx_seg = pts[i+1, 0] - pts[i, 0]
        dy_seg = pts[i+1, 1] - pts[i, 1]
        seg_len[i] = np.sqrt(dx_seg**2 + dy_seg**2)
        
        # 计算中点
        mid[i, 0] = 0.5 * (pts[i, 0] + pts[i+1, 0])
        mid[i, 1] = 0.5 * (pts[i, 1] + pts[i+1, 1])
    
    # 计算中点所在的cell
    ix = np.empty(n_segs, dtype=np.int64)
    iy = np.empty(n_segs, dtype=np.int64)
    
    for i in range(n_segs):
        # 使用二分查找确定网格索引
        # 对于Numba，手动实现二分查找比np.searchsorted更快
        # 查找x索引
        x = mid[i, 0]
        left, right = 0, x_edges.size - 1
        while left < right:
            mid_x = (left + right) // 2
            if x_edges[mid_x] < x:
                left = mid_x + 1
            else:
                right = mid_x
        ix[i] = left - 1
        
        # 查找y索引
        y = mid[i, 1]
        left, right = 0, y_edges.size - 1
        while left < right:
            mid_y = (left + right) // 2
            if y_edges[mid_y] < y:
                left = mid_y + 1
            else:
                right = mid_y
        iy[i] = left - 1
    
    # 裁剪到有效范围
    ix = np.clip(ix, 0, x_edges.size - 2)
    iy = np.clip(iy, 0, y_edges.size - 2)
    
    cell_ij = np.column_stack((ix, iy))
    return pts, seg_len, cell_ij
