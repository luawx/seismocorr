"""
Dispersion Spectrum Picker GUI.

本脚本提供一个用于“频散谱（Dispersion Spectrum）”的交互式拾取与编辑工具，
基于 Tkinter + Matplotlib 实现可视化浏览、圈选（Pick）以及手工点选（Click）编辑，
并将拾取结果保存为 YAML 文件。

主要功能：
  1) 浏览：在多张频散谱图（ds[image, c, f]）之间切换、跳转。
  2) Pick 模式：用多边形圈选 ROI（Region of Interest），为 Search 提供区域约束。
  3) Search：在 ROI 内按频率采样 fp 提取最大脊线（ridge）作为初始曲线点。
  4) Click 模式：用鼠标对曲线点进行增/改/删（删除仅在点击点附近生效）。
  5) 采样参数：用户可在 GUI 中修改 start/step 控制 fp 的采样密度。
  6) 保存：将每张图、每个阶次（order）的曲线点写入 YAML（config.yml）。

数据约定：
  - HDF5 输入文件包含数据集：
      ds: shape = [num_images, nc, nf]
      f : shape = [nf]  (频率轴)
      c : shape = [nc]  (相速度/相位速度轴)
  - 输出 YAML 结构：
      dispersionCurves[image_index][order] = {"f": [...], "c": [...]}

快捷键：
  - Ctrl+Shift+P：切换 Pick 模式
  - Ctrl+Shift+M：切换 Click 模式
  - Ctrl+Shift+S：执行 Search（基于当前多边形 ROI）
  - Ctrl+Shift+C：清空当前 Pick 多边形顶点
  - Ctrl+Shift+D：清空当前阶次曲线点
  - Ctrl+Shift+W：保存 YAML
"""

import os
import argparse
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import h5py
import yaml

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon


def inpolygon(xq, yq, xv, yv):
    vertices = np.vstack((xv, yv)).T
    path = Path(vertices)
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    _in = path.contains_points(test_points)
    _in_on = path.contains_points(test_points, radius=-1e-10)
    _on = _in ^ _in_on
    return _in_on, _on


def get_yaml_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class DispersionSpectrumGUI:
    """频散谱拾取与编辑 GUI（Tkinter + Matplotlib）。

    该类负责构建界面、绘制频散谱、处理鼠标/键盘交互，并维护曲线拾取结果。

    核心概念：
      - 图像索引（indx）：当前显示的 ds 图编号，范围 [0, NumP-1]。
      - 阶次（order）：同一张图可能存在多条模式曲线，用 0~9 区分。
      - 三种工作模式：
          1) idle：空闲模式，点击不产生修改（避免误操作）。
          2) pick：圈选模式，左键添加多边形顶点，右键删除最近顶点。
          3) fine（click）：曲线编辑模式，左键新增/更新点，右键仅删除“点击附近”的点。

    Search（自动提取）逻辑：
      - 用户先在 pick 模式下绘制 ROI 多边形。
      - Search 会根据 fp（由 start/step 采样得到）在 ROI 覆盖的频率范围内，
        对每个采样频率列取 ROI 内幅值最大的相速度位置作为脊线点（f, c）。

    采样参数（GUI 可改）：
      - fp_start：从 f 的哪个下标开始采样（跳过前若干频率点）。
      - fp_step ：采样步长（每隔多少个频率点采样一次）。
      - fp 的密度决定 Search 输出曲线点的稠密程度；step 越小越密。

    Click（手工编辑）逻辑：
      - 左键：
          - 若点击位置距离已有点足够近（按屏幕像素阈值判断），则更新该点；
          - 否则新增一个点（使用鼠标点击的 x/y）。
      - 右键：
          - 仅当点击位置附近存在点时才删除；否则不执行删除，避免误删。
      - 曲线点会按频率 f 升序排序，以保证绘制与数据结构一致。

    Attributes:
      f (np.ndarray): 频率轴，shape=[nf]。
      c (np.ndarray): 相速度轴，shape=[nc]。
      data (np.ndarray): 频散谱数据，shape=[NumP, nc, nf]。
      dispersionCurves (dict): 结果字典，按 image_index 与 order 存储曲线点。
      fp_start (int): fp 采样起始下标。
      fp_step (int): fp 采样步长。
      fp (np.ndarray): 采样频率序列，用于 Search 的横坐标。
      index (list[int]): fp 对应的 f 下标列表（用于定位列）。

    Notes:
      - Pick 多边形点的“密集程度”由用户点击次数决定，不由 start/step 控制。
      - start/step 仅影响 Search 自动提取的曲线点数与分辨率。
    """

    MODE_IDLE = "idle"
    MODE_POLY = "poly"
    MODE_FINE = "fine"

    def __init__(self, f, c, data, outfile, master):
        self.f = np.asarray(f)
        self.c = np.asarray(c)
        self.data = np.asarray(data)
        self.nf = len(self.f)
        self.nc = len(self.c)
        self.NumP = self.data.shape[0]

        self.outfile = outfile
        self.dispersionCurves = get_yaml_data(outfile) if os.path.exists(outfile) else {}

        self.root = master
        self.root.title("Dispersion Curve Picker")
        self.root.geometry("1980x1100")

        self.indx = 0
        self.order = 0
        self.mode = self.MODE_IDLE  # 默认空闲

        # 圈选点
        self.x = []
        self.y = []
        self.point_artists = []
        self.polyline_artist = None
        self.poly_patch = None

        # ========== Search 频率采样参数（用户可改） ==========
        self.fp_start = 3      # 起始下标（原来固定 3）
        self.fp_step = 5       # 步长（原来固定 5）
        self._rebuild_fp()     # 构建 self.index / self.fp

        self.F, self.C = np.meshgrid(self.f, self.c)

        # ========== ttk & 字体 ==========
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        default_font = ("Microsoft YaHei UI", 10)  # mac 可改 ("PingFang SC", 10)
        style.configure(".", font=default_font)
        style.configure("TLabelframe.Label", font=(default_font[0], 10, "bold"))
        style.configure("Tool.TButton", padding=(10, 6))

        # ========== 主布局 ==========
        self.main = ttk.Frame(self.root, padding=10)
        self.main.pack(fill=tk.BOTH, expand=True)

        # 顶部：一排工具栏
        self.topbar = ttk.Frame(self.main)
        self.topbar.pack(side=tk.TOP, fill=tk.X)

        # 中间：图像区域
        self.center = ttk.Frame(self.main)
        self.center.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 8))

        # 底部：左状态栏 + 右下工具栏
        self.bottombar = ttk.Frame(self.main)
        self.bottombar.pack(side=tk.BOTTOM, fill=tk.X)

        self.statusbar = ttk.Frame(self.bottombar)
        self.statusbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.toolbarbar = ttk.Frame(self.bottombar)
        self.toolbarbar.pack(side=tk.RIGHT)

        # ========== Figure ==========
        self.fig, self.ax = plt.subplots(figsize=(11.5, 8.0), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ========== 顶部控件（一排） ==========
        nav_box = ttk.LabelFrame(self.topbar, text="Browse", padding=(10, 6))
        nav_box.pack(side=tk.LEFT, padx=6, pady=2)

        ttk.Button(nav_box, text="Last", style="Tool.TButton", command=self.lastPoint, width=7).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav_box, text="Next", style="Tool.TButton", command=self.nextPoint, width=7).pack(side=tk.LEFT, padx=4)

        jump_box = ttk.LabelFrame(self.topbar, text="Jump", padding=(10, 6))
        jump_box.pack(side=tk.LEFT, padx=6, pady=2)

        ttk.Label(jump_box, text="Figure").pack(side=tk.LEFT, padx=(0, 6))
        self.entryJump = ttk.Entry(jump_box, width=10)
        self.entryJump.pack(side=tk.LEFT, padx=4)
        ttk.Button(jump_box, text="Jump", style="Tool.TButton", command=self.jump, width=6).pack(side=tk.LEFT, padx=4)

        info_box = ttk.LabelFrame(self.topbar, text="Info", padding=(10, 6))
        info_box.pack(side=tk.LEFT, padx=6, pady=2)

        self.info_var = tk.StringVar(value=f"Image: {self.indx}/{self.NumP-1}")
        ttk.Label(info_box, textvariable=self.info_var, width=18).pack(side=tk.LEFT, padx=4)

        order_box = ttk.LabelFrame(self.topbar, text="Order", padding=(10, 6))
        order_box.pack(side=tk.LEFT, padx=6, pady=2)

        self.order_var = tk.StringVar(value=f"Present: {self.order}")
        ttk.Label(order_box, textvariable=self.order_var, width=12).pack(side=tk.LEFT, padx=(0, 6))

        self.order_select_var = tk.StringVar(value=str(self.order))
        self.cmb_order = ttk.Combobox(
            order_box,
            textvariable=self.order_select_var,
            values=[str(i) for i in range(10)],
            width=6,
            state="readonly"
        )
        self.cmb_order.pack(side=tk.LEFT, padx=4)
        self.cmb_order.bind("<<ComboboxSelected>>", self.on_order_selected)

        # ========== 新增：Step 控件（用户可改 Search 点密度） ==========
        step_box = ttk.LabelFrame(self.topbar, text="Sampling", padding=(10, 6))
        step_box.pack(side=tk.LEFT, padx=6, pady=2)

        ttk.Label(step_box, text="start").pack(side=tk.LEFT, padx=(0, 4))
        self.fp_start_var = tk.StringVar(value=str(self.fp_start))
        self.entry_fp_start = ttk.Entry(step_box, textvariable=self.fp_start_var, width=6)
        self.entry_fp_start.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(step_box, text="step").pack(side=tk.LEFT, padx=(0, 4))
        self.fp_step_var = tk.StringVar(value=str(self.fp_step))
        self.entry_fp_step = ttk.Entry(step_box, textvariable=self.fp_step_var, width=6)
        self.entry_fp_step.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(step_box, text="Apply", style="Tool.TButton", command=self.apply_sampling, width=7).pack(
            side=tk.LEFT, padx=4
        )

        self.fp_info_var = tk.StringVar(value=f"fp n={len(self.fp)}")
        ttk.Label(step_box, textvariable=self.fp_info_var, width=10).pack(side=tk.LEFT, padx=4)

        mode_box = ttk.LabelFrame(self.topbar, text="Mode", padding=(10, 6))
        mode_box.pack(side=tk.LEFT, padx=6, pady=2)

        self.poly_mode_var = tk.BooleanVar(value=False)
        self.fine_tune_var = tk.BooleanVar(value=False)

        self.chk_poly = ttk.Checkbutton(
            mode_box, text="Pick", variable=self.poly_mode_var, command=self._on_toggle_poly_mode
        )
        self.chk_poly.pack(side=tk.LEFT, padx=6)

        self.chk_fine = ttk.Checkbutton(
            mode_box, text="Click", variable=self.fine_tune_var, command=self._on_toggle_fine_tune
        )
        self.chk_fine.pack(side=tk.LEFT, padx=6)

        action_box = ttk.LabelFrame(self.topbar, text="Operate", padding=(10, 6))
        action_box.pack(side=tk.LEFT, padx=6, pady=2)

        self.btn_clear = ttk.Button(action_box, text="Redraw", style="Tool.TButton", command=self.clear_selection, width=8)
        self.btn_clear.pack(side=tk.LEFT, padx=4)

        self.btn_clear_order = ttk.Button(action_box, text="Clear", style="Tool.TButton", command=self.clear_current_curve, width=10)
        self.btn_clear_order.pack(side=tk.LEFT, padx=4)

        ttk.Button(action_box, text="Search", style="Tool.TButton", command=self.search, width=7).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_box, text="Save", style="Tool.TButton", command=self.writein, width=6).pack(side=tk.LEFT, padx=4)

        # ========== 底部：状态栏 ==========
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.statusbar, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=2
        )

        # ========== 底部：工具栏（右下） ==========
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarbar)
        self.toolbar.update()

        # ========== 事件 ==========
        self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.cid_key = self.canvas.mpl_connect("key_press_event", self.on_key_press)

        self._update_mode_ui()
        self.drawDS()

    # -------------------- Sampling 辅助 --------------------
    def _rebuild_fp(self):
        start = int(max(0, min(self.fp_start, self.nf - 1)))
        step = int(max(1, self.fp_step))
        self.index = list(range(start, self.nf, step))
        self.fp = self.f[self.index]

    def apply_sampling(self):
        raw_start = (self.fp_start_var.get() or "").strip()
        raw_step = (self.fp_step_var.get() or "").strip()
        try:
            start = int(raw_start)
            step = int(raw_step)
        except ValueError:
            messagebox.showwarning("Input Error", "start/step must be integers.")
            return
        if step <= 0:
            messagebox.showwarning("Input Error", "step must be > 0.")
            return
        if start < 0 or start >= self.nf:
            messagebox.showwarning("Out of Range", f"start must be in [0, {self.nf-1}].")
            return

        self.fp_start = start
        self.fp_step = step
        self._rebuild_fp()
        self.fp_info_var.set(f"fp n={len(self.fp)}")
        self.status_var.set(f"Sampling updated: start={self.fp_start}, step={self.fp_step}, n={len(self.fp)}")

    # -------------------- 模式控制 --------------------
    def _set_mode(self, mode):
        self.mode = mode
        self._update_mode_ui()

    def _update_mode_ui(self):
        if self.mode == self.MODE_POLY:
            self.poly_mode_var.set(True)
            self.fine_tune_var.set(False)
            self.status_var.set("Mode: Pick (Polygon). Left add / Right delete. Ctrl+Shift+P toggle.")
        elif self.mode == self.MODE_FINE:
            self.poly_mode_var.set(False)
            self.fine_tune_var.set(True)
            self.status_var.set("Mode: Click (Edit). Left add/update / Right delete-nearby. Ctrl+Shift+M toggle.")
        else:
            self.poly_mode_var.set(False)
            self.fine_tune_var.set(False)
            self.status_var.set("Mode: Idle. Click does nothing. Ctrl+Shift+P Pick / Ctrl+Shift+M Click.")

    def _on_toggle_poly_mode(self):
        if self.poly_mode_var.get():
            self._set_mode(self.MODE_POLY)
        else:
            self._set_mode(self.MODE_IDLE)
        self.drawDS()

    def _on_toggle_fine_tune(self):
        if self.fine_tune_var.get():
            self._set_mode(self.MODE_FINE)
        else:
            self._set_mode(self.MODE_IDLE)
        self.drawDS()

    # -------------------- 阶次联动 --------------------
    def _sync_order_ui(self, redraw=False):
        self.order_var.set(f"Present: {self.order}")
        try:
            self.order_select_var.set(str(self.order))
        except Exception:
            pass
        self.info_var.set(f"Image: {self.indx}/{self.NumP-1}")

        if redraw:
            self.drawDS()
        else:
            self.ax.set_title(f"Image {self.indx}/{self.NumP - 1} | order: {self.order}")
            self.canvas.draw()

    def on_order_selected(self, event=None):
        raw = (self.order_select_var.get() or "").strip()
        if not raw.isdigit():
            return
        self.order = int(raw)
        self.status_var.set(f"Order switched to {self.order} (combobox).")
        self._sync_order_ui(redraw=True)

    # -------------------- 换图 --------------------
    def nextPoint(self):
        self.indx = (self.indx + 1) % self.NumP
        self.clear_selection(redraw=False)
        self.drawDS()

    def lastPoint(self):
        self.indx = (self.indx - 1) % self.NumP
        self.clear_selection(redraw=False)
        self.drawDS()

    def jump(self):
        raw = self.entryJump.get().strip()
        try:
            target = int(raw)
        except ValueError:
            messagebox.showwarning("Input Error", "Jump needs an integer index.")
            return
        if not (0 <= target < self.NumP):
            messagebox.showwarning("Out of Range", f"Valid range: 0 ~ {self.NumP - 1}")
            return
        self.indx = target
        self.clear_selection(redraw=False)
        self.drawDS()

    # -------------------- 绘图 --------------------
    def drawDS(self):
        self.ax.clear()
        self.ax.pcolormesh(self.f, self.c, self.data[self.indx, :, :], vmin=0.0, vmax=1.0)
        self.ax.set_xlabel("Frequency")
        self.ax.set_ylabel("Phase Velocity")
        self.ax.set_title(f"Image {self.indx}/{self.NumP - 1} | order: {self.order}")

        current_ms = 9
        other_ms = 4

        if self.indx in self.dispersionCurves:
            for ord_key in self.dispersionCurves[self.indx]:
                curve = self.dispersionCurves[self.indx][ord_key]
                if "f" in curve and "c" in curve:
                    try:
                        is_current = (int(ord_key) == int(self.order))
                    except Exception:
                        is_current = (str(ord_key) == str(self.order))
                    if is_current:
                        self.ax.plot(curve["f"], curve["c"], "r.", markersize=current_ms, alpha=1.0)
                    else:
                        self.ax.plot(curve["f"], curve["c"], "r.", markersize=other_ms, alpha=0.35)

        self._redraw_selection_only()
        self.canvas.draw()
        self._sync_order_ui(redraw=False)
        self.root.title(f"Dispersion Curve Picker - {self.indx}/{self.NumP - 1}")

    def _redraw_selection_only(self):
        for a in self.point_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.point_artists = []

        if self.polyline_artist is not None:
            try:
                self.polyline_artist.remove()
            except Exception:
                pass
            self.polyline_artist = None

        if self.poly_patch is not None:
            try:
                self.poly_patch.remove()
            except Exception:
                pass
            self.poly_patch = None

        for xi, yi in zip(self.x, self.y):
            artist = self.ax.plot(xi, yi, "k.", markersize=7)[0]
            self.point_artists.append(artist)

        if len(self.x) >= 3:
            verts = np.column_stack([self.x, self.y])
            self.poly_patch = Polygon(
                verts, closed=True, fill=True, alpha=0.18, edgecolor="k", linewidth=1.2
            )
            self.ax.add_patch(self.poly_patch)
        elif len(self.x) >= 2:
            self.polyline_artist = self.ax.plot(self.x, self.y, "k-", linewidth=1.2)[0]

    # -------------------- 鼠标交互 --------------------
    def on_mouse_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(np.round(event.xdata, 3))
        y = float(np.round(event.ydata, 3))

        if self.mode == self.MODE_IDLE:
            return

        if self.mode == self.MODE_POLY:
            if event.button == 1:
                self.x.append(x)
                self.y.append(y)
                self.status_var.set(f"Pick: add vertex ({x}, {y}) | n={len(self.x)}")
                self._redraw_selection_only()
                self.canvas.draw()
            elif event.button == 3:
                if len(self.x) == 0:
                    self.status_var.set("Pick: no vertices to delete.")
                    return
                idx = self._nearest_point_index(self.x, self.y, x, y)
                rx, ry = self.x[idx], self.y[idx]
                del self.x[idx]
                del self.y[idx]
                self.status_var.set(f"Pick: delete vertex ({rx}, {ry}) | n={len(self.x)}")
                self._redraw_selection_only()
                self.canvas.draw()
            return

        if self.mode == self.MODE_FINE:
            self._edit_curve_by_click(event.button, x, y, event=event)
            return

    def _nearest_point_index(self, xs_list, ys_list, x, y):
        xs = np.array(xs_list, dtype=float)
        ys = np.array(ys_list, dtype=float)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        return int(np.argmin(d2))

    def _nearest_point_index_display(self, xs_list, ys_list, event, max_dist_px=12):
        if len(xs_list) == 0:
            return None
        pts = np.column_stack([np.asarray(xs_list, float), np.asarray(ys_list, float)])
        disp = self.ax.transData.transform(pts)

        mx, my = float(event.x), float(event.y)
        d2 = (disp[:, 0] - mx) ** 2 + (disp[:, 1] - my) ** 2
        idx = int(np.argmin(d2))

        if np.sqrt(d2[idx]) <= max_dist_px:
            return idx
        return None

    # -------------------- 键盘交互 --------------------
    def on_key_press(self, event):
        if not event.key:
            return

        k = event.key.lower()

        if k.isdigit() and (0 <= int(k) <= 9):
            self.order = int(k)
            self.status_var.set(f"Order switched to {self.order} (keyboard).")
            self._sync_order_ui(redraw=True)
            return

        if k == "ctrl+shift+p":
            self._set_mode(self.MODE_IDLE if self.mode == self.MODE_POLY else self.MODE_POLY)
            self.drawDS()
            return

        if k == "ctrl+shift+m":
            self._set_mode(self.MODE_IDLE if self.mode == self.MODE_FINE else self.MODE_FINE)
            self.drawDS()
            return

        if k == "ctrl+shift+s":
            self.search()
            return

        if k == "ctrl+shift+c":
            self.clear_selection()
            return

        if k == "ctrl+shift+d":
            self.clear_current_curve()
            return

        if k == "ctrl+shift+w":
            self.writein()
            return

        self.status_var.set(
            "Hotkeys: 0-9 order | Ctrl+Shift+P Pick | Ctrl+Shift+M Click | Ctrl+Shift+S Search | "
            "Ctrl+Shift+C Clear Pick | Ctrl+Shift+D Clear Order | Ctrl+Shift+W Save"
        )

    # -------------------- 清空圈选 --------------------
    def clear_selection(self, redraw=True):
        self.x = []
        self.y = []
        self.status_var.set("Pick cleared.")
        if redraw:
            self.drawDS()

    # -------------------- 清空当前阶次曲线 --------------------
    def clear_current_curve(self, redraw=True):
        curve = self._get_current_curve(create=False)
        if curve is None:
            self.status_var.set("No curve for current image/order.")
            return
        curve["f"] = []
        curve["c"] = []
        self.status_var.set(f"Cleared curve: image={self.indx}, order={self.order}.")
        if redraw:
            self.drawDS()

    # -------------------- 保存结构 --------------------
    def addPoint(self):
        if self.indx not in self.dispersionCurves:
            self.dispersionCurves[self.indx] = {}

    def addOrder(self):
        self.addPoint()
        if self.order not in self.dispersionCurves[self.indx]:
            self.dispersionCurves[self.indx][self.order] = {"f": [], "c": []}

    def addDC(self, x, y):
        self.addOrder()
        self.dispersionCurves[self.indx][self.order]["f"] = x
        self.dispersionCurves[self.indx][self.order]["c"] = y

    def _get_current_curve(self, create=False):
        if self.indx not in self.dispersionCurves:
            if not create:
                return None
            self.dispersionCurves[self.indx] = {}
        if self.order not in self.dispersionCurves[self.indx]:
            if not create:
                return None
            self.dispersionCurves[self.indx][self.order] = {"f": [], "c": []}
        return self.dispersionCurves[self.indx][self.order]

    # -------------------- Click 模式：按鼠标增删改 --------------------
    def _edit_curve_by_click(self, button, x, y, event=None):
        curve = self._get_current_curve(create=True)
        fx = list(map(float, curve.get("f", [])))
        cy = list(map(float, curve.get("c", [])))

        NEAR_PX = 12  # 可调：点附近判定（像素）

        if button == 1:
            idx = None
            if event is not None:
                idx = self._nearest_point_index_display(fx, cy, event, max_dist_px=NEAR_PX)

            if idx is None:
                fx.append(float(x))
                cy.append(float(y))
                self.status_var.set(f"Click: add ({float(x):.3f}, {float(y):.3f}) | n={len(fx)}")
            else:
                fx[idx] = float(x)
                cy[idx] = float(y)
                self.status_var.set(f"Click: update ({float(x):.3f}, {float(y):.3f}) | n={len(fx)}")

            order = np.argsort(np.array(fx))
            fx = [fx[i] for i in order]
            cy = [cy[i] for i in order]

            curve["f"], curve["c"] = fx, cy
            self.drawDS()
            return

        if button == 3:
            if len(fx) == 0:
                self.status_var.set("Click: no points to delete.")
                return

            idx = None
            if event is not None:
                idx = self._nearest_point_index_display(fx, cy, event, max_dist_px=NEAR_PX)
            else:
                idx = self._nearest_point_index(fx, cy, x, y)

            if idx is None:
                self.status_var.set("Click: no nearby point to delete.")
                return

            rx, ry = fx[idx], cy[idx]
            del fx[idx]
            del cy[idx]
            curve["f"], curve["c"] = fx, cy
            self.status_var.set(f"Click: delete ({rx:.3f}, {ry:.3f}) | n={len(fx)}")
            self.drawDS()
            return

    # -------------------- Search：多边形内取最大脊线 --------------------
    def search(self):
        if len(self.x) < 3:
            messagebox.showwarning("Not enough points", "Need >= 3 vertices to form a polygon. Please use Pick mode.")
            return

        inon, _ = inpolygon(self.F, self.C, np.array(self.x), np.array(self.y))
        inon = np.array(inon).reshape(self.nc, self.nf)
        tmp = inon.astype(float) * np.abs(self.data[self.indx, :, :])

        self.ax.clear()
        self.ax.pcolormesh(self.f, self.c, tmp, vmin=0.0, vmax=1.0)
        self.ax.set_xlabel("Frequency")
        self.ax.set_ylabel("Phase Velocity")
        self.ax.set_title(f"Image {self.indx}/{self.NumP - 1} | order: {self.order} | masked")

        xmin, xmax = min(self.x), max(self.x)
        i1 = int(np.searchsorted(self.fp, xmin, side="left"))
        i2 = int(np.searchsorted(self.fp, xmax, side="right"))
        i1 = max(0, min(i1, len(self.fp)))
        i2 = max(0, min(i2, len(self.fp)))

        if i2 - i1 < 2:
            messagebox.showwarning("Too narrow", "Polygon is too narrow in frequency. Please draw a wider region.")
            self.drawDS()
            return

        x_out = list(map(float, self.fp[i1:i2]))
        y_out = []
        for k in range(i1, i2):
            col = self.index[k]
            y_out.append(float(self.c[int(np.argmax(tmp[:, col]))]))

        self.addDC(x_out, y_out)

        self.ax.plot(x_out, y_out, "k.", markersize=7)
        verts = np.column_stack([self.x, self.y])
        patch = Polygon(verts, closed=True, fill=True, alpha=0.18, edgecolor="r", linewidth=1.4)
        self.ax.add_patch(patch)

        self.canvas.draw()
        self.status_var.set(f"Search done: saved image={self.indx}, order={self.order}, n={len(x_out)}.")

    # -------------------- 写出 yml --------------------
    def writein(self):
        try:
            with open(self.outfile, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.dispersionCurves, f, allow_unicode=True, sort_keys=False)
            self.status_var.set(f"Saved to: {self.outfile}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


def main():
    parser = argparse.ArgumentParser(description="Select dispersion spectrums (GUI with modes)")
    parser.add_argument("--infile", default="examples/test_disper_picker_data/ds.h5", help="input file[ds.h5]")
    parser.add_argument("--outfile", default="examples/test_disper_picker_data/config.yml", help="config file[config.yml]")
    args = parser.parse_args()

    with h5py.File(args.infile, "r") as h5:
        data = h5["ds"][:]
        f = h5["f"][:]
        c = h5["c"][:]

    if data.ndim != 3:
        raise ValueError("ds must be 3D: [num_images, nc, nf]")
    if data.shape[1] != len(c) or data.shape[2] != len(f):
        raise ValueError("ds shape must match c/f: ds[:, nc, nf]")

    root = tk.Tk()
    DispersionSpectrumGUI(f, c, data, args.outfile, root)
    root.mainloop()


if __name__ == "__main__":
    main()
