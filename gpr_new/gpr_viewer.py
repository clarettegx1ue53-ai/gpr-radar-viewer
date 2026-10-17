"""A lightweight 3-D GPR NPY viewer built with Tkinter and NumPy."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np


X_LENGTH_M = 1000.0
Y_WIDTH_M = 1.5
Z_DEPTH_M = 1.25


class GPRViewer(tk.Tk):
    """Show the X, Y and Z orthogonal slices of a 3-D radar volume."""

    def __init__(self) -> None:
        super().__init__()
        self.title("三维 GPR 数据可视化")
        self.geometry("1500x900")
        self.minsize(1000, 650)
        self.configure(bg="#edf1f5")

        self.volume: np.ndarray | None = None  # normalized order: [z, y, x]
        self.file_path: Path | None = None
        self.slice_indices = {"x": 0, "y": 0, "z": 0}
        self.photo_refs: list[tk.PhotoImage] = []
        self._render_after: str | None = None

        self.order_var = tk.StringVar(value="ZYX（深度、宽度、长度）")
        self.status_var = tk.StringVar(value="请导入三维 .npy 文件")
        self.data_info_var = tk.StringVar(value="尚未载入数据")
        self.slice_value_vars = {axis: tk.StringVar(value="—") for axis in "xyz"}
        self.slice_scale_vars = {axis: tk.DoubleVar(value=0) for axis in "xyz"}
        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self, bg="#172b4d", height=72)
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)
        tk.Label(
            top, text="三维 GPR 数据可视化", fg="white", bg="#172b4d",
            font=("Microsoft YaHei UI", 19, "bold"),
        ).pack(side=tk.LEFT, padx=22)
        ttk.Button(top, text="导入 NPY 文件", command=self.open_file).pack(
            side=tk.RIGHT, padx=20, pady=17
        )

        body = tk.Frame(self, bg="#edf1f5")
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(14, 8))

        sidebar = tk.Frame(body, bg="white", width=270, highlightbackground="#d8e0e8", highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="数据与切面", bg="white", fg="#172b4d",
                 font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w", padx=18, pady=(18, 12))

        tk.Label(sidebar, text="NPY 数组维度顺序", bg="white", fg="#53657a",
                 font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=18)
        self.order_box = ttk.Combobox(
            sidebar, textvariable=self.order_var, state="readonly",
            values=("ZYX（深度、宽度、长度）", "XYZ（长度、宽度、深度）",
                    "YXZ（宽度、长度、深度）"),
        )
        self.order_box.pack(fill=tk.X, padx=18, pady=(5, 7))
        self.order_box.bind("<<ComboboxSelected>>", self._order_changed)
        tk.Label(sidebar, text="默认按 shape=(深度, 宽度, 长度) 读取。",
                 bg="white", fg="#77869a", justify=tk.LEFT, wraplength=225,
                 font=("Microsoft YaHei UI", 9)).pack(anchor="w", padx=18, pady=(0, 14))

        ttk.Separator(sidebar).pack(fill=tk.X, padx=18, pady=5)
        self.info_label = tk.Label(sidebar, textvariable=self.data_info_var, bg="white", fg="#33445b",
                                   justify=tk.LEFT, anchor="nw", wraplength=225,
                                   font=("Microsoft YaHei UI", 9), height=5)
        self.info_label.pack(fill=tk.X, padx=18, pady=(10, 6))

        self._add_slice_control(sidebar, "x", "X 切面位置", "沿长度 0–1000 m")
        self._add_slice_control(sidebar, "y", "Y 切面位置", "沿宽度 0–1.5 m")
        self._add_slice_control(sidebar, "z", "Z 切面位置", "沿深度 0–1.25 m")

        tk.Label(sidebar, text="坐标范围", bg="white", fg="#172b4d",
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(sidebar, text="X 长度：1000 m\nY 宽度：1.5 m\nZ 深度：1.25 m",
                 bg="white", fg="#53657a", justify=tk.LEFT,
                 font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=18)

        plot_frame = tk.Frame(body, bg="white", highlightbackground="#d8e0e8", highlightthickness=1)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(plot_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", self._schedule_render)
        self._draw_empty_state()

        bottom = tk.Frame(self, bg="#edf1f5", height=28)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=18, pady=(0, 6))
        tk.Label(bottom, textvariable=self.status_var, bg="#edf1f5", fg="#53657a",
                 anchor="w", font=("Microsoft YaHei UI", 9)).pack(fill=tk.X)

    def _add_slice_control(self, parent: tk.Widget, axis: str, title: str, hint: str) -> None:
        group = tk.Frame(parent, bg="white")
        group.pack(fill=tk.X, padx=18, pady=(8, 0))
        row = tk.Frame(group, bg="white")
        row.pack(fill=tk.X)
        tk.Label(row, text=title, bg="white", fg="#172b4d",
                 font=("Microsoft YaHei UI", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(row, textvariable=self.slice_value_vars[axis], bg="white", fg="#1769aa",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.RIGHT)
        tk.Label(group, text=hint, bg="white", fg="#8996a7",
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=(1, 0))
        scale = ttk.Scale(group, from_=0, to=0, variable=self.slice_scale_vars[axis],
                          command=lambda value, a=axis: self._slice_changed(a, value))
        scale.pack(fill=tk.X, pady=(1, 0))
        setattr(self, f"{axis}_scale", scale)

    def _draw_empty_state(self) -> None:
        self.canvas.delete("all")
        w, h = max(self.canvas.winfo_width(), 600), max(self.canvas.winfo_height(), 400)
        self.canvas.create_text(w / 2, h / 2 - 18, text="导入三维 GPR NPY 数据以查看切面",
                                fill="#53657a", font=("Microsoft YaHei UI", 16, "bold"))
        self.canvas.create_text(w / 2, h / 2 + 18,
                                text="将显示 X、Y、Z 三个方向的正交切面及距离坐标",
                                fill="#8996a7", font=("Microsoft YaHei UI", 11))

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择三维 NPY 文件", filetypes=(("NumPy 文件", "*.npy"), ("所有文件", "*.*"))
        )
        if not path:
            return
        try:
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            if loaded.ndim != 3:
                raise ValueError(f"需要三维数组，当前数组维数为 {loaded.ndim}。")
            if not np.issubdtype(loaded.dtype, np.number):
                raise ValueError("数组必须是数值类型。")
            if any(size < 1 for size in loaded.shape):
                raise ValueError("数组各维长度必须大于 0。")
            # Keep memory-mapped input where possible; transpose is a view.
            # Radar complex cubes are commonly saved as [line, trace, sample]
            # = [Y, X, Z]. The supplied file has the characteristic shape
            # (10, 13293, 511), so select that interpretation automatically.
            if (np.iscomplexobj(loaded) and loaded.shape[0] <= 32
                    and loaded.shape[1] > loaded.shape[0] * 10
                    and loaded.shape[2] > loaded.shape[0]):
                self.order_var.set("YXZ（宽度、长度、深度）")
            self.volume = self._normalize_order(loaded)
            self.file_path = Path(path)
            self.slice_indices = {
                "x": self.volume.shape[2] // 2,
                "y": self.volume.shape[1] // 2,
                "z": self.volume.shape[0] // 2,
            }
            self._update_controls()
            self._update_info()
            self.status_var.set(f"已载入：{self.file_path.name}")
            self._render()
        except Exception as exc:
            messagebox.showerror("无法载入文件", str(exc), parent=self)

    def _order_changed(self, _event: tk.Event | None = None) -> None:
        if self.file_path is None:
            return
        # Reload so the selected source-axis interpretation is applied.
        path = self.file_path
        try:
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            self.volume = self._normalize_order(loaded)
            self.slice_indices = {"x": self.volume.shape[2] // 2,
                                  "y": self.volume.shape[1] // 2,
                                  "z": self.volume.shape[0] // 2}
            self._update_controls()
            self._update_info()
            self._render()
        except Exception as exc:
            messagebox.showerror("无法读取数据", str(exc), parent=self)

    def _normalize_order(self, loaded: np.ndarray) -> np.ndarray:
        """Return a view ordered as [Z, Y, X] for the slice renderer."""
        order = self.order_var.get()
        if order.startswith("ZYX"):
            return loaded
        if order.startswith("XYZ"):
            return loaded.transpose(2, 1, 0)
        # Source [Y, X, Z] -> renderer [Z, Y, X].
        return loaded.transpose(2, 0, 1)

    def _update_controls(self) -> None:
        assert self.volume is not None
        for axis, size in (("x", self.volume.shape[2]), ("y", self.volume.shape[1]),
                           ("z", self.volume.shape[0])):
            scale = getattr(self, f"{axis}_scale")
            scale.configure(from_=0, to=max(size - 1, 0))
            self.slice_scale_vars[axis].set(self.slice_indices[axis])
            self._set_coordinate_label(axis)

    def _update_info(self) -> None:
        assert self.volume is not None and self.file_path is not None
        nz, ny, nx = self.volume.shape
        self.data_info_var.set(
            f"文件：{self.file_path.name}\n"
            f"数据类型：{self.volume.dtype}"
            f"{'（显示复数模值）' if np.iscomplexobj(self.volume) else ''}\n"
            f"标准维度 [Z,Y,X]：{nz} × {ny} × {nx}\n"
            f"显示范围：X 0–{X_LENGTH_M:g} m\nY 0–{Y_WIDTH_M:g} m\nZ 0–{Z_DEPTH_M:g} m"
        )

    def _slice_changed(self, axis: str, value: str) -> None:
        if self.volume is None:
            return
        self.slice_indices[axis] = int(round(float(value)))
        self._set_coordinate_label(axis)
        self._schedule_render()

    def _set_coordinate_label(self, axis: str) -> None:
        if self.volume is None:
            return
        dimension, size = {
            "x": (X_LENGTH_M, self.volume.shape[2]),
            "y": (Y_WIDTH_M, self.volume.shape[1]),
            "z": (Z_DEPTH_M, self.volume.shape[0]),
        }[axis]
        index = self.slice_indices[axis]
        coord = 0.0 if size <= 1 else index * dimension / (size - 1)
        self.slice_value_vars[axis].set(f"{coord:.3f} m")

    def _schedule_render(self, _event: tk.Event | None = None) -> None:
        if self._render_after is not None:
            self.after_cancel(self._render_after)
        self._render_after = self.after(80, self._render)

    @staticmethod
    def _rgb_image(data: np.ndarray, width: int, height: int,
                   low: float, high: float) -> tk.PhotoImage:
        """Create a downsampled grayscale Tk image from a 2-D numeric array."""
        if data.size == 0 or width < 1 or height < 1:
            return tk.PhotoImage(width=1, height=1)
        rows = np.linspace(0, data.shape[0] - 1, height).astype(np.intp)
        cols = np.linspace(0, data.shape[1] - 1, width).astype(np.intp)
        sampled_data = data[np.ix_(rows, cols)]
        if np.iscomplexobj(sampled_data):
            sampled_data = np.abs(sampled_data)
        sampled = np.asarray(sampled_data, dtype=np.float32)
        if high <= low:
            gray = np.full(sampled.shape, 127, dtype=np.uint8)
        else:
            scaled = np.nan_to_num((sampled - low) * (255.0 / (high - low)),
                                   nan=127.0, posinf=255.0, neginf=0.0)
            gray = np.clip(scaled, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
        # Tk's PPM decoder expects the actual P6 bytes here. Base64-encoding
        # PPM bytes is not accepted by Tk and caused a blank canvas on load.
        return tk.PhotoImage(data=ppm, format="PPM")

    def _render(self) -> None:
        self._render_after = None
        if self.volume is None:
            self._draw_empty_state()
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 100 or h < 100:
            return
        self.canvas.delete("all")
        self.photo_refs = []
        gap = 14
        panel_w = (w - 2 * gap - 22) / 3
        panel_h = h - 8
        plans = [
            ("X 切面 · 深度-宽度", self.volume[:, :, self.slice_indices["x"]],
             Y_WIDTH_M, Z_DEPTH_M, "宽度 Y (m)", "深度 Z (m)",
             self.slice_value_vars["x"].get()),
            ("Y 切面 · 深度-长度", self.volume[:, self.slice_indices["y"], :],
             X_LENGTH_M, Z_DEPTH_M, "长度 X (m)", "深度 Z (m)",
             self.slice_value_vars["y"].get()),
            ("Z 切面 · 宽度-长度", self.volume[self.slice_indices["z"], :, :],
             X_LENGTH_M, Y_WIDTH_M, "长度 X (m)", "宽度 Y (m)",
             self.slice_value_vars["z"].get()),
        ]
        try:
            finite_sample = self.volume[::max(1, self.volume.shape[0] // 80),
                                        ::max(1, self.volume.shape[1] // 80),
                                        ::max(1, self.volume.shape[2] // 300)]
            if np.iscomplexobj(finite_sample):
                finite_sample = np.abs(finite_sample)
            finite_sample = np.asarray(finite_sample, dtype=np.float32)
            finite = finite_sample[np.isfinite(finite_sample)]
            if finite.size:
                low, high = (float(x) for x in np.percentile(finite, (1, 99)))
            else:
                low, high = 0.0, 1.0
        except (ValueError, TypeError, OverflowError):
            low, high = 0.0, 1.0

        for i, (title, data, x_extent, y_extent, xlabel, ylabel, position) in enumerate(plans):
            left = 8 + i * (panel_w + gap)
            self._draw_panel(left, 4, panel_w, panel_h, title, data, x_extent, y_extent,
                             xlabel, ylabel, position, low, high)
        self.status_var.set(f"幅值显示范围（1–99 百分位）：{low:.4g} 至 {high:.4g}")

    def _draw_panel(self, left: float, top: float, width: float, height: float,
                    title: str, data: np.ndarray, x_extent: float, y_extent: float,
                    xlabel: str, ylabel: str, position: str,
                    low: float, high: float) -> None:
        c = self.canvas
        c.create_rectangle(left, top, left + width, top + height, fill="white", outline="#dce3ea")
        c.create_text(left + width / 2, top + 21, text=title, fill="#172b4d",
                      font=("Microsoft YaHei UI", 11, "bold"))
        c.create_text(left + width / 2, top + 43, text=f"切片坐标：{position}", fill="#1769aa",
                      font=("Microsoft YaHei UI", 9))

        # Reserve space for numeric ticks and axis names.
        plot_left, plot_top = left + 54, top + 64
        plot_right, plot_bottom = left + width - 12, top + height - 51
        pw, ph = max(1, int(plot_right - plot_left)), max(1, int(plot_bottom - plot_top))
        image = self._rgb_image(data, pw, ph, low, high)
        self.photo_refs.append(image)
        c.create_image(plot_left, plot_top, image=image, anchor="nw")
        c.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="#8996a7")

        # Tick marks, numeric distance coordinates and subtle grid lines.
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = plot_left + frac * (plot_right - plot_left)
            c.create_line(x, plot_bottom, x, plot_bottom + 4, fill="#53657a")
            c.create_text(x, plot_bottom + 13, text=self._format_distance(frac * x_extent),
                          fill="#53657a", font=("Microsoft YaHei UI", 8))
            if frac not in (0.0, 1.0):
                c.create_line(x, plot_top, x, plot_bottom, fill="#ffffff", stipple="gray25")
            y = plot_top + frac * (plot_bottom - plot_top)
            c.create_line(plot_left - 4, y, plot_left, y, fill="#53657a")
            # Image row zero is shown at the top. Depth therefore reads 0 at
            # the top and increases downward, matching the displayed samples.
            coord = frac * y_extent
            c.create_text(plot_left - 7, y, text=self._format_distance(coord), anchor="e",
                          fill="#53657a", font=("Microsoft YaHei UI", 8))
            if frac not in (0.0, 1.0):
                c.create_line(plot_left, y, plot_right, y, fill="#ffffff", stipple="gray25")
        c.create_text((plot_left + plot_right) / 2, top + height - 12, text=xlabel,
                      fill="#33445b", font=("Microsoft YaHei UI", 9))
        c.create_text(left + 13, (plot_top + plot_bottom) / 2, text=ylabel,
                      fill="#33445b", font=("Microsoft YaHei UI", 9), angle=90)
        c.create_text(left + width - 15, top + height - 29, text=f"幅值 {low:.3g} … {high:.3g}",
                      fill="#77869a", anchor="e", font=("Microsoft YaHei UI", 8))

    @staticmethod
    def _format_distance(value: float) -> str:
        return f"{value:.0f}" if abs(value) >= 10 else f"{value:.2f}" if value else "0"


if __name__ == "__main__":
    GPRViewer().mainloop()
