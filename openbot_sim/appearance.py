"""模块 8：外观随机化（Appearance）

几何归 World，颜色归 Camera —— 这个模块负责第三件事：**让同一段几何在
不同"场景实例"里长得不一样**，这样评估出来的成功率才是对场景分布的估计，
而不是对某一个场景的记忆。

随机化的四个因素（全部由种子决定，可复现）：

    floor texture    程序化值噪声，铺在地面上
    illumination     整体增益 gain、gamma、色温 colour temperature
    sensor noise     逐像素高斯噪声
    obstacles        地面上的矩形障碍（由 World 提供几何，这里只上色）

设计要点
--------
- 每个条件（condition）用**固定种子**生成一份 appearance；同一个种子在
  两个策略上重放，因此两个策略看到的是**同一批场景**，可以做配对比较。
- 不引入任何随机性到几何里：障碍物的位置来自 ConditionDesign 的显式取值，
  appearance 只负责"长什么样"。

对外接口：

    spec = sample_appearance(seed)           # -> AppearanceSpec
    apply_appearance(img, spec, px, py, ground, rng)

其中 ``px, py`` 是每个像素对应的地面落点（相机已经算好了），因此纹理可以
贴着世界坐标走，不会随车的位置漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

__all__ = ["AppearanceSpec", "sample_appearance", "apply_appearance", "OBSTACLE_RGB"]


OBSTACLE_RGB = (0.62, 0.34, 0.16)


@dataclass(frozen=True)
class AppearanceSpec:
    """One reproducible scene appearance."""

    seed: int
    texture_strength: float   # 0 = 纯色地面，越大纹理越明显
    texture_scale: float      # 1/米；格子越小纹理越细
    gain: float               # 整体亮度增益
    gamma: float              # 对比度（<1 提亮暗部）
    colour_temp: float        # -1 偏冷 / +1 偏暖
    noise_sigma: float        # 传感器噪声标准差
    vignette: float           # 边缘压暗强度

    def to_dict(self) -> dict:
        return asdict(self)


def sample_appearance(seed: int) -> AppearanceSpec:
    """Draw one appearance from the design distribution (deterministic in seed)."""
    rng = np.random.default_rng(int(seed) ^ 0x9E3779B9)
    return AppearanceSpec(
        seed=int(seed),
        texture_strength=float(rng.uniform(0.0, 0.30)),
        texture_scale=float(rng.uniform(6.0, 26.0)),
        gain=float(rng.uniform(0.70, 1.30)),
        gamma=float(rng.uniform(0.75, 1.35)),
        colour_temp=float(rng.uniform(-0.6, 0.6)),
        noise_sigma=float(rng.uniform(0.0, 0.025)),
        vignette=float(rng.uniform(0.0, 0.35)),
    )


def _lattice_noise(px: np.ndarray, py: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Value noise on the ground plane: bilinear interpolation of a tiny RNG grid.

    Tiled, so the texture repeats every ``grid`` lattice cells; stable in world
    coordinates, so the appearance of the floor does not slide as the vehicle
    moves.
    """
    grid = 8
    lat = np.random.default_rng(int(seed) & 0xFFFFFFFF).random((grid, grid)).astype(np.float32)
    u = (px * scale) % grid
    v = (py * scale) % grid
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = (u0 + 1) % grid
    v1 = (v0 + 1) % grid
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)
    fu = fu * fu * (3.0 - 2.0 * fu)     # smoothstep
    fv = fv * fv * (3.0 - 2.0 * fv)
    a = lat[v0 % grid, u0 % grid]
    b = lat[v0 % grid, u1 % grid]
    c = lat[v1 % grid, u0 % grid]
    d = lat[v1 % grid, u1 % grid]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def apply_appearance(
    img: np.ndarray,
    spec: AppearanceSpec,
    px: np.ndarray,
    py: np.ndarray,
    ground: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply texture, illumination, vignette and sensor noise in place.

    ``img`` is ``(H, W, 3)`` float in [0, 1]; ``px``/``py`` are the ground-plane
    hit points for every pixel and ``ground`` the mask of pixels that hit the
    floor.  Returns a new array.
    """
    out = img.astype(np.float32, copy=True)

    if spec.texture_strength > 0 and ground.any():
        safe_px = np.where(ground, px, 0.0)
        safe_py = np.where(ground, py, 0.0)
        noise = _lattice_noise(safe_px, safe_py, spec.texture_scale, spec.seed)
        mult = 1.0 + spec.texture_strength * (noise - 0.5) * 2.0
        mult = np.where(ground, mult, 1.0)[..., None]
        out *= mult

    # illumination: per-channel gain from colour temperature, then gamma
    warm = spec.colour_temp
    chan_gain = np.array([1.0 + 0.18 * warm, 1.0, 1.0 - 0.18 * warm], dtype=np.float32)
    out *= (spec.gain * chan_gain)[None, None, :]
    out = np.clip(out, 0.0, 1.0) ** np.float32(spec.gamma)

    if spec.vignette > 0:
        h, w = out.shape[:2]
        yy = (np.arange(h, dtype=np.float32) / max(h - 1, 1)) - 0.5
        xx = (np.arange(w, dtype=np.float32) / max(w - 1, 1)) - 0.5
        r2 = (yy[:, None] ** 2 + xx[None, :] ** 2) / 0.5
        out *= (1.0 - spec.vignette * np.clip(r2, 0.0, 1.0))[..., None]

    if spec.noise_sigma > 0:
        gen = rng if rng is not None else np.random.default_rng(spec.seed + 1)
        out += gen.normal(0.0, spec.noise_sigma, size=out.shape).astype(np.float32)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
