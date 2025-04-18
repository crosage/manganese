import cv2
import math
import numpy as np
import sys
import os
from pathlib import Path
import rasterio
import tifffile
from tqdm import tqdm
import warnings
from typing import Tuple, Optional

# --- Constants ---
NUMPY_EPS = np.finfo(np.float64).eps
UINT16_MAX = 65535.0

DEFAULT_DARK_CHANNEL_PATCH_SIZE = 15
DEFAULT_TRANSMISSION_PATCH_SIZE = 15
DEFAULT_GUIDED_FILTER_RADIUS = 60
DEFAULT_GUIDED_FILTER_EPS = 0.0001
DEFAULT_TRANSMISSION_THRESHOLD = 0.1
DEFAULT_OMEGA = 0.99

DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 128
DEFAULT_GLOBAL_A_DOWNSAMPLE_FACTOR = 16

# --- Helper Functions ---

def check_input_image(im: np.ndarray, expected_channels: int, func_name: str) -> None:
    """Validates the input image dimensions and number of channels."""
    if im.ndim != 3 or im.shape[2] != expected_channels:
        raise ValueError(
            f"{func_name}: 输入图像需要是 HxWx{expected_channels} 格式, 收到 {im.shape}"
        )
    if im.dtype != np.float64:
        warnings.warn(
            f"{func_name}: 输入图像数据类型为 {im.dtype}, 期望 float64. "
            "内部计算将使用 float64。",
            UserWarning
        )

# --- Core Dehazing Functions (Optimized for 4 Channels) ---

def dark_channel(im: np.ndarray, sz: int) -> np.ndarray:
    """
    计算 N 通道图像的暗通道图。

    Args:
        im: 输入图像 (HxWxN, float64, 范围 [0, 1])。
        sz: 计算暗通道时使用的邻域（patch）的大小（边长）。

    Returns:
        暗通道图 (HxW, float64)。
    """
    n_channels = im.shape[2]
    check_input_image(im, n_channels, "dark_channel")

    dc = np.min(im, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def atmospheric_light(im: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    根据暗通道图估计 N 通道图像的全局大气光 A。

    Args:
        im: 原始输入图像 (HxWxN, float64, 范围 [0, 1])。
        dark: 暗通道图 (HxW, float64)。

    Returns:
        估计的大气光值 A (1xN, float64)。
    """
    n_channels = im.shape[2]
    check_input_image(im, n_channels, "atmospheric_light")
    if dark.ndim != 2 or dark.shape[:2] != im.shape[:2]:
         raise ValueError(f"atmospheric_light: 暗通道图尺寸 {dark.shape} 与图像尺寸 {im.shape[:2]} 不匹配")

    h, w = im.shape[:2]
    imsz = h * w
    num_pixels_to_consider = int(max(math.floor(imsz / 1000), 1))

    dark_flat = dark.reshape(imsz)
    im_flat = im.reshape(imsz, n_channels)

    brightest_indices = np.argsort(dark_flat)[imsz - num_pixels_to_consider:]

    A = np.mean(im_flat[brightest_indices], axis=0, keepdims=True)

    return A

def transmission_estimate(im: np.ndarray, A: np.ndarray, sz: int, omega: float = DEFAULT_OMEGA) -> np.ndarray:
    """
    初步估计 N 通道图像的透射率图。

    Args:
        im: 原始输入图像 (HxWxN, float64, 范围 [0, 1])。
        A: 估计的大气光 (1xN, float64)。
        sz: 计算暗通道时使用的邻域大小。
        omega: 保留少量雾的系数。

    Returns:
        初步估计的透射率图 (HxW, float64)。
    """
    n_channels = im.shape[2]
    check_input_image(im, n_channels, "transmission_estimate")
    if A.shape != (1, n_channels):
        raise ValueError(f"transmission_estimate: 大气光 A 形状错误. A:{A.shape}, 需要 (1, {n_channels})")

    A_clipped = np.maximum(A, NUMPY_EPS)
    im_normalized = np.empty_like(im)
    for i in range(n_channels):
        im_normalized[:, :, i] = im[:, :, i] / A_clipped[0, i]

    transmission = 1.0 - omega * dark_channel(im_normalized, sz)
    return transmission

def guided_filter(im_guide: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    引导滤波实现。

    Args:
        im_guide: 引导图像 (HxW 或 HxWxC, float64)。建议使用灰度图。
        p: 需要滤波的输入图像 (HxW, float64)。
        r: 滤波器的窗口半径。
        eps: 正则化参数。

    Returns:
        滤波后的图像 (HxW, float64)。
    """
    if im_guide.dtype != np.float64: im_guide = im_guide.astype(np.float64)
    if p.dtype != np.float64: p = p.astype(np.float64)

    if im_guide.ndim == 3:
        warnings.warn("guided_filter: 引导图像是多通道，将使用其平均值进行滤波。", UserWarning)
        guide = np.mean(im_guide, axis=2)
    else:
        guide = im_guide

    mean_I = cv2.boxFilter(guide, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(guide * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * guide + mean_b
    return q

def transmission_refine(im_float_bgrn: np.ndarray, et: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    使用引导滤波优化透射率图。使用 BGR 通道转化的灰度图作为引导。

    Args:
        im_float_bgrn: 原始输入图像 (HxWx4, float64, 范围 [0, 1])。
        et: 初步估计的透射率图 (HxW, float64)。
        r: 引导滤波半径。
        eps: 引导滤波正则化参数。

    Returns:
        优化后的透射率图 (HxW, float64)。
    """
    check_input_image(im_float_bgrn, 4, "transmission_refine")

    im_float_bgr = im_float_bgrn[:, :, :3]
    if im_float_bgr.dtype != np.float32:
        im_float_bgr = im_float_bgr.astype(np.float32)

    gray_guide = cv2.cvtColor(im_float_bgr, cv2.COLOR_BGR2GRAY)

    t = guided_filter(gray_guide, et, r, eps)
    return t

def recover(im: np.ndarray, t: np.ndarray, A: np.ndarray, tx: float = DEFAULT_TRANSMISSION_THRESHOLD) -> np.ndarray:
    """
    根据大气散射模型恢复 N 通道无雾图像。

    Args:
        im: 原始输入图像 (HxWxN, float64, 范围 [0, 1])。
        t: 优化后的透射率图 (HxW, float64)。
        A: 估计的大气光 (1xN, float64)。
        tx: 透射率的下限阈值。

    Returns:
        恢复的无雾图像 (HxWxN, float64)，值被裁剪到 [0, 1]。
    """
    n_channels = im.shape[2]
    check_input_image(im, n_channels, "recover")
    if A.shape != (1, n_channels):
        raise ValueError(f"recover: 大气光 A 形状错误. A:{A.shape}, 需要 (1, {n_channels})")
    if t.ndim != 2 or t.shape[:2] != im.shape[:2]:
        raise ValueError(f"recover: 透射率图 t 尺寸 {t.shape} 与图像尺寸 {im.shape[:2]} 不匹配")

    t_clipped = np.maximum(t, tx)

    t_expanded = np.expand_dims(t_clipped, axis=2)

    res = (im - A) / t_expanded + A

    return np.clip(res, 0, 1)

# --- File Handling and Tiling ---

def estimate_global_A(filepath: Path, scale_factor: int, dark_patch_sz: int) -> Optional[np.ndarray]:
    """
    从低分辨率图像估算全局大气光 A。

    Args:
        filepath: 输入图像文件路径 (TIFF)。
        scale_factor: 下采样因子。
        dark_patch_sz: 计算暗通道时使用的 patch 大小。

    Returns:
        估算的全局大气光 A (1x4)，如果出错则返回 None。
    """
    print(f"开始估算全局大气光 A (下采样因子: {scale_factor})...")
    try:
        with tifffile.TiffFile(filepath) as tif:
            if not tif.series or len(tif.series[0].shape) < 3:
                raise ValueError(f"TIFF 文件 {filepath} 无效或缺少有效的 Series/Page 或维度不足。")

            original_shape = tif.series[0].shape
            original_dtype = tif.series[0].dtype
            num_channels = original_shape[2]

            if num_channels != 4:
                raise ValueError(f"图像通道数 ({num_channels}) 不是 4")

            H, W = original_shape[:2]
            new_H, new_W = H // scale_factor, W // scale_factor

            if new_H < dark_patch_sz or new_W < dark_patch_sz:
                print(f"警告：下采样后尺寸 ({new_H}, {new_W}) 小于暗通道patch尺寸 ({dark_patch_sz})。")
                print("将尝试使用原始图像估算 A（可能耗尽内存）。")
                scale_factor = 1
                new_H, new_W = H, W
                img_low_res_uint = tif.series[0].asarray()
            else:
                print(f"  读取并下采样图像到 ({new_H}, {new_W})...")
                try:
                    img_full_res_uint = tif.series[0].asarray()
                    img_low_res_uint = cv2.resize(img_full_res_uint, (new_W, new_H), interpolation=cv2.INTER_AREA)
                    del img_full_res_uint
                except MemoryError:
                    print("错误：读取完整图像以进行下采样时内存不足。无法估算全局 A。")
                    print("建议：尝试更大的 scale_factor 或确保有足够内存。")
                    return None
                except Exception as e:
                    print(f"读取或调整图像大小时出错: {e}")
                    return None

            if img_low_res_uint.dtype != np.uint16:
                warnings.warn(
                    f"读取的下采样图像数据类型为 {img_low_res_uint.dtype}, 期望 uint16。将进行转换。",
                    UserWarning
                )
                img_low_res_uint = img_low_res_uint.astype(np.uint16)

            img_low_res_float = img_low_res_uint.astype(np.float64) / UINT16_MAX
            del img_low_res_uint

            print("  计算下采样图像的暗通道...")
            dark_low = dark_channel(img_low_res_float, dark_patch_sz)

            print("  估计大气光值...")
            global_A = atmospheric_light(img_low_res_float, dark_low)

            print(f"全局大气光 A 估算完成: {np.array2string(global_A, precision=4, floatmode='fixed')}")
            return global_A

    except FileNotFoundError:
        print(f"错误：输入文件未找到 {filepath}")
        return None
    except ValueError as ve:
        print(f"错误：处理 TIFF 文件时值错误: {ve}")
        return None
    except Exception as e:
        print(f"估算全局大气光 A 时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_tile(padded_tile_src_uint16: np.ndarray, global_A: np.ndarray) -> Optional[np.ndarray]:
    """
    对单个带重叠的块进行去雾处理 (uint16 输入, float64 处理, float64 输出 [0,1])

    Args:
        padded_tile_src_uint16: 输入的带重叠块 (HxWxC, uint16)。
        global_A: 估算的全局大气光 (1xN, float64)。

    Returns:
        处理后的带重叠块 (HxWxC, float64, 范围 [0,1])，如果出错则返回 None。
    """
    try:
        if padded_tile_src_uint16.dtype != np.uint16:
             padded_tile_src_uint16 = padded_tile_src_uint16.astype(np.uint16)

        padded_tile_I = padded_tile_src_uint16.astype(np.float64) / UINT16_MAX
        del padded_tile_src_uint16

        dark_t = dark_channel(padded_tile_I, DEFAULT_DARK_CHANNEL_PATCH_SIZE)
        te_t = transmission_estimate(padded_tile_I, global_A, DEFAULT_TRANSMISSION_PATCH_SIZE, DEFAULT_OMEGA)
        t_t = transmission_refine(padded_tile_I, te_t, DEFAULT_GUIDED_FILTER_RADIUS, DEFAULT_GUIDED_FILTER_EPS)
        J_tile_padded_float = recover(padded_tile_I, t_t, global_A, DEFAULT_TRANSMISSION_THRESHOLD)

        return J_tile_padded_float

    except MemoryError:
        print(f"\n错误：处理块时内存不足！请尝试减小 tile_size 或 overlap。")
        raise MemoryError("Tile processing failed due to insufficient memory.")
    except ValueError as ve:
        print(f"\n错误: 处理块时发生值错误: {ve}")
        return None
    except Exception as e:
        print(f"\n错误：处理块时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_dehazing_pipeline(input_filepath: Path, output_filepath: Path):
    """
    执行完整的分块去雾流程

    Args:
        input_filepath: 输入 TIFF 图像文件路径。
        output_filepath: 输出去雾后 TIFF 图像文件路径。
    """

    global_A = estimate_global_A(input_filepath,
                                 scale_factor=DEFAULT_GLOBAL_A_DOWNSAMPLE_FACTOR,
                                 dark_patch_sz=DEFAULT_DARK_CHANNEL_PATCH_SIZE)
    if global_A is None:
        print("无法估算全局大气光 A，处理中止。")
        return

    print("\n开始分块去雾处理 (使用 Rasterio)...")
    try:
        with rasterio.open(input_filepath) as src_dataset:
            H, W = src_dataset.height, src_dataset.width
            num_channels = src_dataset.count
            if num_channels != 4:
                 raise ValueError(f"错误：输入文件通道数 ({num_channels}) 不是 4。")
            if src_dataset.dtypes[0] != 'uint16':
                 warnings.warn(
                    f"输入文件数据类型为 {src_dataset.dtypes[0]}, 期望 uint16。处理仍将继续。",
                    UserWarning
                 )

            profile = src_dataset.profile
            profile.update(dtype=rasterio.uint16, count=num_channels, nodata=None, compress=None, tiled=False)

            print(f"图像尺寸: {H}x{W}, 通道数: {num_channels}")
            print(f"分块设置: tile_size={DEFAULT_TILE_SIZE}, overlap={DEFAULT_OVERLAP}")

            print(f"创建输出文件: {output_filepath}")
            with rasterio.open(output_filepath, 'w', **profile) as dst_dataset:

                num_tiles_y = math.ceil(H / DEFAULT_TILE_SIZE)
                num_tiles_x = math.ceil(W / DEFAULT_TILE_SIZE)
                total_tiles = num_tiles_y * num_tiles_x
                print(f"总块数: {total_tiles} ({num_tiles_y} x {num_tiles_x})")

                pbar = tqdm(total=total_tiles, desc="处理块", unit="块")
                for y_idx in range(num_tiles_y):
                    for x_idx in range(num_tiles_x):
                        y_start = y_idx * DEFAULT_TILE_SIZE
                        y_end = min(y_start + DEFAULT_TILE_SIZE, H)
                        x_start = x_idx * DEFAULT_TILE_SIZE
                        x_end = min(x_start + DEFAULT_TILE_SIZE, W)
                        write_width = x_end - x_start
                        write_height = y_end - y_start

                        read_y_start = max(0, y_start - DEFAULT_OVERLAP)
                        read_y_end = min(H, y_end + DEFAULT_OVERLAP)
                        read_x_start = max(0, x_start - DEFAULT_OVERLAP)
                        read_x_end = min(W, x_end + DEFAULT_OVERLAP)
                        read_width = read_x_end - read_x_start
                        read_height = read_y_end - read_y_start

                        read_window = rasterio.windows.Window(read_x_start, read_y_start, read_width, read_height)
                        write_window = rasterio.windows.Window(x_start, y_start, write_width, write_height)

                        try:
                            padded_tile_chw = src_dataset.read(window=read_window)
                        except Exception as e:
                            print(f"\n错误：Rasterio 读取块 ({y_idx},{x_idx}) 时出错: {e}")
                            pbar.update(1)
                            continue

                        if padded_tile_chw is None or padded_tile_chw.size == 0:
                            print(f"\n警告：Rasterio 读取块 ({y_idx},{x_idx}) 结果为空，跳过。")
                            pbar.update(1)
                            continue
                        if padded_tile_chw.shape != (num_channels, read_height, read_width):
                             print(f"\n警告：读取块 ({y_idx},{x_idx}) 形状不匹配 ({padded_tile_chw.shape})，预期 ({num_channels}, {read_height}, {read_width})，跳过。")
                             pbar.update(1)
                             continue

                        padded_tile_hwc_uint16 = np.transpose(padded_tile_chw, (1, 2, 0))
                        del padded_tile_chw

                        J_tile_padded_float = process_tile(padded_tile_hwc_uint16, global_A)
                        del padded_tile_hwc_uint16

                        if J_tile_padded_float is None:
                            pbar.update(1)
                            continue

                        inner_y_start = y_start - read_y_start
                        inner_y_end = inner_y_start + write_height
                        inner_x_start = x_start - read_x_start
                        inner_x_end = inner_x_start + write_width

                        J_tile_valid_float = J_tile_padded_float[inner_y_start:inner_y_end, inner_x_start:inner_x_end, :]
                        del J_tile_padded_float

                        if J_tile_valid_float.shape != (write_height, write_width, num_channels):
                            print(f"\n警告：提取的有效区域形状 ({J_tile_valid_float.shape}) 与预期 ({write_height, write_width, num_channels}) 不符，跳过写入块 ({y_idx},{x_idx})。")
                            pbar.update(1)
                            continue

                        J_tile_out_hwc_uint16 = (J_tile_valid_float * UINT16_MAX).clip(0, UINT16_MAX).astype(np.uint16)
                        del J_tile_valid_float

                        J_tile_out_chw = np.transpose(J_tile_out_hwc_uint16, (2, 0, 1))
                        del J_tile_out_hwc_uint16

                        try:
                            dst_dataset.write(J_tile_out_chw, window=write_window)
                        except Exception as e:
                             print(f"\n错误：Rasterio 写入块 ({y_idx},{x_idx}) 时出错: {e}")
                        del J_tile_out_chw

                        pbar.update(1)

                pbar.close()
            print("\n所有块处理完成，输出文件已保存。")

    except rasterio.RasterioIOError as e:
        print(f"\nRasterio 文件 IO 错误: {e}")
        import traceback
        traceback.print_exc()
    except MemoryError:
         print("\n处理因内存不足而中止。")
    except ValueError as ve:
        print(f"\n处理因值错误而中止: {ve}")
    except Exception as e:
        print(f"\n处理过程中发生未捕获的严重错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n脚本执行完毕。")


# --- Main Execution ---
if __name__ == '__main__':
    default_input = Path(r'D:\Train\origin\GF1_PMS1_E51.6_N49.6_20231106_L1A13154817001-MSS1_fuse.tiff')
    default_output = Path('./result_dehazed_tiled_rasterio_refactored_no_inline_comments.tif')

    input_file = default_input
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
        print(f"使用命令行提供的输入文件: {input_file}")
    else:
        print(f"未提供输入文件，使用默认文件: {input_file}")

    if not input_file.exists():
        print(f"错误：输入文件不存在！ {input_file}")
        sys.exit(1)

    output_file = default_output
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
        print(f"使用命令行提供的输出文件: {output_file}")
    else:
         print(f"使用默认输出文件: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    run_dehazing_pipeline(input_file, output_file)