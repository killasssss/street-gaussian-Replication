#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


def IDFT(time, dim):
    """
    该函数用于计算 **归一化的逆离散傅里叶变换 (Inverse Discrete Fourier Transform, IDFT) 基矩阵**，
    主要用于时间信息的频域编码。常见于深度学习中的 **时间特征处理、位置编码** 等场景。

    输入：
    - time (float 或 torch.Tensor): 需要进行傅里叶变换的时间值（或时间序列）。
    - dim (int): 生成 IDFT 矩阵的维度，即傅里叶基的数量。

    输出：
    - idft (torch.Tensor): 形状为 `(len(time), dim)` 的 IDFT 矩阵，其中每一行是对应 `time` 值的 IDFT 变换向量。
    """

    # 如果 `time` 是浮点数，则转换为 PyTorch 张量   isinstance(object, classinfo) 是 Python 内置的 类型检查函数，用于判断某个对象是否是指定类型或类型元组中的一个。
    if isinstance(time, float):
        time = torch.tensor(time)

    # 将时间 `time` 转换为 2D 形状 `(N, 1)`，其中 `N` 是时间点的数量  view(-1, 1)中-1代表行数自动，列数为1
    t = time.view(-1, 1).float()  # 确保 `t` 是浮点型，以便进行后续计算

    # 初始化 IDFT 变换矩阵，形状为 `(N, dim)`
    idft = torch.zeros(t.shape[0], dim)  # 生成零矩阵，后续填充

    # 生成从 0 到 dim-1 的索引序列
    indices = torch.arange(dim)  # `indices = [0, 1, 2, ..., dim-1]`
    # 选取偶数索引（用于余弦计算）
    even_indices = indices[::2]  # `even_indices = [0, 2, 4, ..., dim-2]`sequence[start:end:step] start：切片起始索引（包含） end:切片结束索引（不包含） step步长（步进大小）
    # 选取奇数索引（用于正弦计算）
    odd_indices = indices[1::2]  # `odd_indices = [1, 3, 5, ..., dim-1]`

    # 在 `idft` 矩阵的偶数列填充 `cos(π * t * even_indices)`
    idft[:, even_indices] = torch.cos(torch.pi * t * even_indices)

    # 在 `idft` 矩阵的奇数列填充 `sin(π * t * (odd_indices + 1))`
    # 这里 `odd_indices + 1` 的目的是保证正弦函数的周期性不偏移
    idft[:, odd_indices] = torch.sin(torch.pi * t * (odd_indices + 1))

    return idft
