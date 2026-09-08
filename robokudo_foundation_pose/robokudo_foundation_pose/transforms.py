"""The two ``pytorch3d.transforms`` functions FoundationPose needs.

FoundationPose imports exactly two names from PyTorch3D, both of which are plain
tensor math with no compiled kernel behind them::

    from pytorch3d.transforms import rotation_6d_to_matrix, so3_exp_map

Building PyTorch3D from source just to get them is not worth it here: its
``pulsar`` point renderer fails to link against CUDA 13 (``undefined reference
to pulsar::Renderer::render<true>``), and the compile takes a quarter of an hour
before reaching that error.  So the two functions are reproduced here, faithful
to the upstream implementations (PyTorch3D, BSD-3-Clause, Meta Platforms).

Only ``so3_exp_map`` is actually reached with the shipped weights - the refiner
config sets ``rot_rep: axis_angle`` - but both are provided so either branch of
``PoseRefinePredictor`` works.
"""

import torch
import torch.nn.functional as F


def hat(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric ('hat') operator.

    Maps each 3-vector to the matrix that implements a cross product with it,
    so that ``hat(a) @ b == torch.cross(a, b)``.

    :param torch.Tensor v: N x 3
    :return: N x 3 x 3
    :rtype: torch.Tensor
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def so3_exp_map(log_rot: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Exponential map of so(3): axis-angle vectors to rotation matrices.

    Rodrigues' formula, ``R = I + sin(t)/t * K + (1-cos(t))/t^2 * K^2``, with
    ``K = hat(log_rot)`` and ``t`` the rotation angle.

    ``eps`` clamps the *squared* angle before the square root, which keeps the
    ``1/t`` factors finite at ``t = 0``; the result is still exactly the
    identity there, because ``K`` and ``K^2`` both vanish.

    :param torch.Tensor log_rot: N x 3, rotation axis scaled by the angle in radians
    :param float eps: lower clamp on the squared rotation angle
    :return: N x 3 x 3 rotation matrices
    :rtype: torch.Tensor
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)

    # phis ... squared norms
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles

    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())

    skews = hat(log_rot)                    # N x 3 x 3
    skews_square = torch.bmm(skews, skews)  # N x 3 x 3

    return (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation to rotation matrices.

    Takes the two 3-vectors of the 6D representation as the first two rows of
    the matrix and orthonormalises them with Gram-Schmidt; the third row is
    their cross product.  From Zhou et al., 'On the Continuity of Rotation
    Representations in Neural Networks' (CVPR 2019).

    :param torch.Tensor d6: (..., 6)
    :return: (..., 3, 3) rotation matrices
    :rtype: torch.Tensor
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)
