import numpy as np

from typing_extensions import Dict, List


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, List[int]]:
    """Converts a binary image mask into the uncompressed (COCO) RLE format.

    Note: Based on "https://stackoverflow.com/a/76990451"

    :param np.ndarray binary_mask: H x W  (also H x W x 1 or 1 x H x W)
    :return: Dict containing the mask shape ('size') and the RLE encoded mask contend ('count').
    :rtype: Dict[str, List[int]]
    """
    rle = {"counts": [], "size": list(binary_mask.squeeze().shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle      # {'counts': N, 'size': 2,}


def rle_to_mask(rle: Dict[str, List[int]]) -> np.ndarray:
    """Compute a binary mask from an uncompressed (COCO) RLE format.

    :param Dict[str, List[int]] rle: Compressed masks im RLE format
    :return: Uncompressed binary H x W mask
    :rtype: np.ndarray
    """
    height, width = rle["size"]
    mask = np.empty(height * width, dtype=bool)		# (H*W)

    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx: idx + count] = parity
        idx += count
        parity ^= True

    # put in C order
    mask = mask.reshape(width, height).transpose()		# H x W

    return mask		# H x W
