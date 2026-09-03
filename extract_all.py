"""Extract a FoundationPose-ready RGB-D dataset from an MCAP recording.

Writes the layout the FoundationPose demos expect:

    <out>/rgb/00000.jpg      colour frames (raw JPEG bytes, no re-encode)
    <out>/depth/00000.png    uint16 depth in millimetres, index-aligned to rgb/
    <out>/cam_K.txt          3x3 intrinsic matrix
    <out>/camera_info.json   full CameraInfo (distortion, P, R, ...)
    <out>/tf_static.json     all static transforms
    <out>/frame_index.csv    per-pair timestamps and the colour/depth time offset
    <out>/masks/             created empty - FoundationPose still needs a mask per object

Colour and depth are published on separate topics with separate stamps, so depth
frames are paired with the nearest colour frame in time (see PAIR_TOLERANCE_MS).
"""

import csv
import json
import os
import shutil

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

MCAP_PATH = "/home/student/robokudo_foundation_pose/01_0.mcap"
OUT_ROOT = "extracted"

COLOR_TOPIC = "/camera/color/image_raw/compressed"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
TF_TOPIC = "/tf_static"

# a depth frame further than this from its nearest colour frame is dropped
PAIR_TOLERANCE_MS = 40.0


def msg_to_dict(msg):
    """Recursively convert a decoded ROS 2 message to plain Python.

    The mcap_ros2 dynamic classes declare __slots__ but still expose an empty
    __dict__, so vars()/__dict__ silently yields {} - walk __slots__ instead.
    """
    slots = getattr(type(msg), "__slots__", None)
    if slots is not None:
        return {name: msg_to_dict(getattr(msg, name)) for name in slots}
    if isinstance(msg, (bytes, bytearray)):
        return f"<{len(msg)} bytes>"
    if isinstance(msg, np.ndarray):
        return msg.tolist()
    if isinstance(msg, (list, tuple)):
        return [msg_to_dict(x) for x in msg]
    return msg


def stamp_ns(msg, fallback):
    """Sensor timestamp in nanoseconds, falling back to the log time."""
    header = getattr(msg, "header", None)
    if header is None:
        return fallback
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def decode_depth(msg):
    """Decode a depth Image message to a uint16 millimetre array."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "16UC1":
        return buf.view(np.uint16).reshape(msg.height, msg.width)
    if msg.encoding == "32FC1":
        metres = buf.view(np.float32).reshape(msg.height, msg.width)
        return np.nan_to_num(metres * 1000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)
    raise ValueError(f"unsupported depth encoding {msg.encoding!r}")


def decode_color(msg):
    """Decode a colour message to BGR, plus the raw bytes when already JPEG."""
    if hasattr(msg, "format"):
        raw = bytes(msg.data)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode compressed image, format={msg.format!r}")
        is_jpeg = "jpeg" in msg.format.lower() or "jpg" in msg.format.lower()
        return img, (raw if is_jpeg else None)

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "mono8":
        img = cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"unsupported colour encoding {msg.encoding!r}")
    return img, None


def main():
    rgb_dir = os.path.join(OUT_ROOT, "rgb")
    depth_dir = os.path.join(OUT_ROOT, "depth")
    stage_dir = os.path.join(OUT_ROOT, "_color_staging")
    for d in (rgb_dir, depth_dir, stage_dir, os.path.join(OUT_ROOT, "masks")):
        os.makedirs(d, exist_ok=True)

    # ---- pass 1: colour frames and the one-off metadata topics -------------
    color_stamps, color_paths = [], []
    wrote_info = wrote_tf = False

    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        topics = [COLOR_TOPIC, INFO_TOPIC, TF_TOPIC]
        for _, channel, message, msg in reader.iter_decoded_messages(topics=topics):
            if channel.topic == COLOR_TOPIC:
                img, raw_jpeg = decode_color(msg)
                idx = len(color_stamps)
                if raw_jpeg is not None:
                    path = os.path.join(stage_dir, f"{idx:05d}.jpg")
                    with open(path, "wb") as fp:
                        fp.write(raw_jpeg)
                else:
                    path = os.path.join(stage_dir, f"{idx:05d}.png")
                    cv2.imwrite(path, img)
                color_stamps.append(stamp_ns(msg, message.log_time))
                color_paths.append(path)

            elif channel.topic == INFO_TOPIC and not wrote_info:
                K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                np.savetxt(os.path.join(OUT_ROOT, "cam_K.txt"), K, fmt="%.10f")
                with open(os.path.join(OUT_ROOT, "camera_info.json"), "w") as fp:
                    json.dump(msg_to_dict(msg), fp, default=str, indent=2)
                wrote_info = True
                print(f"cam_K.txt written  ({msg.width}x{msg.height}, "
                      f"frame_id={msg.header.frame_id})")
                print(np.array2string(K, precision=3))

            elif channel.topic == TF_TOPIC and not wrote_tf:
                with open(os.path.join(OUT_ROOT, "tf_static.json"), "w") as fp:
                    json.dump(msg_to_dict(msg), fp, default=str, indent=2)
                wrote_tf = True

    if not color_stamps:
        raise RuntimeError(f"no messages on {COLOR_TOPIC}")
    print(f"colour frames: {len(color_stamps)}")

    # ---- pass 2: depth frames, each paired with the nearest colour frame ---
    stamps = np.asarray(color_stamps, dtype=np.int64)
    order = np.argsort(stamps)
    stamps_sorted = stamps[order]

    rows, offsets_ms = [], []
    kept = dropped = 0

    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, message, msg in reader.iter_decoded_messages(topics=[DEPTH_TOPIC]):
            d_ns = stamp_ns(msg, message.log_time)

            pos = int(np.searchsorted(stamps_sorted, d_ns))
            cands = [p for p in (pos - 1, pos) if 0 <= p < len(stamps_sorted)]
            best = min(cands, key=lambda p: abs(int(stamps_sorted[p]) - d_ns))
            c_idx = int(order[best])
            offset_ms = (d_ns - int(stamps[c_idx])) / 1e6

            if abs(offset_ms) > PAIR_TOLERANCE_MS:
                dropped += 1
                continue

            src = color_paths[c_idx]
            dst = os.path.join(rgb_dir, f"{kept:05d}{os.path.splitext(src)[1]}")
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)   # hardlink: no second copy on disk
            except OSError:
                shutil.copyfile(src, dst)

            cv2.imwrite(os.path.join(depth_dir, f"{kept:05d}.png"), decode_depth(msg))

            rows.append([f"{kept:05d}", int(stamps[c_idx]), d_ns, f"{offset_ms:.3f}"])
            offsets_ms.append(abs(offset_ms))
            kept += 1

    with open(os.path.join(OUT_ROOT, "frame_index.csv"), "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "color_stamp_ns", "depth_stamp_ns", "offset_ms"])
        writer.writerows(rows)

    # staged colour frames are hardlinked into rgb/, so the staging dir can go
    shutil.rmtree(stage_dir, ignore_errors=True)

    print(f"paired frames: {kept}  (dropped {dropped} beyond {PAIR_TOLERANCE_MS} ms)")
    if offsets_ms:
        print(f"colour/depth offset: mean {np.mean(offsets_ms):.2f} ms, "
              f"max {np.max(offsets_ms):.2f} ms")
    print(f"\nWrote {OUT_ROOT}/ - masks/ is empty, FoundationPose still needs "
          f"one mask per object to run register().")


if __name__ == "__main__":
    main()
