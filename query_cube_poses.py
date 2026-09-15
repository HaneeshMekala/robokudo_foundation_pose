"""Query the live pipeline for cube poses and check them against the table.

Sends a 'robokudo_msgs/action/Query' goal to '/robokudo/query' - exactly what the
planning team's client does - prints the 'PoseStamped' that comes back for each
object, and then checks it for physical sense:

  * position in the 'table' frame, whose origin lies on the tabletop, so its z
    is the object's height above the table;
  * the height the object's centre *should* have if it rests on the table,
    worked out from its mesh and the estimated orientation - the difference is
    the estimator's vertical error;
  * which mesh axis points up, how far it is tilted, and the object's yaw.

Run it with the pipeline up and QUERY_DRIVEN = True in demo_tracy_cubes_live.py:

    source env_fp_tracy.sh
    python query_cube_poses.py               # one query
    python query_cube_poses.py --repeat 10   # repeatability over 10 queries

Put a cube flat on the table, measure where it is from a known point, and
compare with the 'table' frame numbers.
"""

from __future__ import annotations

import argparse
import math
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener

from robokudo_msgs.action import Query


ROOT = osp.dirname(osp.abspath(__file__))


def pose_to_matrix(position, orientation) -> np.ndarray:
    """4 x 4 transform from a position and an (x, y, z, w) quaternion."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]).as_matrix()
    matrix[:3, 3] = [position.x, position.y, position.z]
    return matrix


def resting_height(rotation: np.ndarray, extents: np.ndarray) -> float:
    """Height of the mesh centre above the surface it would rest on.

    Half the vertical size of the object's box once rotated: each mesh axis
    contributes its extent times how much of it points up.  For an object lying
    on one of its faces this is simply half that face's thickness.
    """
    return 0.5 * float(np.abs(rotation[2, :]) @ extents)


class CubePoseQuery(Node):
    def __init__(self, action: str, table_frame: str, mesh_dir: str):
        super().__init__("query_cube_poses")
        self.table_frame = table_frame
        self.mesh_dir = mesh_dir
        self.client = ActionClient(self, Query, action)
        self.tf_buffer = Buffer()
        # No 'spin_thread': that would spin this same node from a second executor
        # while the action calls spin it from the main thread.  TF data arrives
        # whenever the node is spun instead - during the query, and in
        # 'in_table_frame' when it is still missing.
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._extents = {}

    def mesh_extents(self, classname: str):
        """Mesh-frame extents of the object's CAD model, or None if not found.

        Read from the '.ply' itself rather than copied from the AE, so the check
        follows the mesh if it is ever replaced.
        """
        if classname not in self._extents:
            path = osp.join(self.mesh_dir, f"{classname}.ply")
            if osp.exists(path):
                import trimesh
                mesh = trimesh.load(path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                self._extents[classname] = np.asarray(mesh.extents, dtype=float)
            else:
                self._extents[classname] = None
        return self._extents[classname]

    def query(self, timeout: float) -> Query.Result:
        goal = Query.Goal()     # the pipeline does not filter by goal contents
        send = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=timeout)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("the query was not accepted")

        result = handle.get_result_async()
        try:
            rclpy.spin_until_future_complete(self, result, timeout_sec=timeout)
        finally:
            if not result.done():
                # never leave a goal behind on the server - on timeout or Ctrl+C
                # the pipeline would otherwise still be holding it
                rclpy.spin_until_future_complete(self, handle.cancel_goal_async(), timeout_sec=2.0)

        if result.result() is None:
            raise RuntimeError(f"no answer within {timeout:.0f} s - the action server exists but "
                               "nothing is consuming goals; is QUERY_DRIVEN = True?")
        return result.result().result

    def in_table_frame(self, pose_stamped) -> np.ndarray:
        """The object's pose as a 4 x 4 matrix in the table frame."""
        frame = pose_stamped.header.frame_id
        deadline = time.time() + 3.0
        while not self.tf_buffer.can_transform(self.table_frame, frame, rclpy.time.Time()) \
                and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        tf = self.tf_buffer.lookup_transform(self.table_frame, frame, rclpy.time.Time())
        table_from_frame = pose_to_matrix(tf.transform.translation, tf.transform.rotation)
        frame_from_obj = pose_to_matrix(pose_stamped.pose.position, pose_stamped.pose.orientation)
        return table_from_frame @ frame_from_obj

    def report(self, designator, samples) -> None:
        name = designator.type or "?"
        print(f"\n  {name}")

        if not designator.pose:
            print("    detected, but no pose - FoundationPose produced nothing for it")
            return

        ps = designator.pose[0]
        p, q = ps.pose.position, ps.pose.orientation
        source = designator.pose_source[0] if designator.pose_source else "?"

        # exactly what the planning team receives
        print(f"    frame_id           : {ps.header.frame_id}   (stamp {ps.header.stamp.sec} s, source {source})")
        print(f"    position           : x={p.x:+.4f}  y={p.y:+.4f}  z={p.z:+.4f}  m")
        print(f"    orientation (xyzw) : [{q.x:+.4f} {q.y:+.4f} {q.z:+.4f} {q.w:+.4f}]")

        try:
            pose = self.in_table_frame(ps)
        except Exception as error:   # TF missing - still show the raw answer
            print(f"    no transform to '{self.table_frame}': {type(error).__name__}")
            return

        rotation, t = pose[:3, :3], pose[:3, 3]
        print(f"    in '{self.table_frame}' frame   : x={t[0]:+.4f}  y={t[1]:+.4f}  height={t[2]:+.4f}  m")

        # which mesh axis points up, and how far from vertical it is
        up = int(np.argmax(np.abs(rotation[2, :])))
        tilt = math.degrees(math.acos(min(1.0, abs(rotation[2, up]))))
        print(f"    upward mesh axis   : {'xyz'[up]}   (tilted {tilt:.1f} deg from vertical)")

        extents = self.mesh_extents(name)
        yaw = None
        if extents is not None:
            expected = resting_height(rotation, extents)
            print(f"    expected height    : {expected:+.4f} m if resting on the table"
                  f"   -> vertical error {1000 * (t[2] - expected):+.1f} mm")

            # Yaw of the longest horizontal mesh axis, for comparing with a tape
            # measure.  An axis has no direction on the table, so it is folded
            # modulo 180 deg - or 90 deg when the footprint is square and "longest"
            # is arbitrary.  The quaternion above is untouched; the planners get
            # the full orientation.
            horizontal = [i for i in range(3) if i != up]
            long_axis = max(horizontal, key=lambda i: extents[i])
            square = abs(extents[horizontal[0]] - extents[horizontal[1]]) < 0.005
            period = 90.0 if square else 180.0
            yaw = math.degrees(math.atan2(rotation[1, long_axis], rotation[0, long_axis]))
            yaw = (yaw + period / 2) % period - period / 2
            footprint = "square footprint, " if square else ""
            print(f"    yaw of long axis   : {yaw:+.1f} deg   (mesh {'xyz'[long_axis]},"
                  f" {100 * extents[long_axis]:.1f} cm; {footprint}folded modulo {period:.0f} deg)")

        samples[name].append((*t, yaw if yaw is not None else float("nan")))


def summarise(samples) -> None:
    print("\nrepeatability (table frame)")
    for name, rows in samples.items():
        data = np.asarray(rows, dtype=float)
        if len(data) < 2:
            continue
        spread = data.max(axis=0) - data.min(axis=0)
        std = data.std(axis=0)
        print(f"  {name}  over {len(data)} queries")
        for i, axis in enumerate(("x", "y", "height")):
            print(f"    {axis:<6} mean {data[:, i].mean():+.4f} m   std {1000 * std[i]:5.1f} mm"
                  f"   max-min {1000 * spread[i]:5.1f} mm")
        if not np.isnan(data[:, 3]).all():
            print(f"    yaw    mean {np.nanmean(data[:, 3]):+.1f} deg  std {np.nanstd(data[:, 3]):.2f} deg")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1, help="number of queries to send")
    parser.add_argument("--action", default="/robokudo/query", help="query action name")
    parser.add_argument("--table-frame", default="table",
                        help="frame whose origin is on the tabletop (default: table)")
    parser.add_argument("--mesh-dir", default=ROOT, help="where '<classname>.ply' files live")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds to wait per query")
    args = parser.parse_args()

    rclpy.init()
    node = CubePoseQuery(args.action, args.table_frame, args.mesh_dir)
    try:
        if not node.client.wait_for_server(timeout_sec=5.0):
            raise SystemExit(f"'{args.action}' is not available - is the pipeline running?")

        samples = defaultdict(list)
        for i in range(args.repeat):
            began = time.time()
            result = node.query(args.timeout)
            print(f"\nquery {i + 1}/{args.repeat}: {len(result.res)} object(s) in {time.time() - began:.1f} s")
            for designator in result.res:
                node.report(designator, samples)

        if args.repeat > 1:
            summarise(samples)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("\ninterrupted")
    except RuntimeError as error:
        raise SystemExit(f"query failed: {error}")
    finally:
        node.tf_listener.unregister()
        node.destroy_node()
        # Ctrl+C already shuts the context down; a second shutdown() would raise
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
