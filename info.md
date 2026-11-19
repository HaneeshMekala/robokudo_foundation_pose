
## Live Noctis Annotator + Foundation Pose Startup

**Terminal 1 — Camera**
```bash
workon cram-env
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py \
  depth_registration:=true \
  align_mode:=SW \
  align_target_stream:=COLOR
```

**Terminal 2 — RoboKudo + NOCTIS + FoundationPose**
```bash
workon robokudo_noctis
source ~/ros2_ws/install/setup.bash
ros2 run robokudo_ros main \
  _ae=demo_tracy_cubes_live_noctis \
  _ros_pkg=robokudo_foundation_pose
```

**Only after package code changes — Build**
```bash
cd ~/ros2_ws
colcon build --merge-install --packages-select robokudo_foundation_pose
```
