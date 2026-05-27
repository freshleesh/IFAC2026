# vesc_driver_mac

macOS launch configuration and teleop for VESC-based Ackermann vehicles. Designed for F1TENTH cars driven from a Mac.

Provides a single launch file that brings up the full VESC stack (driver + Ackermann conversion + odometry) and a teleop node.

## Tested Environment

| Item | Version |
|---|---|
| Hardware | MacBook Air M4 |
| OS | macOS 26.4 Tahoe |
| ROS2 | Jazzy (conda) |
| Python | 3.12 |
| VESC | VESC 6 (F1TENTH) |

## Dependencies

This package requires the following ROS2 packages in your workspace:

- [vesc](https://github.com/f1tenth/vesc) — VESC driver and Ackermann nodes

## Nodes

### teleop_joy

Converts speed/steering input to `/ackermann_cmd` for VESC Ackermann control.

| Subscribed Topic | Type | Description |
|---|---|---|
| `/ackermann_cmd` | `ackermann_msgs/AckermannDriveStamped` | Speed and steering command |

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_speed` | 2.0 | Maximum speed in m/s |
| `max_steering_angle` | 0.34 | Maximum steering angle in radians |

## Launch

The launch file starts 4 nodes:

| Node | Package | Description |
|---|---|---|
| vesc_driver_node | vesc_driver | Serial communication with VESC |
| ackermann_to_vesc_node | vesc_ackermann | Ackermann cmd → motor speed + servo |
| vesc_to_odom_node | vesc_ackermann | VESC state → odometry + TF |
| teleop_joy_node | vesc_driver_mac | Teleop → Ackermann cmd |

## Topic Flow

```
/ackermann_cmd
    ↓  (ackermann_to_vesc)
/commands/motor/speed + /commands/servo/position
    ↓  (vesc_driver)
VESC Hardware → /sensors/core → (vesc_to_odom) → /odom + /tf
```

## Configuration

Edit `config/vesc_config.yaml` before launching:

```yaml
# Serial port — check with: ls /dev/tty.usbmodem*
port: "/dev/tty.usbmodem3041"

# Speed limits in eRPM
speed_max: 23250.0
speed_min: -23250.0

# Servo limits (0.0 ~ 1.0)
servo_max: 0.85
servo_min: 0.15

# Ackermann conversion gains
speed_to_erpm_gain: 4614.0
steering_angle_to_servo_gain: -1.2135
steering_angle_to_servo_offset: 0.5304
```

## Install

### 1. Conda environment setup

```bash
conda create -n jazzy python=3.12 -y
conda activate jazzy
conda config --env --add channels robostack-staging
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

conda install ros-jazzy-desktop -y
conda install ros-jazzy-ackermann-msgs -y
```

### 2. Standalone ASIO headers

conda's `asio` package does not include standalone headers required by `serial_driver`. Copy them manually:

```bash
cd /tmp
git clone --depth 1 --branch asio-1-30-2 https://github.com/chriskohlhoff/asio.git
cp /tmp/asio/asio/include/asio.hpp ~/conda/envs/jazzy/include/
cp -r /tmp/asio/asio/include/asio ~/conda/envs/jazzy/include/
```

### 3. Clone and build

```bash
cd ~/jazzy/src
git clone https://github.com/jeongsang-ryu/vesc_mac.git
cd ~/jazzy
colcon build --cmake-args \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3
source install/setup.zsh
```

> **Note:** macOS updates can break conda's bundled cross-compiler (`arm64-apple-darwin20.0.0-clang`). The flags above force the system compiler and conda Python, which avoids SDK mismatch issues.

### macOS source modifications (already applied)

The following changes are already applied in the included `vesc` source:

- `vesc_driver`: `#include <experimental/optional>` → `#include <optional>`, `std::experimental::optional` → `std::optional`
- `vesc_driver`: `FlowControl::HARDWARE` → `FlowControl::NONE` (VESC USB CDC does not support hardware flow control)

## Usage

```bash
ros2 launch vesc_driver_mac vesc_all.launch.xml
```

Or with a custom config:

```bash
ros2 launch vesc_driver_mac vesc_all.launch.xml config:=/path/to/your_config.yaml
```
