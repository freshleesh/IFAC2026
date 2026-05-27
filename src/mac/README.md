# VESC Driver - macOS Setup Guide (ROS 2 Jazzy)

macOS (Apple Silicon)에서 conda/robostack을 이용한 VESC 드라이버 빌드 가이드.

> **참고:** macOS robostack ROS 2는 시각화/개발 용도에 적합합니다. 프로덕션 하드웨어 제어는 Linux(RPi/Jetson) 환경을 권장합니다.

## 사전 요구사항

- macOS (Apple Silicon)
- [Miniforge](https://github.com/conda-forge/miniforge) 또는 conda 설치
- VESC 하드웨어 (USB 연결)

## 1. Conda 환경 생성

```bash
conda create -n jazzy python=3.12 -y
conda activate jazzy
conda config --env --add channels robostack-staging
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```

## 2. ROS 2 및 의존성 설치

```bash
# ROS 2 Jazzy Desktop
conda install ros-jazzy-desktop -y

# conda에서 제공하는 추가 의존성
conda install ros-jazzy-ackermann-msgs -y
```

## 3. conda에 없는 의존성 소스 클론

`serial_driver`, `udp_msgs`는 macOS용 conda 패키지가 없으므로 소스에서 빌드합니다.

```bash
cd ~/jazzy/src
git clone https://github.com/ros-drivers/transport_drivers.git
git clone https://github.com/flynneva/udp_msgs.git
```

## 4. Standalone ASIO 헤더 설치

conda의 `asio` 패키지는 standalone 헤더를 포함하지 않으므로 직접 복사합니다.

```bash
cd /tmp
git clone --depth 1 --branch asio-1-30-2 https://github.com/chriskohlhoff/asio.git
cp /tmp/asio/asio/include/asio.hpp ~/conda/envs/jazzy/include/
cp -r /tmp/asio/asio/include/asio ~/conda/envs/jazzy/include/
```

## 5. VESC 소스 코드 수정 (macOS 호환)

### `experimental/optional` -> `optional`

macOS clang은 `<experimental/optional>`을 지원하지 않습니다.

**`vesc_driver/include/vesc_driver/vesc_driver.hpp`:**
```cpp
// 변경 전
#include <experimental/optional>
std::experimental::optional<double>

// 변경 후
#include <optional>
std::optional<double>
```

**`vesc_driver/src/vesc_driver.cpp`:**
```cpp
// 변경 전
const std::experimental::optional<double> & min_lower,
const std::experimental::optional<double> & max_upper)

// 변경 후
const std::optional<double> & min_lower,
const std::optional<double> & max_upper)
```

### FlowControl 변경

VESC USB CDC는 하드웨어 흐름 제어를 지원하지 않습니다.

**`vesc_driver/src/vesc_interface.cpp`:**
```cpp
// 변경 전
auto fc = drivers::serial_driver::FlowControl::HARDWARE;

// 변경 후
auto fc = drivers::serial_driver::FlowControl::NONE;
```

## 6. 컴파일러 오버라이드 (macOS 26+ Xcode SDK 호환)

macOS 26(Tahoe) 이상에서는 Xcode SDK의 시스템 라이브러리 스텁(`.tbd`) 파일이 `arm64e` 아키텍처만 선언합니다.
conda의 크로스 컴파일러(`arm64-apple-darwin20.0.0-clang`)는 `arm64` 타겟으로 링크를 요청하기 때문에,
링커가 `___stack_chk_fail`, `___assert_rtn` 등 기본 C 심볼을 찾지 못해 빌드가 실패합니다.

시스템 clang(`/usr/bin/clang`)은 `arm64e`로 빌드하므로 이 문제가 없습니다. 아래 스크립트로 `conda activate` 시 자동으로 시스템 컴파일러를 사용하도록 설정합니다.

```bash
# activate 시 시스템 clang 사용
cat > $CONDA_PREFIX/etc/conda/activate.d/z_override_compiler.sh << 'EOF'
#!/bin/bash
export CONDA_BACKUP_CC_OVERRIDE="$CC"
export CONDA_BACKUP_CXX_OVERRIDE="$CXX"
export CONDA_BACKUP_OBJC_OVERRIDE="$OBJC"
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export OBJC=/usr/bin/clang
EOF

# deactivate 시 원래 값 복원
cat > $CONDA_PREFIX/etc/conda/deactivate.d/z_override_compiler.sh << 'EOF'
#!/bin/bash
[ -n "$CONDA_BACKUP_CC_OVERRIDE" ] && export CC="$CONDA_BACKUP_CC_OVERRIDE" && unset CONDA_BACKUP_CC_OVERRIDE
[ -n "$CONDA_BACKUP_CXX_OVERRIDE" ] && export CXX="$CONDA_BACKUP_CXX_OVERRIDE" && unset CONDA_BACKUP_CXX_OVERRIDE
[ -n "$CONDA_BACKUP_OBJC_OVERRIDE" ] && export OBJC="$CONDA_BACKUP_OBJC_OVERRIDE" && unset CONDA_BACKUP_OBJC_OVERRIDE
EOF

# 적용을 위해 환경 재활성화
conda deactivate && conda activate jazzy
```

> **참고:** conda-forge에서 새 Xcode SDK에 대응하는 컴파일러 업데이트가 나오면 이 단계는 불필요해집니다.

## 7. 빌드

빌드는 2단계로 진행합니다. 1차 빌드 후 `source`를 해야 `serial_driver` 등의 경로가 잡힙니다.

```bash
cd ~/jazzy

# 1차 빌드
colcon build

# install 경로 반영
source install/setup.zsh

# 2차 빌드 (vesc_driver가 serial_driver를 찾을 수 있도록)
# vesc_driver 빌드 캐시 삭제 필요
rm -rf build/vesc_driver
colcon build
```

## 8. 실행

```bash
source install/setup.zsh
ros2 launch vesc_driver vesc_driver_node.launch.py
```

### VESC 시리얼 포트 설정

`vesc_driver/params/vesc_config.yaml`에서 포트를 확인/수정합니다:

```yaml
/**:
  ros__parameters:
    port: "/dev/tty.usbmodem3041"  # ls /dev/tty.usb* 로 실제 포트 확인
```

## 알려진 제한사항

| 항목 | 설명 |
|------|------|
| `install_name_tool` warning | macOS dylib 서명 경고. 무시 가능 |
| `CMake Deprecation Warning` | cmake_minimum_required 버전 경고. 무시 가능 |
| `thread affinity` error | macOS에서 DDS thread affinity 미지원. 무시 가능 |
| `ros2 topic list` killed | FastDDS shared memory 문제. `export FASTDDS_BUILTIN_TRANSPORTS=UDPv4` 시도 |

## 디렉토리 구조

```
~/jazzy/src/
  vesc/                     # VESC 패키지 (vesc_driver, vesc_msgs, vesc_ackermann)
  transport_drivers/        # serial_driver 포함 (소스 클론)
  udp_msgs/                 # udp_msgs (소스 클론)
```
