# autonomous-driving-platform

ROS2 workspace와 ML 실험 디렉토리를 분리해 운영하는 저장소입니다.

## Repository Structure

autonomous-driving-platform/
├── src/
│   ├── neuro_decision/
│   ├── vehicle_serial_bridge/
│   └── yolopv2_ros/
├── perception/
│   └── lane_detection/
├── docs/
├── scripts/
├── configs/
├── models/
├── README.md
└── .gitignore

## Workspace Layout

- ROS2 패키지는 모두 src/ 아래에 위치합니다.
	- src/neuro_decision
	- src/vehicle_serial_bridge
	- src/yolopv2_ros
- perception/lane_detection은 ROS2 패키지가 아닌 ML/데이터 작업 디렉토리입니다.
- docs/는 개발/운영 문서(예: MCU 패치 가이드)를 보관합니다.

## Package Roles

- neuro_decision: planning/control 계열 ROS2 Python 패키지
- vehicle_serial_bridge: cmd_vel/estop을 MCU serial 단일 문자 명령으로 변환하는 브리지 패키지
- yolopv2_ros: perception 계열 ROS2 Python 패키지

## Build and Run

저장소 루트에서 colcon을 실행합니다.

colcon build --packages-select neuro_decision vehicle_serial_bridge yolopv2_ros
source install/setup.bash

## End-to-End Bringup Safety

- `src/autonomous_bringup` contains the integrated bringup launch files.
- Use `ros2 launch autonomous_bringup e2e_mock.launch.py mock_serial:=true` for normal perception -> planning -> `/cmd_vel` -> mock MCU checks.
- `e2e_hardware_safe.launch.py` is for real MCU preparation only. It defaults to `enable_vehicle_bridge:=false` and `mock_serial:=true`.
- Real serial requires the explicit confirmation arguments documented in `src/autonomous_bringup/README.md`.
- Before connecting hardware, run `scripts/e2e_preflight_check.sh` and follow `docs/hardware_preflight_checklist.md`.
- CPU-only YOLOPv2 inference may require `lane_timeout_s:=3.0` or `4.0` for bench inspection; reduce the timeout again after improving inference FPS for real driving.

## Artifact Policy

- build/, install/, log/는 colcon 생성 산출물이며 Git에 포함하지 않습니다.
- Python 캐시 및 테스트 캐시(__pycache__, *.pyc, *.egg-info, .pytest_cache, .mypy_cache, .ruff_cache)도 Git에 포함하지 않습니다.

## Notes

- 모델/추론 아티팩트는 .gitignore 정책에 따라 기본적으로 제외됩니다.
- 루트의 모델 파일 경로를 코드가 참조할 수 있으므로, 가중치 파일 이동은 참조 경로를 점검한 뒤 진행하세요.
