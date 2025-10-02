# CRANE-X7

未来ロボティクス学科 ロボット設計制作論実習3

## 概要

VLAを使ってマニュピレータを制御する。

## 必須

- Native Linux or WSL
- Docker

## 実行

1. .envの作成

.env.templateからコピーして作成

各環境変数の説明
- `USB_DEVICE` ホストに認識されているのUSBデバイスのパス

2. 実行

実機の場合
```bash
docker compose --profile real up
```

シミュレータ(Gazebo)の場合
```bash
docker compose --profile sim up
```

## 開発

### Docker

1. ビルド

```bash
./scripts/build.sh
```

2. 実行

実機で実行
```bash
./scripts/run.sh real

# colcon build --symlink-install
# source install/setup.bash
```

シミュレータ(Gazebo)で実行
```bash
./scripts/run.sh sim

# colcon build --symlink-install
# source install/setup.bash
```

## 参考情報

### RT

- https://github.com/rt-net/crane_x7
- https://github.com/rt-net/crane_x7_ros
- https://github.com/rt-net/crane_x7_Hardware
- https://github.com/rt-net/crane_x7_samples

