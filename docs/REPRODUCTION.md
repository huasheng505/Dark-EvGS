# Dark-EvGS 独立复现说明

这是一套根据论文方法和公开数据编写的**独立训练与推理实现**，不是作者未公开训练代码的恢复，也不能据此保证论文中的数值。`dark_evgs/`、`train.py`、`infer.py` 和先验训练脚本构成新的流程；作者原有推理文件保留，方便对照。

## 已核实的来源和复现边界

- 用户提供的论文是 [arXiv:2507.11931v2](https://arxiv.org/abs/2507.11931v2)，2026-04-13 版本；方法见 III 节，训练超参数见 IV-B，噪声滤波消融见表 V。
- [作者仓库](https://github.com/Wujingqian/Dark-EvGS)和 `huasheng505/Dark-EvGS` 在初始核查时的公开 HEAD 均为 `a29a7e784922a7c0ea91743b676ba8b3be602977`。该版本提供渲染、数据及部分检查点，没有完整 Dark-EvGS 训练或 CTCM 实现。
- 论文基于 [EvLowLight](https://github.com/sherrycattt/EvLowLight)，核查版本 `77a7f63420b69b4aaeca5567a576edec0e2ed53e`。它公开了网络、推理与 Git LFS 预训练权重；仓库虽有 BasicSR 训练入口，README 仍将完整训练代码和训练集列为 TODO。它的多帧传播、光流和可变形卷积模块不能直接映射到这里的紧凑先验。
- 本实现的 CTCM 是具备帧/事件编码、融合、CTMB 通道注意力和曝光预测的可训练替代网络。它**不能直接加载 EvLowLight 或作者的 CTCM 权重**；加载检查会拒绝不兼容模型，而不是忽略参数后用随机权重生成结果。
- 作者使用 EvLowLight 初始化，加上随机初始化 CTMB，再用额外采集的 500 组无关场景暗/亮图像和事件训练 5 个 epoch。该辅助数据、微调权重以及完整 CTCM 架构没有随当前仓库提供。独立实现提供了先验训练接口，但这些实验材料仍需自行准备或向作者获取。
- 论文仅给出 8:2 划分，未公开逐帧划分清单。本实现每 5 个按时间排序的视图留出 1 个测试视图（要求时间与文件名顺序一致），划分明确、可重复，但不代表作者的原始索引。

## 环境

CPU 路径用来运行单元测试和小规模合成数据端到端检查。真实场景的 30,000 步训练应使用 NVIDIA GPU、CUDA toolkit 和 CUDA 光栅化扩展。

```bash
cd /workspace/Dark-EvGS
bash scripts/setup_reproduction.sh cpu
source /workspace/.venvs/dark-evgs/bin/activate
python -m pytest -q tests
python scripts/smoke_reproduction.py --workdir runs/smoke_01
```

脚本默认安装 PyTorch 2.5.1、Python 环境当前选择的解释器，以及 `requirements-reproduction.txt` 中的依赖。推荐 Python 3.10–3.12。可以通过 `DARK_EVGS_VENV`、`DARK_EVGS_PYTHON`、`DARK_EVGS_CACHE` 指定解释器和可写路径。不会修改原 `environment.yml`，也不依赖其缺失的 `simple-knn`。

在有 NVIDIA 驱动和完整 CUDA toolkit 的机器上：

```bash
# 默认 PyTorch cu121；本地 nvcc 应与之兼容，建议 CUDA 12.1。
DARK_EVGS_VENV=/workspace/.venvs/dark-evgs-cuda \
  bash scripts/setup_reproduction.sh cuda
source /workspace/.venvs/dark-evgs-cuda/bin/activate
```

如使用 CUDA 11.8，将 `DARK_EVGS_CUDA_VARIANT=cu118` 一并传入。PyTorch CUDA wheel 不包含构建扩展所需的完整 `nvcc`。编译器还须满足所选 NVIDIA toolkit 的要求。脚本保留 TLS 和 Git 校验，在缓存目录检出固定 CUDA 光栅化版本 `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`，并初始化其 GLM 子模块。该版本来自 [原始 3DGS](https://github.com/graphdeco-inria/gaussian-splatting/tree/472689c0dc70417448fb451bf529ae532d32c095) 的固定子模块依赖，接口与本仓库早期光栅化调用一致。编译默认 `MAX_JOBS=4`，可按显存/内存资源调整。CUDA 分支现已在 Colab T4 上完成实际验证，详见下文 Colab 路径。

可选 LPIPS 评估：安装时设置 `DARK_EVGS_LPIPS=1`，或在同一环境安装对应 CUDA/CPU 的 `torchvision==0.20.1` 和 `lpips==0.1.4`。首次 LPIPS 评估还需要下载官方 Torchvision 特征权重；断网环境请事先缓存，不能用随机网络代替。

### 已验证的 Colab 路径

在启用 GPU 的 Colab 终端中，克隆仓库后执行：

```bash
cd /content/Dark-EvGS              # 改成实际克隆位置
DARK_EVGS_LPIPS=1 bash scripts/setup_colab.sh
/content/.venvs/dark-evgs/bin/python -m pytest -q tests
/content/.venvs/dark-evgs/bin/python scripts/smoke_reproduction.py \
  --workdir /content/dark-evgs-smoke --device cuda --backend cuda
```

Colab 镜像可能只有 CUDA 版 PyTorch 而没有 `nvcc`；Python 3.13 也不匹配这里的固定 wheel。`setup_colab.sh` 在单独目录准备 Python 3.12、CUDA 12.6 编译组件和 PyTorch 2.5.1/cu121。自动安装编译器的分支针对 Ubuntu 24.04，不安装或替换 NVIDIA 驱动，默认仅用两个编译进程。旧版扩展在新 CUDA 下缺少间接 `<cstdint>` 头文件的问题通过 `NVCC_PREPEND_FLAGS=--pre-include=cstdint` 修复，固定上游源码不改动。CUDA 12.6 与 wheel 的 12.1 有次版本差异；当前 T4 上已编译并运行验证，不能据此保证其他工具链。

GPU 光栅化反向含原子累加，独立运行不能要求逐参数位级一致。CUDA 测试会精确核验检查点参数、统计量和 Adam 状态恢复；端到端检查核验视图采样顺序及 RNG，并对两条训练路径的测试渲染使用 RMSE `<1e-3`、最大像素误差 `<1e-2` 的阈值。CPU 参考后端仍要求逐参数精确一致。详细实测见 [COLAB_RESULTS.md](COLAB_RESULTS.md)。

## 数据与防止评估泄漏

真实场景结构：

```text
data/lion/
  low_light_frames/*.png
  gt_bright_frames/*.png          # 只用于最终评估
  event.npz                      # x,y,t,p
  timestamps.txt                 # 按帧对应的时间
  sparse/0/{cameras,images}.bin   # 或 output/cameras.json 作为已有位姿来源
```

训练只读取暗图、事件、相机及训练视图的伪亮图，不读取 `gt_bright_frames`。先验训练必须使用不同场景的辅助数据；目标场景名称须加入 manifest 的 `held_out_scenes`，加载先验生成目标场景伪图时会检查这一声明。场景名称检查不能证明数据来源，应自行确认辅助数据不含目标场景或其重复视图。

时间单位与对应关系直接影响事件监督。帧时间和事件时间必须具有相同起点；不能单独减掉两个序列各自的起点。代码仅对明确的绝对时间量级进行单位推断，其他情况请显式传入 `--timestamp-scale` 和 `--event-timestamp-scale`。提供的 lion/baseball/panda 使用微秒时间，通常设置 `1e-6`。代码会检查时间顺序、数组长度、极性编码和像素范围，不会默默裁剪错误坐标或猜测采样比例。

现有样本的已知问题：

| 场景 | 公开样本检查结果 | 所需处理 |
| --- | --- | --- |
| lion、baseball | 帧时间数量和图像一致，有 COLMAP 相机 | 可作为首选真实数据实验 |
| panda | 帧时间数量一致，缺少 sparse 相机，但有 output/cameras.json | 使用公开相机 JSON；位姿来源记录在数据加载器中 |
| house | 200 个时间戳，100 张暗图 | 提供确定的一对一帧时间映射，不能默认隔帧取样 |
| badminton_down | 135 个时间戳，100 张暗图 | 提供确定的一对一帧时间映射 |
| cat | 有 low_light_frames/timestamp.txt；根目录同名文件实际为事件导出；无相机位姿 | 使用帧目录的时间文件，并补充标定/位姿，不能凭空生成相机 |

有歧义时可用无标题行的显式命名时间清单传给 `--timestamps`，每行 `000001.png,1460000000000000`（也可用空白分隔），必须且只能逐一命名所有输入帧。具体格式见 `dark_evgs/data.py`。不应为了通过检查将训练目标与不同时间的事件强行匹配。

先执行 `python inspect_data.py data/lion`，或执行 `python inspect_data.py` 检查全部场景。只要有阻塞项，诊断命令返回非零状态。该检查验证数据完整性及数值范围，不证明事件/图像的像素级标定正确。当前 Lion、Baseball、Panda 通过；Panda 使用公开相机 JSON，但相机轨迹约束弱，初始化范围需要额外检查。

## 1. 训练独立 CTCM 先验

准备来自目标场景之外的辅助数据清单，例如 `auxiliary.json`：

```json
{
  "held_out_scenes": ["lion", "baseball", "house", "panda", "badminton_down", "cat"],
  "train": [{
    "scene": "aux_train_01",
    "dark0": "aux/train/dark0.png",
    "dark1": "aux/train/dark1.png",
    "bright0": "aux/train/bright0.png",
    "bright1": "aux/train/bright1.png",
    "event_voxel": "aux/train/voxel.npy"
  }],
  "val": [{
    "scene": "aux_validation_01",
    "dark0": "aux/val/dark0.png",
    "dark1": "aux/val/dark1.png",
    "bright0": "aux/val/bright0.png",
    "bright1": "aux/val/bright1.png",
    "event_voxel": "aux/val/voxel.npy"
  }]
}
```

路径相对于清单所在目录。每条样本对应同一场景两个时间的配准暗/亮图，以及两时刻之间的有符号事件体素 `event_bins × H × W`。`.npz` 体素使用 `voxel` 键，`.npy` 直接保存数组；通常使用 5 个时间 bin。事件体素化工具在 `dark_evgs/events.py`，先验内部对非零体素作 RMS 归一化。训练和验证 scene 必须不同，且不能出现在 `held_out_scenes` 中。

例如将辅助样本的原始事件转成体素（`t0`、`t1` 必须是该图像对在共同时间轴上的实际秒数）：

```python
import numpy as np
from dark_evgs.events import EventStream
events = EventStream("aux/train/event.npz", timestamp_scale=1e-6)
# t0, t1 = 两张辅助图的实际时间；height, width = 标定后的原始尺寸
voxel = events.voxel(t0, t1, bins=5, height=height, width=width)
np.save("aux/train/voxel.npy", voxel)
```

```bash
python train_prior.py --manifest auxiliary.json --output runs/prior \
  --epochs 5 --device cuda --event-bins 5 --width 32
```

这条命令需要真实辅助数据；合成 smoke 样本只能验证训练管线，不是论文级先验训练。当前伪图生成入口仅接受本实现训练的先验；若要用其他增强器，需要单独实现与记录适配器，不能冒充作者 CTCM。

## 2. 为训练视图生成伪亮图

```bash
python prepare_pseudo.py --source data/lion --output runs/pseudo/lion \
  --checkpoint runs/prior/last.pt --device cuda --holdout 5 \
  --timestamp-scale 1e-6 --event-timestamp-scale 1e-6
```

脚本使用相邻训练视图以及相应事件，不为测试视图生成训练伪图；元数据记录来源、先验和划分。必须使用已训练且格式兼容的检查点，不存在自动使用随机权重的后备路径。

## 3. 训练 3D Gaussian 场景

```bash
python train.py -s data/lion -m runs/lion --pseudo-dir runs/pseudo/lion \
  --iterations 30000 --device cuda --backend cuda --num-points 1000 \
  --holdout 5 --timestamp-scale 1e-6 --event-timestamp-scale 1e-6 \
  --event-contrast 0.2 --event-filter density \
  --filter-window-us 2000 --filter-radius 1 --filter-min-neighbors 2
```

- 位置学习率从 `1.6e-4` 衰减到 `1.6e-6`；特征、透明度、尺度、旋转学习率分别为 `2.5e-3`、`5e-2`、`5e-3`、`1e-3`。默认 1,000 个随机初始点、30,000 步，与论文公开数值对应。
- 三项损失为帧级伪亮图 MSE、事件对数亮度差 MSE，以及渲染差与伪图差的 MSE；后两项权重均为 0.25，对应式 (6)–(9)。`gamma=2.2`、对数偏移 `1e-5`。
- RGB 先 gamma 线性化，再以 `[0.2126,0.7152,0.0722]` 变换为单通道亮度。论文没有指定 RGB 到单色事件的精确规约，这是本实现的显式选择。
- 使用连续可微对数差，不对网络输出施加不可微的硬事件触发阈值。事件目标是对比阈值乘以有符号事件计数；`0.2` 是待校准的复现默认值，论文没有给出传感器的完整阈值标定。
- 密度滤波使用论文描述的时空邻域原理；表 V 中 2,000µs 表现最好。空间半径与密度阈值未公开，默认参数是实现假设，应通过单独验证集校准，而不是根据目标测试集调参。`--event-filter none` 用于无滤波消融。
- 随机初始化体积由训练相机光轴交会和相机尺度确定；论文没有公开随机点的精确采样范围。初始化距离使用 Torch，不依赖缺失的 simple-knn。
- 对前向平移或近乎平行光轴，自动初始化体积可能不可靠。可用 `--init-bounds XMIN YMIN ZMIN XMAX YMAX ZMAX` 指定标定世界坐标中的边界，并用同一边界续训；不要根据测试图成绩反推边界。
- `--backend reference --device cpu` 是便于梯度检查及小数据测试的慢速参考路径，不代表 CUDA 渲染速度，也不能将其 smoke 结果当作真实场景复现。参考实现有 800 万个 Gaussian×pixel 的规模保护；真实图像应使用 CUDA，调试时可减少 `--num-points` 并使用 `--max-width`。

断点续训：

```bash
python train.py -s data/lion -m runs/lion --pseudo-dir runs/pseudo/lion \
  --iterations 30000 --resume runs/lion/checkpoints/latest.pt \
  --device cuda --backend cuda --num-points 1000 --holdout 5 \
  --timestamp-scale 1e-6 --event-timestamp-scale 1e-6 --event-contrast 0.2 \
  --event-filter density --filter-window-us 2000 --filter-radius 1 --filter-min-neighbors 2
```

持续使用相同数据、时间单位和划分参数。检查点及点云写入 `runs/`，不会覆盖公开样本的 `data/*/output`。

## 4. 推理与评估

```bash
python infer.py -m runs/lion --split test --device cuda --backend cuda
python evaluate.py --renders runs/lion/test/ours_30000/renders \
  --ground-truth data/lion/gt_bright_frames --device cuda
# 已安装 LPIPS 和特征权重时另加 --lpips。
```

推理文件名保留原始帧名称，评估按文件名对齐，不能按目录顺序将不相同的视图配对。使用同一测试划分比较所有基线；不要将训练视图误算入测试成绩。评估报告 PSNR、SSIM，以及显式启用的 LPIPS；未运行 LPIPS 不能写成零值或通过。

公开 Gaussian PLY 可通过新入口加载，不依赖公开 `cfg_args` 中过期的绝对路径：

```bash
python infer.py -m data/lion/output -s data/lion --images low_light_frames \
  --split test --iteration 30000 --device cuda --backend cuda
```

它只验证公开模型推理；即使该模型得分良好，也不证明新训练过程达到论文结果。

公开 PLY 的静态推理仅使用图像与相机，不读取事件或不可靠的帧时间；因此 House/Badminton 的同步问题不会阻止已有模型的静态渲染。训练仍要求完整同步。新训练模型的推理会保持其已保存的时间/测试划分。

## 当前验证与下一步实验

初始 Codex 环境的 CPU 验证为 **28 passed**。后续已连接用户的 Colab T4，完成 CUDA 构建和 **31 passed**，并运行真实场景先验训练、作者检查点 GPU 推理及 Lion 独立训练，包含 PSNR/SSIM/LPIPS 评估。更新后的范围和实测值见 [COLAB_RESULTS.md](COLAB_RESULTS.md)。

合成端到端结果只验证软件行为。测试先验仅训练一个小样本、一轮，高斯只训练六步，不能据此评价真实重建质量。详细验证范围见 [VALIDATION.md](VALIDATION.md)。

要判断是否达到论文结果，需要固定并公开：辅助先验数据来源和检查点、目标场景帧/事件时间标定、相机位姿、训练/测试文件清单、随机种子、事件阈值和滤波设置、GPU/工具链、每场景训练日志、30,000 步模型及逐图 PSNR/SSIM/LPIPS。建议先对 lion 跑通短程 CUDA 训练和梯度，再执行完整训练；缺失或歧义的数据先补齐。不要在未完成这些实验时声称已复现论文中的 31.22dB 或其他数值。
