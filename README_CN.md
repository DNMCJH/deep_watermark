# Deep Watermark — AIGC 图像版权保护深度水印系统

基于深度学习的图像水印系统，支持端到端训练的水印嵌入与提取，目标是实现对屏摄攻击的鲁棒性。

## 系统概述

```
原始图像 + 水印比特 → 编码器(Encoder) → 水印图像 → 攻击层(Attack) → 解码器(Decoder) → 恢复水印
```

当前阶段：基线系统（简单攻击：噪声 / 模糊 / JPEG压缩 / 缩放 / 裁剪）

目标指标：PSNR > 35 dB，Bit Accuracy > 80%

---

## 项目路线图

### 第一阶段：环境搭建与基线训练（第 1-2 周）

- [x] 搭建项目框架，创建模块化代码结构
- [x] 实现 Encoder / Decoder / AttackLayer 基线模型
- [x] 实现训练管线（损失函数、训练循环、checkpoint 保存）
- [x] 下载 DIV2K 数据集，完成 CPU 端到端验证
- [ ] 在 GPU 上完成 100 epoch 正式训练
- [ ] 达到目标指标：PSNR > 35，Bit Accuracy > 80%

### 第二阶段：模型优化与攻击增强（第 3-4 周）

- [ ] 分析基线模型的弱点（哪些攻击下 BitAcc 最低）
- [ ] 增强 Encoder 架构（更深的网络 / U-Net 结构 / 注意力机制）
- [ ] 增强 Decoder 鲁棒性（多尺度特征提取）
- [ ] 引入 LPIPS 感知损失，提升水印图像视觉质量
- [ ] 扩展攻击层：组合攻击、更强的 JPEG 压缩、颜色抖动

### 第三阶段：屏摄攻击建模（第 5-8 周）

- [ ] 实现屏摄攻击模拟（透视变换、摩尔纹、色偏、光照）
- [ ] 引入 STN（空间变换网络）做几何校正
- [ ] 参考 PIMoG 等方法设计屏摄鲁棒水印方案
- [ ] 在真实屏摄数据上测试与微调

### 第四阶段：实验与论文（第 9-12 周）

- [ ] 系统性消融实验（攻击类型、水印长度、λ 权重等）
- [ ] 与现有方法对比（HiDDeN、RivaGAN、MBRS 等）
- [ ] 整理实验结果，撰写论文

---

## PIMoG 屏摄噪声层集成方案

项目已获取 PIMoG 论文的开源实现，其核心价值是 `Noise_Layer.py` 中的 `ScreenShooting` 类，完整模拟了屏摄退化过程：

| 模拟组件 | 实现方式 | 权重 |
|----------|----------|------|
| 透视变换 | 四角点随机扰动 ±2px | - |
| 光照畸变 | 线性渐变 + 径向渐变 | 85% |
| 摩尔纹干扰 | 圆形 + 线性干涉图案 | 15% |
| 高斯噪声 | σ=0.001 | 叠加 |

### 集成路径

1. **第二阶段**：李将 PIMoG 噪声层适配到 `models/attack_layer.py`（需调整分辨率 128→256、值域 [-1,1]→[0,1]）
2. **第二阶段**：参考 PIMoG 的 U-Net Encoder 和 ResNet Decoder 升级模型架构
3. **第三阶段**：用完整屏摄噪声层做正式的鲁棒性训练

> PIMoG 的训练循环和数据加载器不需要搬，我们已有更清晰的实现。

---

## 团队分工

| 成员 | 核心负责 | 当前阶段任务 |
|------|----------|-------------|
| 陈 | 实验与系统、水印嵌入 | Push 代码、GPU 正式训练、Loss 调参、系统集成 |
| 李 | 攻击建模 | 研究 PIMoG Noise_Layer.py，准备移植屏摄噪声层 |
| 黄 | 水印提取 | 在当前 Decoder 基础上研究 ResNet 增强 + STN |
| 贾 | 数据管理 | 用 Stable Diffusion 生成 AIGC 测试图 |
| 龚 | 水印嵌入、UI | 研究 LPIPS 损失集成 + Streamlit UI 设计 |

---

## 环境配置

### 1. 创建虚拟环境

```bash
python -m venv watermark_env
# Windows
watermark_env\Scripts\activate
# Linux / Mac
source watermark_env/bin/activate
```

### 2. 安装依赖

仅 CPU（无 GPU 的电脑）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

CUDA 11.8（有 NVIDIA GPU）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> 代码会自动检测 GPU，CPU 和 CUDA 环境使用同一套代码，无需修改。

### 3. 准备数据集

自动下载 DIV2K（800 张高清图，约 3.3GB）：
```bash
python scripts/download_div2k.py
```

或手动将图片放入 `dataset/train/` 目录。详见 [dataset/README.md](dataset/README.md)。

### 4. 验证框架

运行可视化验证脚本，确认整个 pipeline 正常工作：

```bash
python scripts/verify_pipeline.py
```

会在 `assets/verify/` 下生成对比图，每张包含 6 个面板：

| 面板 | 含义 |
|------|------|
| Original | 原始图片 |
| Watermarked | 嵌入水印后（附 PSNR/SSIM 指标） |
| Residual (x10) | 水印残差放大 10 倍，展示模型修改了哪些像素 |
| + Noise | 加高斯噪声后的效果 |
| + Blur | 加模糊后的效果 |
| + Crop | 裁剪后的效果 |

> 未训练的模型 BitAcc 约 50%（随机猜），训练后会提升到 80%+。这个脚本的目的是验证数据流通路正确，不是验证模型性能。

---

## 完整操作流程

### Step 1：训练模型

```bash
cd deep_watermark
python -m train.train --config configs/train.yaml
```

训练过程会自动：
- 创建实验目录 `experiments/<时间戳>/`
- 保存训练配置 `config.yaml`
- 保存指标记录 `metrics.json`
- 保存最佳模型 `checkpoint.pt`
- 更新全局实验日志 `experiments/experiment_log.md`

如需调整超参数，编辑 `configs/train.yaml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 16 | 批大小（CPU 建议改为 4） |
| num_epochs | 100 | 训练轮数 |
| learning_rate | 0.001 | 学习率 |
| lambda_watermark | 5.0 | 水印损失权重 |
| image_size | 256 | 输入图像尺寸 |
| watermark_length | 32 | 水印比特数 |
| attack_prob | 0.8 | 训练时施加攻击的概率 |

### Step 2：评估模型

```bash
python -m eval.evaluate --checkpoint experiments/<实验ID>/checkpoint.pt --test_dir dataset/test
```

输出指标：
- PSNR（峰值信噪比）：衡量水印图像质量，越高越好
- SSIM（结构相似性）：衡量结构保真度
- Bit Accuracy（比特准确率）：水印提取正确率
- BER（误码率）：= 1 - Bit Accuracy

### Step 3：交互式 Demo

```bash
streamlit run demo/streamlit_app.py
```

上传图片 → 输入水印比特 → 查看嵌入效果和提取结果。

---

## 项目结构

```
deep_watermark/
├── configs/
│   └── train.yaml              # 训练超参数配置
├── models/                     # 核心模型
│   ├── encoder.py              # 水印编码器（残差学习）
│   ├── decoder.py              # 水印解码器（CNN + 全连接）
│   └── attack_layer.py         # 可微攻击层（噪声/模糊/JPEG/缩放/裁剪）
├── data/                       # 数据管线
│   ├── dataset.py              # 图片数据集加载器
│   └── watermark_generator.py  # 随机水印生成
├── train/                      # 训练系统
│   ├── train.py                # 训练入口
│   ├── trainer.py              # 训练循环与 checkpoint 管理
│   └── loss.py                 # 损失函数：MSE + λ·BCE
├── eval/                       # 评估
│   ├── metrics.py              # PSNR / SSIM / BitAcc / BER
│   └── evaluate.py             # 评估脚本
├── utils/                      # 工具函数
│   ├── image_utils.py          # 图像读写与可视化
│   └── logging_utils.py        # 实验追踪
├── demo/
│   └── streamlit_app.py        # Streamlit 交互 Demo
├── scripts/
│   └── download_div2k.py       # 数据集下载脚本
├── dataset/                    # 训练/测试图片
├── experiments/                # 实验记录（自动生成）
└── assets/                     # 素材文件
```

---

## Git 协作规范

### 组员快速上手

```bash
# 1. 克隆仓库
git clone https://github.com/DNMCJH/deep_watermark.git
cd deep_watermark

# 2. 创建虚拟环境并安装依赖（见上方"环境配置"）

# 3. 下载数据集
python scripts/download_div2k.py

# 4. 验证框架
python scripts/verify_pipeline.py
# 查看 assets/verify/ 下的对比图，确认 pipeline 正常
```

### 提交代码流程

```bash
# 日常开发在 dev 分支
git checkout -b dev
git push -u origin dev

# 新功能开 feature 分支
git checkout -b feature/screen-camera-attack
# ... 开发完成后
git add .
git commit -m "feat: add screen camera attack simulation"
git push origin feature/screen-camera-attack
# 在 GitHub 上发起 Pull Request 合并到 dev
```

分支结构：
- `main` — 稳定版本
- `dev` — 日常开发
- `feature/*` — 功能分支

### 建议分工

路线图中的任务可以按模块分配给不同组员：

| 方向 | 工作内容 | 阶段 |
|------|----------|------|
| 训练与调参 | GPU 正式训练、超参数搜索、学习率调度 | 第一~二阶段 |
| 模型改进 | Encoder/Decoder 架构升级（U-Net、注意力等） | 第二阶段 |
| 攻击层扩展 | 新增攻击类型、组合攻击、屏摄模拟 | 第二~三阶段 |
| 评估与实验 | 消融实验、对比实验、结果可视化 | 全程 |
| 论文撰写 | 相关工作调研、方法描述、实验分析 | 第三~四阶段 |

> 每次实验结果会自动记录在 `experiments/experiment_log.md`，方便组员之间对比不同实验。

---

## 协议

见 [LICENSE](LICENSE)。
