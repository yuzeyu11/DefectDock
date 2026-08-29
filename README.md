# DefectDock

> Build, verify, and ship industrial vision models.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-149ECA.svg)](apps/web)
[![API](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey.svg)](LICENSE)

DefectDock 是面向工业视觉项目的模型全生命周期工作台。它把数据接入、标注协同、质量检查、训练、工业指标验收、模型激活和现场视频流推理放在一条可审计的链路中，让一个演示算法能够逐步变成可部署、可复现、可维护的产品。

当前版本处于工程基线收口阶段：后端、CLI、数据治理、TorchVision 训练/推理、实验记录和 API 已具备；React + TypeScript 工作台已建立产品骨架。干净制品验证、权限体系、模型注册表和生产监控仍在路线图中，完成状态以测试和验收记录为准。

## 为什么是 DefectDock

- **交付导向**：配置快照、数据版本、实验状态、指标和模型产物形成可追溯记录。
- **工业指标**：除通用检测指标外，内置按类别检测率、漏检分析和阈值扫描。
- **可替换引擎**：训练引擎位于独立适配层，首个内置适配器基于 PyTorch/TorchVision。
- **许可证边界清晰**：内置链路不依赖 Ultralytics 包、源码或预训练权重。
- **前后端分离**：FastAPI 提供稳定接口，正式工作台采用 React + TypeScript。
- **现场接口预留**：支持图片、USB 摄像头、视频文件和 RTSP 输入。

## 架构

```text
React + TypeScript workbench
            │ REST
            ▼
FastAPI ── application services ── SQLite metadata
   │               │
   │               ├── dataset ingestion / CVAT / quality checks
   │               ├── evaluation / miss analysis / reports
   │               └── run configuration / events / artifacts
   ▼
engine adapter ── TorchVision Faster R-CNN ── DefectDock checkpoint
```

详细说明见 [架构文档](docs/architecture.md) 和 [迁移清单](docs/migration.md)。
产品与架构参考的许可证核验、可借鉴范围和引入流程见
[参考项目清单](docs/reference-projects.md)。

## 快速开始

### 1. 安装

```bash
git clone <your-repository-url> defectdock
cd defectdock
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[train]"
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[train]"
```

检查环境：

```bash
defectdock doctor
```

命令默认把当前目录作为可写工作区；也可以通过全局参数或环境变量显式指定。数据库、上传数据、模型激活状态和输出都从该工作区派生，不会写入 Python 安装目录：

```powershell
defectdock --workspace E:\AI-LLM\defectdock doctor
$env:DEFECTDOCK_WORKSPACE = "E:\AI-LLM\defectdock"
```

> 首次使用 `pretrained: true` 训练时，TorchVision 会下载其官方预训练权重。离线环境请提前缓存，或在配置中设为 `false`。

### 2. 准备与检查数据

导入 GC10-DET：

```bash
defectdock data import-gc10 /path/to/GC10-DET datasets/gc10-v1
defectdock data check datasets/gc10-v1/data.yaml
defectdock data stats datasets/gc10-v1/data.yaml
```

DefectDock 当前支持 `data.yaml + images/ + labels/` 目录布局，标签行格式为 `class cx cy width height`，坐标归一化到 `[0, 1]`。这是通用数据交换约定，不代表对特定训练框架的运行时依赖。

### 3. 训练

先检查计划，不启动训练：

```bash
defectdock validate configs/examples/gc10-torchvision.yaml
defectdock plan configs/examples/gc10-torchvision.yaml
defectdock run configs/examples/gc10-torchvision.yaml --dry-run
```

执行训练：

```bash
defectdock run configs/examples/gc10-torchvision.yaml
```

每次运行都会保存配置快照、事件流、指标、`best.ckpt` 和 `last.ckpt`。SQLite 记录默认位于 `.defectdock/defectdock.db`，模型和运行产物默认位于 `outputs/`；两者均不会进入 Git。

### 4. 推理与 API

激活模型并检测图片：

```bash
defectdock deploy outputs/<project>/object-detection/<run-id>/trainer_output/weights/best.ckpt
defectdock detect samples/example.jpg
```

启动 API：

```bash
defectdock serve --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000/docs` 查看交互式接口文档，健康检查位于 `GET /api/health`。

容器默认构建轻量 API 镜像，工作区固定挂载到 `/data`：

```bash
docker compose up --build
```

如果确实需要在同一镜像内安装 PyTorch/TorchVision 训练栈，可显式设置
`DEFECTDOCK_INSTALL_TRAIN=1` 后再构建。该模式会显著增大镜像，并且 GPU
交付还必须结合目标 CUDA/驱动环境单独验证；轻量镜像的健康检查会如实返回
`training_submission_enabled: false`，不会接受训练任务。

### 5. React 工作台

```bash
cd apps/web
pnpm install
pnpm run dev
```

开发环境默认将 `/api` 代理到 `http://127.0.0.1:8000`。

## 仓库结构

```text
defectdock/
├── apps/web/                 # React + TypeScript 正式工作台
├── configs/examples/         # 可验证的训练配置
├── docs/                     # 架构、迁移和许可证决策
├── scripts/                  # 许可证边界等发布检查
├── src/defectdock/
│   ├── api/                  # FastAPI 接口
│   ├── config/               # 严格配置与稳定哈希
│   ├── data/                 # 上传、CVAT、转换、检查和统计
│   ├── db/                   # SQLite 运行与数据集元数据
│   ├── engines/              # 可替换训练引擎适配器
│   ├── eval/                 # 工业验收、漏检和报告
│   ├── inference/            # 统一推理服务
│   └── stream/               # USB / RTSP / 视频流
└── tests/                    # 单元与 API 回归测试
```

## 许可证与商业交付

DefectDock 原创代码当前采用 [专有许可证](LICENSE)，在权利人明确开源策略之前不得对外复制或商业分发。依赖项保留各自许可证，关键边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [docs/licensing.md](docs/licensing.md)。

本仓库明确排除 Ultralytics Python 包、Ultralytics 源代码、训练脚本和预训练权重。CI 中的许可证边界检查会阻止相关运行时导入或依赖再次进入主链路。任何对外发布仍必须在冻结依赖后执行 SBOM、漏洞和许可证扫描，并由法律顾问审核最终分发组合。

## 路线图

- [x] 独立命名空间、配置模型和运行记录
- [x] 数据接入、CVAT 同步、质量检查和工业评测
- [x] TorchVision 训练、检查点和统一推理
- [x] React + TypeScript 工作台骨架
- [x] 单机单 GPU 后台任务队列、协作式取消和重启状态恢复
- [ ] 持久化任务队列、自动重试和多设备资源调度
- [ ] 参考 Geti 建立项目、数据版本、模型版本和部署版本的完整追溯关系
- [ ] 参考 Anomalib Studio 增加异常检测任务、热力图和阈值验收
- [ ] 用户、组织、角色与审计日志
- [ ] 模型注册、审批、灰度发布和回滚
- [ ] 参考 AWS DDA 增加云中立的边缘设备、部署状态和断网续传能力
- [ ] ONNX/TensorRT 可选部署适配器与基准测试
- [ ] PostgreSQL、对象存储和生产可观测性

## 协作与安全

提交规范与开发检查见 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要在公开 Issue 中提交凭据、客户数据或模型资产。

---

**English summary:** DefectDock is an industrial computer-vision lifecycle workbench with a React/TypeScript UI, FastAPI backend, auditable dataset/run metadata, industrial evaluation, and a built-in PyTorch/TorchVision detection adapter. Version `0.1.0` is an engineering foundation; production identity, asynchronous scheduling, registry, and observability remain planned work.
