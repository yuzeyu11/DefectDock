# DefectDock

> Build, verify, and ship industrial vision models.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-149ECA.svg)](apps/web)
[![API](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey.svg)](LICENSE)

DefectDock 是面向工业视觉项目的模型全生命周期工作台。它把数据接入、标注协同、质量检查、训练、工业指标验收、模型激活和现场视频流推理放在一条可审计的链路中，让一个演示算法能够逐步变成可部署、可复现、可维护的产品。

当前版本已经建立工程交付基线：后端、CLI、数据治理、TorchVision 训练/推理、实验记录、自动标注审核、模型注册与回滚均已具备；React + TypeScript 工作台已接通首条可操作产品纵切。真实客户数据与目标硬件验收、权限体系和生产监控仍在路线图中，完成状态以测试和验收记录为准。

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
pip install -e ".[train,export]"
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[train,export]"
```

仓库同时提交跨平台 `uv.lock`。需要严格复现已锁定依赖时，可使用：

```bash
uv sync --locked --extra train --extra export --extra dev
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

默认以本地模式启动 API，仅允许监听回环地址且不要求登录：

```bash
defectdock serve --host 127.0.0.1 --port 8000
```

打开 `http://localhost:8000/docs` 查看交互式接口文档，健康检查位于 `GET /api/health`。

需要从其他机器访问时，必须显式启用网络模式并提供至少 32 字节的共享 Token；生产环境还应在可信反向代理处终止 TLS：

```powershell
$env:DEFECTDOCK_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
defectdock serve --mode network --host 0.0.0.0 --port 8000
```

网络模式使用 `Authorization: Bearer <token>`。写操作、鉴权失败和超限请求会以不含请求体和凭据的 JSONL 记录到 `.defectdock/audit.jsonl`；单次请求默认上限为 512 MiB，可通过 `DEFECTDOCK_MAX_REQUEST_BYTES` 下调。

容器默认构建轻量 API 镜像，工作区固定挂载到 `/data`：

```powershell
$env:DEFECTDOCK_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
docker compose up --build
```

容器固定运行在网络模式，未设置合格的 `DEFECTDOCK_API_TOKEN` 时 Compose 会拒绝启动。健康检查保持公开，其余 API 要求 Bearer Token。

GPU 训练使用显式 Compose 覆盖，它会从同一份 `uv.lock` 安装训练栈并申请
NVIDIA GPU：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build api
docker compose -f compose.yaml -f compose.gpu.yaml up api
```

轻量镜像的健康检查会如实返回 `training_submission_enabled: false`，不会接受
训练任务。GPU 镜像的本地探针、训练烟雾测试、自托管 CI 与验收边界见
[GPU 交付说明](docs/gpu-delivery.md)。

### 5. React 工作台

```bash
cd apps/web
pnpm install
pnpm run dev
```

开发环境默认将 `/api` 代理到 `http://127.0.0.1:8000`。工作台已支持：

- 创建图片数据集并查看去重后的图片预览；
- 上传 YOLO 文本标注，或用已注册模型生成候选框并经人工批准；
- 创建/打开 CVAT 任务、同步完成的标注，以及查询不可变训练快照；
- 配置并提交训练、轮询状态、取消运行、查看指标和模型产物；
- 注册并原子激活成功运行的最佳 checkpoint、审计历史与一键回滚；
- 导出带哈希、运行时和数值一致性报告的 ONNX 包，并提交单图推理验证。

轻量 API 镜像可以浏览和接入数据，但会明确禁用训练提交；需要从页面发起
训练时应使用 GPU 镜像。具体操作和当前边界见
[工作台说明](docs/workbench.md)。

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

本仓库明确排除 Ultralytics Python 包、Ultralytics 源代码、训练脚本和预训练权重。CI 中的许可证边界检查会阻止相关运行时导入或依赖再次进入主链路。发布证据工作流会针对锁定的 Python 与前端生产依赖生成 SBOM、漏洞报告、许可证清单及校验和；流程与人工复核边界见 [发布证据说明](docs/release-evidence.md)。最终分发组合仍须由法律顾问审核。

## 路线图

- [x] 独立命名空间、配置模型和运行记录
- [x] 数据接入、CVAT 同步、质量检查和工业评测
- [x] TorchVision 训练、检查点和统一推理
- [x] React + TypeScript 工作台首条操作纵切（数据、标注、快照、训练、产物、激活和推理）
- [x] 单机单 GPU 后台任务队列、协作式取消和重启状态恢复
- [x] Alembic 数据库迁移、升级前备份、失败恢复和显式标注版本外键
- [x] Python/前端 SBOM、漏洞与许可证发布证据工作流
- [x] 轻量容器固定基础镜像摘要、严格使用 `uv.lock`，并移除运行时构建工具
- [x] CUDA 13 GPU 训练镜像、Compose 覆盖、运行时探针与容器内训练烟雾验收
- [x] 模型版本注册、制品完整性校验、原子激活、审计历史和回滚
- [x] 已注册模型自动标注、候选预览、人工批准与冻结门禁
- [x] 训练快照登记、标注版本外键、哈希验证和列表/详情查询
- [x] ONNX 导出契约、数值一致性验证、CPU 基准和部署 manifest
- [ ] 持久化任务队列、自动重试和多设备资源调度
- [ ] 参考 Geti 扩展到多项目和部署实例级的完整追溯关系
- [ ] 参考 Anomalib Studio 增加异常检测任务、热力图和阈值验收
- [ ] 用户、组织、角色与审计日志
- [ ] 用户身份下的多级模型审批与灰度发布（当前已有单工作站注册、审计和回滚）
- [ ] 参考 AWS DDA 增加云中立的边缘设备、部署状态和断网续传能力
- [ ] 在客户目标硬件验收 ONNX 性能并按需求增加 TensorRT 适配器
- [ ] PostgreSQL、对象存储和生产可观测性

## 协作与安全

提交规范与开发检查见 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要在公开 Issue 中提交凭据、客户数据或模型资产。

---

**English summary:** DefectDock is an industrial computer-vision lifecycle workbench with a React/TypeScript UI, FastAPI backend, auditable dataset/run/model metadata, reviewed model-assisted labeling, industrial evaluation, ONNX export, and a built-in PyTorch/TorchVision detection adapter. Version `0.1.0` is an engineering foundation; production identity, asynchronous scheduling, target-hardware qualification, and observability remain planned work.
