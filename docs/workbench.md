# React 工作台

DefectDock 工作台位于 `apps/web`，使用 React、TypeScript 和 Vite。它调用
FastAPI 已有契约，不在浏览器中重新实现数据检查、训练或推理逻辑。

## 启动

先在仓库根目录启动 API。仅浏览和接入数据可以使用核心依赖：

```powershell
uv run defectdock serve --host 127.0.0.1 --port 8000
```

需要提交训练时，使用带 NVIDIA runtime 的 GPU Compose 服务：

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up api
```

然后启动开发工作台：

```powershell
cd apps/web
corepack pnpm install --frozen-lockfile
corepack pnpm run dev
```

打开 `http://127.0.0.1:5173`。Vite 会把 `/api` 请求代理到本机 8000 端口。

当 API 运行在网络模式时，工作台会显示凭据对话框。Token 只写入当前浏览器会话的
`sessionStorage`，关闭会话后失效；受保护的图片预览也通过携带认证头的请求加载。

## 页面与路由

工作台使用 React Router 拆分为独立页面，不再把完整业务链路堆叠在同一个长页面：

| 路由 | 职责 |
| --- | --- |
| `/overview` | 运行总览与主流程入口 |
| `/datasets` | 数据集列表 |
| `/datasets/:datasetId` | 图片、标注版本、冻结和训练快照 |
| `/training/new?datasetId=...&snapshotId=...` | 可复现训练配置 |
| `/runs` | 训练运行列表 |
| `/runs/:runId` | 运行状态、指标、失败原因和产物 |
| `/models` | 模型版本、激活历史、ONNX 导出和回滚 |
| `/inference` | 已激活模型的单图推理验证 |

数据集、训练快照和运行标识会进入 URL，因此详情页支持直接访问、刷新恢复以及浏览器
前进/后退。运行详情所需的事件与产物只在详情页每三秒轮询，离开页面后自动停止；
总览与运行列表仅刷新轻量摘要。

## 首条产品纵切

1. 创建数据集并上传原始图片；服务端验证图片、按内容哈希去重并记录元数据。
2. 上传 YOLO `.txt`、同步 CVAT，或用当前模型生成候选标注；模型候选必须检查并批准。
3. 冻结数据并创建训练快照；只有通过质量检查的快照才能进入训练配置。
4. 选择轮数、设备和预训练权重，提交 TorchVision 训练任务。
5. 提交后跳转到独立运行详情页；该页面每三秒刷新，展示轮次、设备、检出率、精确率、失败原因与产物路径。
6. 成功运行可登记为模型版本，完成哈希验证后原子激活；模型页可查看历史、回滚和导出 ONNX 包。
7. 在推理区选择现场图片完成单图验证；训练快照可通过列表和详情接口恢复。

页面不会把 API 失败静默吞掉：数据质量失败、训练依赖缺失、非法状态转换和推理
不可用都会显示后端返回的具体原因。训练可以取消；页面关闭不会终止后台任务。

## 当前边界

- 当前模型注册与批准者记录面向单工作站共享身份；多用户分级审批和灰度发布尚未实现；
- ONNX 导出会执行源模型一致性验证和本机 CPU 基准，但客户目标 GPU/加速器仍需单独验收；
- 开发服务器只用于本机开发；网络模式已有共享 Token、请求上限和审计基线，但独立前端生产镜像、TLS 终止、用户/组织/RBAC 和速率限制仍属于后续交付任务；
- 当前 UI 不替代真实客户数据、最终硬件和产线节拍验收；
- 现阶段没有用户、组织和角色隔离，不应直接暴露到不可信网络。

## 前端门禁

```powershell
corepack pnpm run lint
corepack pnpm run test
corepack pnpm run build
corepack pnpm audit --audit-level high
```

生产构建使用严格 TypeScript 检查。依赖锁定、SBOM、许可证与漏洞证据由发布工作流生成。
