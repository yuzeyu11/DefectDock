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

## 首条产品纵切

1. 创建数据集并上传原始图片；服务端验证图片、按内容哈希去重并记录元数据。
2. 上传与原图片同名的 YOLO `.txt` 标注；服务端生成不可变标注版本与 manifest 哈希。
3. 冻结数据并创建训练快照；只有通过质量检查的快照才能进入训练配置。
4. 选择轮数、设备和预训练权重，提交 TorchVision 训练任务。
5. 页面每三秒刷新所选运行，展示轮次、设备、检出率、精确率、失败原因与产物路径。
6. 成功运行可以激活 `best.ckpt`，随后在推理区选择现场图片完成单图验证。

页面不会把 API 失败静默吞掉：数据质量失败、训练依赖缺失、非法状态转换和推理
不可用都会显示后端返回的具体原因。训练可以取消；页面关闭不会终止后台任务。

## 当前边界

- 当前激活接口是单机配置切换，不是正式模型注册表，也没有审批、灰度和原子回滚；
- 训练快照在页面会话中生成，刷新页面后需要重新生成，后续应增加快照列表 API；
- 开发服务器只用于本机开发，独立前端生产镜像和网络安全模式属于后续交付任务；
- 当前 UI 不替代真实客户数据、最终硬件和产线节拍验收；
- 现阶段没有用户、组织和角色隔离，不应直接暴露到不可信网络。

## 前端门禁

```powershell
corepack pnpm run lint
corepack pnpm run build
corepack pnpm audit --audit-level high
```

生产构建使用严格 TypeScript 检查。依赖锁定、SBOM、许可证与漏洞证据由发布工作流生成。
