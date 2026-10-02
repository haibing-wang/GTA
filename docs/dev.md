# GTA (General Tool Agent) 开发与使用指南 (dev.md)

本文档专为本 Fork 仓库开发者与使用者编写，涵盖 **GTA-1 (GTA-Atomic)** 与 **GTA-2 (GTA-Workflow)** 的架构全貌、环境搭建、数据集准备、三种核心评测流程、常见避坑点以及本仓库所做的优化修复。

---

## 目录
1. [项目全貌与评测分层体系](#1-项目全貌与评测分层体系)
2. [环境搭建与管理建议](#2-环境搭建与管理建议)
3. [数据集准备与校验](#3-数据集准备与校验)
4. [三种评测模式实战指南](#4-三种评测模式实战指南)
   - [模式一：标准 OpenCompass + Lagent + AgentLego 工具闭环评测](#模式一标准-opencompass--lagent--agentlego-工具闭环评测)
   - [模式二：自定义模型与 Agent 快速接入](#模式二自定义模型与-agent-快速接入)
   - [模式三：外部黑盒 Agent 产物端到端评测 (Manus / OpenClaw / 自定义 Agent)](#模式三外部黑盒-agent-产物端到端评测-manus--openclaw--自定义-agent)
5. [辅助诊断脚本使用](#5-辅助诊断脚本使用)
6. [常见陷阱与排查方案 (FAQ)](#6-常见陷阱与排查方案-faq)
7. [本次 Fork 仓库修复与优化清单](#7-本次-fork-仓库修复与优化清单)

---

## 1. 项目全貌与评测分层体系

GTA 是针对大语言模型与通用工具智能体（General Tool Agent）构建的基准评测体系，包含两个层级：

| 评测层级 | 核心目标 | 评测重点 | 关键文件与数据集 |
| :--- | :--- | :--- | :--- |
| **GTA-Workflow (GTA-2)** | 开放式、长程生产力工作流 | 关注 Agent 最终产出的**可交付成果**（报告、图表、代码、多媒体等），由 GPT-5.2 多模态评测器做叶子节点与全局加权打分 | 配置：`opencompass/configs/eval_gta_bench_v2.py`<br>数据集：`data/gta_dataset_v2`<br>端到端评测：`agent_app_eval/` |
| **GTA-Atomic (GTA-1)** | 短程原子工具调用 | 评估工具选择（ToolAcc）、参数预测（ArgAcc）、指令跟随（InstAcc）与总结能力（SummAcc） | 说明：`README_GTA-1.md`<br>配置：`opencompass/configs/eval_gta_bench.py`<br>数据集：`data/gta_dataset` |

### 核心目录结构
```text
GTA/
├── opencompass/           # 评测主框架（基于 OpenCompass 与 Lagent）
│   ├── configs/           # 评测配置，重点为 eval_gta_bench_v2.py
│   ├── data/              # 数据集存放目录（需将数据集解压至此）
│   │   ├── gta_dataset_v2 # GTA-Workflow 数据集
│   │   └── gta_dataset    # GTA-Atomic 数据集
│   └── opencompass/       # OpenCompass 核心源码与 Lagent 模型适配器
├── agentlego/             # 工具执行服务（提供数十种多模态/逻辑/文件处理工具）
│   ├── benchmark.py       # 工具扩展与适配
│   └── benchmark_toollist_v2.txt # GTA-2 使用的 37 种工具清单
├── agent_app_eval/        # 针对外部产品级 Agent（如 Manus、OpenClaw）的端到端评测模块
│   ├── run_agents.py      # 驱动外部 Agent 执行
│   ├── score_with_gpt52.py# 独立打分脚本（调用 GPT-5.2 多模态打分器）
│   └── examples/          # 结果转换适配器示例
├── scripts/               # 辅助开发与诊断脚本（本仓库新增）
│   ├── check_gta_env.py   # 环境、数据集、服务连通性诊断
│   └── download_dataset.py# 数据集一键下载与解压脚本
└── docs/                  # 详细文档与说明（含 dev.md）
```

---

## 2. 环境搭建与管理建议

由于本项目涉及 **OpenCompass 评测引擎**、**AgentLego 工具服务（CV/音视频/OCR 栈）** 以及 **轻量打分评测**，官方推荐采用 Conda 进行环境隔离以避免依赖冲突。

### 方案 A：完整开发环境（推荐用于全流程测试与本地部署工具）

#### 1) AgentLego 工具服务环境 (Python 3.11)
```bash
conda create -n agentlego python=3.11.9 -y
conda activate agentlego

cd agentlego
pip install -r requirements_all.txt
pip install -r requirements_gta_v2.txt
pip install -e .
mim install mmengine
mim install mmcv==2.1.0

# 关键兼容性修复：启用 SDPA 支持
# 编辑 ~/anaconda3/envs/agentlego/lib/python3.11/site-packages/transformers/modeling_utils.py
# 将第 1279 行的 _supports_sdpa = False 改为 _supports_sdpa = True
```

#### 2) OpenCompass 评测环境 (Python 3.10)
```bash
conda create --name opencompass python=3.10 -y
conda activate opencompass

# 安装 PyTorch (根据你的机器平台选择 CUDA 或 CPU/MPS)
pip install torch torchvision

cd agentlego
pip install -e .
cd ../opencompass
pip install -e .
pip install huggingface_hub==0.25.2 transformers==4.40.1 requests tiktoken
```

### 方案 B：轻量评测环境（针对外部 Agent 产物打分与 API 评测）
如果仅评测外部 Agent 产物（`agent_app_eval`），或者只运行 step-by-step（无需本地工具执行服务），可直接使用 `uv` 或现有 Python 虚拟环境：
```bash
# 进入仓库根目录
uv pip install requests openai tiktoken
```

---

## 3. 数据集准备与校验

### 1) 一键下载脚本（推荐）
在项目根目录下直接运行：
```bash
# 下载 GTA-Workflow (GTA-2) 数据集 (约 6GB)
python scripts/download_dataset.py --dataset workflow

# 或下载 GTA-Atomic (GTA-1) 数据集
python scripts/download_dataset.py --dataset atomic

# 或全部下载
python scripts/download_dataset.py --dataset all
```

### 2) 手动下载与解压路径要求
若手动下载，请确保解压后的目录结构如下：
```text
opencompass/data/
├── gta_dataset_v2/           # GTA-Workflow 数据集根目录
│   ├── end.json             # 评测题目与交付物评价树
│   ├── toolmeta.json        # 工具元数据描述
│   └── ...                  # 附带的多模态上下文文件
└── gta_dataset/              # GTA-Atomic 数据集（如需运行 GTA-1）
    ├── toolmeta.json
    └── ...
```

官方下载直链：
- [GTA-Workflow (v0.2.0)](https://github.com/open-compass/GTA/releases/download/v0.2.0/gta_workflow_dataset.zip)
- [GTA-Atomic (v0.1.0)](https://github.com/open-compass/GTA/releases/download/v0.1.0/gta_dataset.zip)

---

## 4. 三种评测模式实战指南

### 模式一：标准 OpenCompass + Lagent + AgentLego 工具闭环评测

适用场景：在本地/服务器完整运行模型自主调工具、获取执行结果、迭代推理直至输出。

#### 步骤 1：部署并启动被测模型服务
支持 LMDeploy、vLLM 或任意兼容 OpenAI 接口的模型服务。例如使用 LMDeploy 启动本地模型：
```bash
conda activate lmdeploy
lmdeploy serve api_server ~/models/qwen1.5-7b-chat --server-port 12580 --model-name qwen1.5-7b-chat
```

#### 步骤 2：启动 AgentLego 工具服务
部署需要使用的 API Key（搜索与数学 OCR）：
```bash
export SERPER_API_KEY='your_serper_key_for_google_search_tool'
export MATHPIX_APP_ID='your_mathpix_app_id'
export MATHPIX_APP_KEY='your_mathpix_app_key'

conda activate agentlego
cd agentlego
agentlego-server start --port 16181 --extra ./benchmark.py `cat benchmark_toollist_v2.txt` --host 0.0.0.0
```

#### 步骤 3：配置并执行评测
本仓库对 `opencompass/configs/eval_gta_bench_v2.py` 进行了优化，已支持从环境变量中动态读取 API 地址、密钥与工具服务地址，无需每次硬编码：

```bash
conda activate opencompass
cd opencompass

# 设置环境变量
export OPENAI_API_BASE="http://127.0.0.1:12580/v1/chat/completions" # 你的模型接口
export OPENAI_API_KEY="EMPTY"                                        # 本地模型设为 EMPTY
export OPENAI_MODEL_NAME="qwen1.5-7b-chat"                           # 目标模型名
export GTA_TOOL_SERVER="http://127.0.0.1:16181"                     # 工具服务地址
export OPENCOMPASS_TOOLMETA_PATH="data/gta_dataset_v2/toolmeta.json"

# 1. 仅执行推理阶段 (生成预测结果)
python run.py configs/eval_gta_bench_v2.py --max-num-workers 8 --debug --mode infer

# 2. 仅执行评测打分阶段 (复用之前推理生成的时间戳文件夹)
python run.py configs/eval_gta_bench_v2.py --max-num-workers 8 --debug --reuse [YYYYMMDD_HHMMSS] --mode eval

# 3. 推理并评测一体化执行
python run.py configs/eval_gta_bench_v2.py -p llmit -q auto --max-num-workers 8 --debug
```

---

### 模式二：自定义模型与 Agent 快速接入

如果你需要接入云端商业模型（如 DeepSeek-V3、Claude 3.5/4.5、Gemini 2.5、GPT-5）或第三方聚合中转，只需在 `opencompass/configs/eval_gta_bench_v2.py` 的 `models = [...]` 中启用或新增相应配置项：

```python
dict(
    abbr='my-custom-model',
    type=LagentAgent,
    agent_type=ReAct,
    max_turn=10,
    llm=dict(
        type=OpenAI,
        path='gpt-4o',                                     # 模型代号
        key=os.getenv('OPENAI_API_KEY'),                  # 读取环境变量 API Key
        openai_api_base='https://api.openai.com/v1/chat/completions', # 注意：必须包含 /chat/completions
        query_per_second=2,
        max_seq_len=131072,
    ), 
    tool_server=os.getenv('GTA_TOOL_SERVER', 'http://127.0.0.1:16181'),
    tool_meta='data/gta_dataset_v2/toolmeta.json',
    batch_size=4,
)
```

> **提示**：如果评测不需要实际调用工具（即 step-by-step 模式，仅打分模型在每一步给定黄金上下文时的决策准确率），可不填写 `tool_server`，仅保留 `tool_meta`。详细说明见 [docs/ADDING_NEW_AGENT_OR_LLM.md](ADDING_NEW_AGENT_OR_LLM.md)。

---

### 模式三：外部黑盒 Agent 产物端到端评测 (Manus / OpenClaw / 自定义 Agent)

适用场景：评估 Manus、OpenClaw、Kortix 等独立 Agent 产品的实际交付物表现。无需运行 OpenCompass 或 Lagent。

#### 步骤 1：收集并存放 Agent 执行产物
为每个 `task_id` 建立独立子目录，存放输出结果：
```text
agent_app_eval/agent_app_result/
  ├── 1/
  │   ├── final.txt          # 必选：Agent 最终回复内容
  │   └── files/             # 可选：Agent 生成的附件成果（图表、pdf、代码、音频等）
  │       ├── report.pdf
  │       └── figure.png
  └── 2/
      └── final.txt
```

#### 步骤 2：生成评测评估包 (Eval-Pack)
使用转换适配器生成标准 `eval-pack.json`：
```bash
python agent_app_eval/examples/build_eval_pack_from_agent_app_result.py \
  --result-dir agent_app_eval/agent_app_result \
  --out-pack agent_app_eval/runs/agent_app_eval_pack.json
```
> **优化亮点**：本仓库已优化该脚本，`--task-ids` 参数变为可选，会自动扫描 `result_dir` 中的任务目录或从 `end.json` 中自动索引，无须手动准备 task_ids 文件。

#### 步骤 3：使用 GPT-5.2 打分器进行多模态评分
配置打分器环境变量（注意：Evaluator 使用 OpenAI Responses API，其 Base URL 需以 `/v1` 结尾，**不能**包含 `/chat/completions`）：

```bash
export EVAL_OPENAI_API_KEY="sk-..."
export EVAL_OPENAI_BASE_URL="https://api.openai.com/v1"  # 必须以 /v1 结尾
# 可选代理配置
export EVAL_PROXY="http://127.0.0.1:7890"

# 1. 运行 Dry-Run 校验（不消耗 Token，验证所有文件与格式是否齐全）
python agent_app_eval/score_with_gpt52.py \
  --in-pack agent_app_eval/runs/agent_app_eval_pack.json \
  --dry-run

# 2. 正式打分（支持断点续评，中断后重新运行同命令即可继续）
python agent_app_eval/score_with_gpt52.py \
  --in-pack agent_app_eval/runs/agent_app_eval_pack.json
```

打分完成后会自动输出明细文件（`*_scores.json`）与汇总评分（`*_scores_summary.json`）。

---

## 5. 辅助诊断脚本使用

为了方便快速排查环境与配置问题，本仓库提供了专属诊断工具：

```bash
# 执行完整环境与配置自检
python scripts/check_gta_env.py
```

诊断工具会依次检测：
1. **Python 环境与关键库**：Python 版本、`requests`、`openai`、`tiktoken`、`mmengine`、`lagent`、`agentlego` 的安装状态。
2. **数据集目录**：`opencompass/data/gta_dataset_v2` 与 `gta_dataset` 的完整性，检查 `toolmeta.json`、`end.json` 是否就位。
3. **工具服务连通性**：检测 `http://127.0.0.1:16181` 是否正常监听。
4. **环境变量与密钥配置**：检查推理端和打分端的 API Key、Base URL 及代理设置。

---

## 6. 常见陷阱与排查方案 (FAQ)

### Q1：为什么 `score_with_gpt52.py` 报错 `base_url` 不正确？
- **原因**：OpenCompass 推理模型配置中的 `openai_api_base` 使用原生 `requests.post`，需写完整接口地址（如 `https://xxx/v1/chat/completions`）；而 `score_with_gpt52.py` 使用官方 OpenAI Python SDK，`base_url` 必须写为 `https://xxx/v1`。
- **解决**：本仓库已在 `score_with_gpt52.py` 中增加了自动规范化处理（会自动剥离末尾的 `/chat/completions`），确保环境变量写错时也能自适应。

### Q2：运行 `eval_gta_bench_v2.py --mode eval` 提示任务数为 0？
- **原因**：早期配置中缺少显式 `eval` 分区块，导致 OpenCompass 在 eval 模式下找不到 runner 任务。
- **解决**：配置文件已增加 `eval = dict(partitioner=..., runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)))`，可正常按时间戳复用推理结果打分。

### Q3：工具调用时报 `ValueError: too many values to unpack`？
- **原因**：上游 Lagent 的 `ActionExecutor` 在解析类似 `Solver.v2.solve` 这类带有多点的工具名称时，使用 `name.split('.')` 导致解包失败。
- **解决**：在 `opencompass/opencompass/models/lagent.py` 中已包含安全 monkey-patch（`_patch_lagent_action_executor_split`），改用 `rsplit('.', 1)`，平滑支持多级工具命名。

### Q4：macOS 环境下运行 AgentLego 报找不到 CUDA 或 mim 安装失败？
- **建议**：AgentLego 工具服务中包含部分 GPU 模型（如目标检测、视频生成）。在 macOS 上如果仅用于评测纯文本/API 类 Agent，推荐使用**模式三（agent_app_eval）**或通过远程 Linux 服务器启动 `agentlego-server`，本地配置远程 IP 端口即可。

---

## 7. 本次 Fork 仓库修复与优化清单

针对原始开源项目在实际部署和使用过程中暴露的问题，本项目做出了如下核心修复与增强：

1. **修复 `agent_app_eval` 根目录寻址 Bug**：
   - 修复了 `agent_app_eval/score_with_gpt52.py` 和 `agent_app_eval/run_agents.py` 中 `_repo_root()` 使用 `parents[2]` 导致寻址越界到上级目录的错误，更正为精准指向 GTA 仓库根目录。
2. **修复 `build_eval_pack_from_agent_app_result.py` 强依赖不存在的固定文件**：
   - 将必选的 `--task-ids` 优化为可选参数，支持自动从 Agent 产物子目录自动识别数字 Task ID，并增加了对 `end.json` 不存在时的清晰错误提示。
3. **解除未使用的强制重依赖**：
   - 优化 `opencompass/opencompass/datasets/gta_bench_v2.py`，将未实际使用但会导致环境报错的 `sentence_transformers` 改为安全导入，降低环境依赖门槛。
4. **动态环境变量支持**：
   - 优化 `opencompass/configs/eval_gta_bench_v2.py`，支持 `OPENAI_API_BASE`、`OPENAI_API_KEY`、`OPENAI_MODEL_NAME`、`GTA_TOOL_SERVER` 等环境变量，不再需要对代码进行硬编码修改。
5. **文档链接纠错**：
   - 修正根目录 `README.md` 中指向已更名文件 `README_GTA1.md` 的断链，更新为 `README_GTA-1.md`。
6. **提供配套实用工具**：
   - 新增 `scripts/check_gta_env.py`：开箱即用的环境与数据集诊断工具。
   - 新增 `scripts/download_dataset.py`：官方数据集自动下载解压脚本。
