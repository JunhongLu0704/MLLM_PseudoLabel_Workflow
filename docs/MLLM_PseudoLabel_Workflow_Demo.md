# 面向视觉训练数据质量控制的受控多模态伪标签验证工作流

![项目总览](figures/project_poster.png)

---

## 1. 项目概览

### 1.1 项目简介

本项目是一个面向视觉训练数据质量控制的自主研究 demo，目标是在检测器自动生成伪标签之后，使用本地部署的多模态大模型对候选标注进行质量验证与筛选。系统通过固定 LangGraph 工作流规定数据流、工具节点、模型输入输出格式和回退逻辑，使多模态模型在明确边界内参与视觉证据判断、补充证据选择和最终质量决策。

当前 demo 以 VOC→Clipart 固定 256 图作为最小验证场景。输入是 teacher detector 生成的候选伪标签，包括候选框、预测类别和检测器置信度；输出是经过多模态质量验证工作流筛选后的伪标签集合。

本项目可以应用在半监督、跨域适应等依靠伪标签对无监督数据进行学习的方法中，作为比传统的基于统计的阈值方法更有效的伪标签筛选方法，也可以用于开放词汇模型和基于视觉模型生成标注的验证后处理。

### 1.2 核心流程

```text
教师伪标签
  ↓
图像级先验
  ↓
候选池与重叠关系
  ↓
候选框级视觉验证
  ↓
补充证据选择
  ↓
受控最终决策
  ↓
过滤后的伪标签
```

该流程体现两个设计目标：

1. **固定流程约束模型行为**：工作流决定每个阶段的输入、输出和执行顺序，模型只在预设节点内按结构化协议生成结果。
2. **受限节点释放模型能力**：在图像级先验、候选框级视觉验证、补充证据选择和最终决策等节点，模型可以在限定输入和输出协议下发挥视觉理解能力。

### 1.3 当前验证结果

| 项目 | 内容 |
|---|---|
| 项目定位 | 面向视觉训练数据质量控制的受控多模态伪标签验证工作流 |
| 任务场景 | VOC→Clipart 伪标签筛选 |
| 图像规模 | 256 张 Clipart 图像 |
| 类别数量 | 20 类 VOC 目标 |
| 候选标注总数 | 1921 |
| 有效候选框 | 1776 |
| 对比基线 | `score≥0.20` 固定置信度阈值 |
| 主结果 | F1：0.2815 → 0.4548；mAP50：0.0850 → 0.1443；FP：559 → 349 |

---
注：有效候选框指排除面积过小、宽高异常等退化候选后，可形成合法 MLLM 视觉输入并进入正式评测统计的候选框。

## 2. 项目动机

### 2.1 为什么需要伪标签质量验证

视觉模型训练中，检测器、开放词汇模型或基础视觉模型可以批量生成候选标注，用于自动标注、半监督学习、跨域适应和垂直场景模型构建。此类候选标注可以降低人工标注成本，但也会引入噪声：

- 背景区域被误检为目标；
- 视觉相似类别之间发生混淆；
- 同一目标周围产生多个重复框；
- 候选框只覆盖目标局部，或框位置明显偏移；
- 高置信度候选框仍然可能在视觉上无效。

固定置信度阈值是常见筛选方式，但检测器得分并不能直接回答候选区域中的视觉内容是否真实、类别是否匹配、候选框是否覆盖有效目标。对于训练数据质量控制而言，仅依赖得分阈值会在 precision 与 recall 之间做机械取舍，难以针对具体候选框补充视觉证据。

### 2.2 为什么需要多模态模型

候选伪标签本质上同时包含视觉信息和结构化信息：

- 视觉输入：整图、候选框裁剪图、标记全图、同图参考图、源域类别参考图；
- 结构化输入：预测类别、检测器得分、候选框坐标、候选框尺寸、重叠关系、候选来源原因；
- 任务先验：VOC 类别集合、目标外观提示、容易混淆的类别和场景级提示。

多模态模型适合在这些信息之间建立联系：它可以观察局部裁剪区域是否包含目标，也可以结合标记全图判断候选框位置是否合理，还可以参考同图候选样本或源域类别样本进行外观对照。

### 2.3 为什么要使用受控工作流，而不是自由 agent

当前项目没有让大模型自由规划流程或直接操作全部伪标签。系统使用固定 LangGraph 工作流组织状态流和工具节点，并在特定节点给予模型受限自由度：

- 工作流决定执行顺序和工具边界；
- 模型只能在指定节点生成图像级先验、候选框验证结果、补充证据选择和最终决策；
- 补充证据选择只发生在工作流判定需要额外证据的候选框上；
- 模型输出必须是固定 JSON 字段；
- 输出非法、候选编号不一致或模型调用失败时，系统回退到规则逻辑。

因此，本项目的重点是设计一个让 LLM 能够稳定参与视觉数据质量控制的工作流。

---

## 3. 任务设置与评估协议

### 3.1 输入数据

目标伪标签数据来自固定的 VOC→Clipart 256 图导出文件。每张图包含 teacher detector 生成的候选框及其预测类别和置信度。系统处理的字段包括：

| 输入项 | 用途 |
|---|---|
| 目标图像 | 图像级先验和标记全图输入 |
| bbox | 候选框裁剪、尺寸判断和重叠关系计算 |
| predicted class | 伪标签预测类别 |
| score | 伪标签预测分数 |
| area / short side | 小目标、退化框和边界风险判断 |

### 3.2 数据设置

| 指标 | 数值 |
|---|---:|
| 图像数量 | 256 |
| 类别数量 | 20 |
| 候选标注总数 | 1921 |
| 有效候选框 | 1776 |
| 有效目标数 | 198 |

### 3.3 GT 安全边界

目标域真实标注只在推理结束后用于离线评测，不进入任何模型消费输入。具体而言，目标域 GT 不进入：

- 图像级先验 prompt；
- 候选框级视觉验证 prompt；
- 补充证据选择输入；
- 受控最终决策输入；
- repair prompt 或 validated model output。

源域 VOC 标注仅用于构建类别参考样本库，作为外观参考，不参与目标域评测，也不直接决定目标候选框的保留或删除。

### 3.4 对比方法与评价指标

本文档只将多模态质量验证工作流与固定 score 阈值基线比较。

| 方法 | 说明 |
|---|---|
| `score≥0.20` 基线 | 只根据 teacher detector 置信度筛选候选框 |
| 多模态质量验证工作流 | 使用固定 LangGraph 工作流、候选框级视觉验证、补充证据选择和受控最终决策筛选伪标签 |

评价指标包括 Precision、Recall、F1、FP 和 filtered mAP50。

---

## 4. 系统设计

### 4.1 总体流程

```text
Run-level preflight
├─ 构建源域类别参考样本库
└─ 作为后续候选框判断的外观参考

Per-image workflow
├─ 图像级先验
├─ 重叠关系分析
├─ 有界候选池构建
├─ 候选框级视觉验证
├─ 补充证据选择
├─ 同图参考 / 源域参考检索
└─ 受控最终决策

Post-run stage
├─ 候选框级结果表
├─ 工具调用轨迹
├─ 图像级处理轨迹
├─ token 与延迟统计
└─ 离线评测指标
```

### 4.2 LangGraph 工作流

当前展示版本使用固定 LangGraph 图结构：

```text
build_priors
  → build_duplicate_clusters
  → build_candidates
  → process_proposals
  → finalize_image
  → END
```

在每张图进入 LangGraph 前，系统先执行一次 run-level preflight：

```text
build_class_exemplar_bank
```

这个 preflight 基于源域 VOC2012 标注裁剪类别样本，构建小型类别外观参考库。该参考库仅作为外观证据，不使用目标域 GT。

### 4.3 工作流节点

| 阶段 | 输入 | 动作 | 输出 | 约束 |
|---|---|---|---|---|
| 源域参考样本库 | VOC2012 源域标注 | 裁剪代表性类别样本 | class exemplar bank | 仅源域 GT；不访问目标域 GT |
| 图像级先验 | 整图、候选摘要、VOC 类别 | MLLM 生成场景软提示 | visible / unlikely classes、confusions、guidance | 不输出 keep/drop；只用 VOC 类别 |
| 重叠关系分析 | bbox、score、class | class-agnostic IoU 聚类 | duplicate cluster info | 只提供结构证据，不决策 |
| 候选池构建 | 所有候选框、重叠信息 | 选择值得 MLLM 处理的候选框 | candidate proposal ids / reasons | 固定策略；有界候选池 |
| 候选框级视觉验证 | crop、marked image、metadata、prior | MLLM 判断目标是否存在、类别是否匹配、框是否有效 | visual validity / evidence strength / recommendation | JSON schema 校验；失败回退 |
| 补充证据选择 | 视觉验证结果、重叠信息、可用证据类型 | LLM 在预定义范围内选择补充证据 | selected evidence tools | 不直接决定 keep/drop |
| 补充证据检索 | 同图候选 / 源域参考库 | 读取参考裁剪图 | same-image references / source exemplars | 仅作为 supporting evidence |
| 受控最终决策 | 主验证结果、补充证据、score、metadata | 输出 keep / low_weight / drop | final decision | 编号校验、枚举校验、保护规则、回退 |

### 4.4 候选池策略

系统不会无约束地把所有候选框都交给多模态模型。候选池由固定策略产生，候选进入原因包括：

- `uncertain_score_band`：得分处于不确定区间；
- `border_touching`：候选框接触图像边界；
- `small_box`：候选框短边过小；
- `duplicate_cluster`：属于重叠簇；
- `top_score_sanity_check`：每张图中高分候选的 sanity check。

非候选框跳过 MLLM 验证，使用规则逻辑完成处理。这样既控制模型调用成本，也避免把流程变成无界大模型推理。

### 4.5 受限证据选择

补充证据选择只发生在系统判定需要额外证据的候选框上。触发条件包括：

```text
候选框有效
且满足以下任一条件：
  visual_validity == uncertain
  evidence_strength == weak
  proposal 属于重叠簇
  candidate_reasons 包含 duplicate_cluster
```

当前可选补充证据类型为：

| 补充证据 | 作用 |
|---|---|
| 同图同类参考样本 | 从同一目标图中检索同类候选框裁剪图，辅助判断当前候选是否与同图同类外观一致 |
| 源域类别参考样本 | 从源域 VOC 类别参考库中读取对应类别裁剪图，作为类别外观参考 |

模型只能在预定义范围内选择证据类型，不能任意调用未定义工具，也不能通过证据工具直接决定最终标签。

---

## 5. MLLM 输入输出设计

### 5.1 多模态输入

系统为不同 MLLM 节点构造不同输入。

#### 图像级先验输入

| 输入 | 作用 |
|---|---|
| full image | 让模型理解图像风格、可能出现的类别和场景复杂度 |
| teacher proposal summary | 提供候选框数量、top classes、score range、小框与大框数量 |
| VOC class list | 限制模型只能使用 VOC 类别名 |
| heuristic prior | 提供图像尺寸、RGB 统计和基本审核提示 |

#### 候选框级视觉验证输入

| 输入 | 作用 |
|---|---|
| crop image | 聚焦候选框局部外观 |
| marked full image | 显示候选框在整图中的位置和上下文 |
| proposal metadata | class、score、bbox、short side、area |
| compact prior | 图像级提示、类别提示和审核约束 |
| context summary | near border、score band 等结构化上下文 |

#### 最终决策输入

| 输入 | 作用 |
|---|---|
| teacher score | 检测器原始置信度参考 |
| proposal metadata | 候选框类别、位置、尺寸 |
| VerifySingleBox_MM 结果 | 主要视觉证据 |
| optional evidence | 同图参考和源域类别参考 |
| decision policy | 指定主证据、辅助证据和保护规则 |

### 5.2 输出协议

#### 图像级先验输出

```json
{
  "scene_type": "string",
  "image_style": "clipart | natural | mixed | uncertain",
  "visible_object_hints": ["VOC class names only"],
  "unlikely_classes": ["VOC class names only"],
  "potential_confusions": [
    {"class_a": "VOC class", "class_b": "VOC class", "reason": "short string"}
  ],
  "review_guidance": ["short strings, no keep/drop decisions"],
  "confidence": "low | medium | high"
}
```

#### 候选框级视觉验证输出

```json
{
  "proposal_id": "same id",
  "visual_validity": "valid | uncertain | invalid | unavailable",
  "evidence_strength": "weak | medium | strong | unavailable",
  "recommended_action": "keep | low_weight | drop | unavailable"
}
```

#### 补充证据选择输出

```json
{
  "proposal_id": "same id",
  "selected_tools": [
    {
      "tool_name": "retrieve_same_class_image_crops",
      "arguments": {"top_k": 4},
      "reason": "compare with same-class candidates in this image"
    }
  ],
  "reason": "short string"
}
```

#### 最终决策输出

```json
{
  "proposal_id": "same id",
  "decision": "keep | low_weight | drop",
  "confidence": "low | medium | high",
  "reason": "short string",
  "used_signals": [
    "teacher_score",
    "visual_validity",
    "evidence_strength",
    "prior",
    "context"
  ]
}
```

### 5.3 约束与回退机制

| 机制 | 目的 |
|---|---|
| JSON 解析校验 | 避免自由文本直接进入后续流程 |
| 枚举字段校验 | 确保模型输出落在允许取值内 |
| VOC-only 类别校验 | 避免引入非任务类别 |
| proposal_id 一致性检查 | 防止模型输出错配候选框 |
| 保护规则 | 强视觉证据不能被后续决策随意推翻 |
| 确定性回退 | 模型调用失败、输出非法或校验失败时回退到规则逻辑 |

---

## 6. 实验结果

### 6.1 主结果

| 方法 | TP | FP | FN | Precision | Recall | F1 | mAP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `score≥0.20` 基线 | 124 | 559 | 74 | 0.1816 | 0.6263 | 0.2815 | 0.0850 |
| 多模态质量验证工作流 | 161 | 349 | 37 | 0.3157 | 0.8131 | 0.4548 | 0.1443 |

相较 `score≥0.20` 固定阈值基线，当前工作流：

- F1 从 0.2815 提升至 0.4548；
- mAP50 从 0.0850 提升至 0.1443；
- Recall 从 0.6263 提升至 0.8131；
- FP 从 559 降至 349，减少 210 个错误伪标签。

### 6.2 固定阈值扫描对比

![固定阈值扫描对比](figures/threshold_vs_mllm_metrics.png)

固定 score 阈值从 0.1 扫描至 0.9。蓝色曲线表示固定 score 阈值，红色虚线表示多模态质量验证工作流。结果显示，固定阈值方法只能通过提高阈值换取 precision，但会快速损失 recall；当前工作流在 F1、Recall 和 mAP50 上整体优于固定阈值扫描。

### 6.3 结果解释

固定阈值方法只利用 teacher detector 的置信度。当前工作流引入候选框级视觉证据和受限补充证据，可以针对具体候选框判断其视觉内容是否真实、类别是否匹配、位置是否合理。因此，提升不是来自某一个特殊阈值点，而是来自受控多模态证据流程。

---

## 7. 工具调用与运行统计

### 7.1 工具调用统计

| 工具 / 节点 | 调用次数 | 状态 |
|---|---:|---|
| `mllm_scene_prior` | 256 | success |
| `proposal_verification` | 1735 | success |
| `optional_tool_planner` | 1339 | success |
| `llm_final_decision` | 1735 | success |
| `build_class_exemplar_bank` | 1 | success |
| `resolve_duplicate_cluster_info` | 256 | success |
| `build_candidate_pool` | 256 | success |
| `retrieve_same_class_image_crops` | 1290 | success |
| `retrieve_class_exemplars` | 766 | success |

当前 256 图运行中没有 provider error、parse error 或 validation error；未触发由模型调用失败或输出非法导致的异常回退。非候选框会进入规则处理路径，不计为模型失败。

### 7.2 补充证据选择统计

| 选择结果 | 数量 |
|---|---:|
| 同图参考 + 源域参考 | 717 |
| 仅同图参考 | 573 |
| 仅源域参考 | 49 |
| 未触发补充证据选择 | 582 |

这说明系统并不是每次无脑调用所有补充证据，而是在工作流判定需要额外证据后，由模型在预定义范围内选择证据类型。

### 7.3 Token 与延迟统计

| 模型节点 | Calls | 输入 tokens | 输出 tokens | 总 tokens | 平均总 tokens | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 图像级先验 | 256 | 266,075 | 75,736 | 341,811 | 1,335.20 | 7,822.66 ms |
| 候选框视觉验证 | 1735 | 2,411,016 | 73,240 | 2,484,256 | 1,431.85 | 2,041.95 ms |
| 补充证据选择 | 1339 | 1,640,708 | 169,744 | 1,810,452 | 1,352.09 | 3,827.20 ms |
| 最终决策 | 1735 | 2,358,547 | 129,421 | 2,487,968 | 1,433.99 | 2,612.14 ms |
| 合计 | 5065 | 6,676,346 | 448,141 | 7,124,487 | — | — |

Token 统计来自本地 OpenAI-compatible 服务返回的 `usage` / `timings` 字段，按输入 tokens 与输出 tokens 分别汇总。确定性工具和检索工具记录工具调用轨迹，但不产生模型 token 消耗。

本地 MLLM 服务由局域网内另一台单卡 RTX 4090 服务器提供，部署模型为 `Qwen3.6-27B-UD-Q4_K_XL.gguf` 量化版本，使用 32k 上下文窗口，并通过 llama.cpp 提供 OpenAI-compatible 接口。

### 7.4 输出产物

当前运行输出包括：

```text
config.json
summary_metrics.json
proposal_results.csv
tool_call_summary.json
tool_call_trace.jsonl
image_level_trace.jsonl
image_traces/*.json
token_usage_summary.json
metric_diff_report.md
examples/success_drop_fp.md
examples/success_keep_tp.md
examples/fallback_case.md
same_class_refs/
```

这些产物用于定位每个候选框经历了哪些工具节点、模型输出是否通过校验、最终决策来自模型还是规则逻辑、离线评测中该候选框属于 TP/FP/FN/TN。

---

## 8. 案例追踪

### 8.1 高分错误候选被删除

```text
Teacher proposal
image_id = 0
proposal_id = 0
class = bird
score = 0.9531
bbox = [433.0, 211.0, 616.0, 385.0]

Proposal verification
visual_validity = invalid
evidence_strength = strong
recommended_action = drop
inspection_status = success

Final decision
llm_final_decision = drop
confidence = high
final_decision_source = llm_final_decision
fallback_used = false

Offline evaluation
gt_correct = false
eval_label = tn
```

该案例说明高分 teacher proposal 仍可能是错误伪标签。工作流通过候选框级视觉验证和受控最终决策将其删除。

### 8.2 真实目标被保留

```text
Teacher proposal
image_id = 1
proposal_id = 1
class = person
score = 0.5244
bbox = [2.0, 51.0, 568.0, 772.0]

Proposal verification
visual_validity = valid
evidence_strength = strong
recommended_action = keep
inspection_status = success

Final decision
llm_final_decision = keep
confidence = high
final_decision_source = llm_final_decision
fallback_used = false

Offline evaluation
gt_correct = true
eval_label = tp
```

该案例说明工作流并不是简单删除低置信或复杂候选，而是能够在视觉证据较强时保留真实目标。

### 8.3 非候选框的规则处理案例

```text
Teacher proposal
image_id = 3
proposal_id = 2
class = dog
score = 0.1672
bbox = [7.0, 683.0, 95.0, 684.0]

Proposal verification
visual_validity = unavailable
evidence_strength = unavailable
recommended_action = drop
inspection_status = not_candidate_heuristic

Final action
final_decision = drop
final_decision_source = deterministic_gate
fallback_used = true

Offline evaluation
gt_correct = false
eval_label = tn
```

该案例表示并非所有候选框都会进入 MLLM 节点。对于非候选框，系统使用规则逻辑处理，从而控制模型调用范围。

---

## 9. Prompt / Schema Showcase

### 9.1 图像级先验 system prompt 摘要

```text
You are a visual prior construction tool.
Return one JSON object only.
Do not make keep/drop decisions.
Do not use ground truth.
Use only allowed enum strings and VOC class names from the provided class list.
```

### 9.2 候选框级视觉验证 system prompt 摘要

```text
You are a proposal-level visual verification tool.
Return one JSON object only.
Do not use markdown.
Do not use booleans.
Do not use accept/reject.
Do not use ground truth.
visual_validity must be one of valid, uncertain, invalid, unavailable.
evidence_strength must be one of weak, medium, strong, unavailable.
recommended_action must be one of keep, low_weight, drop, unavailable.
```

### 9.3 补充证据选择 system prompt 摘要

```text
You are a bounded optional evidence tool planner inside a fixed pseudo-label QA workflow.
Return one JSON object only.
Do not use ground truth.
Select one or more tools only from the provided available_tools list.
These tools provide supporting evidence only and must not directly decide keep/drop.
```

### 9.4 最终决策 system prompt 摘要

```text
You are a constrained final decision tool inside a fixed workflow.
Return one JSON object only.
Use only provided evidence.
Do not use ground truth.
proposal_id must exactly match the provided proposal_id.
confidence must be low, medium, or high.
The inspection_result comes from VerifySingleBox_MM.
Its recommended_action is a tool recommendation, not a deterministic rule and not ground truth.
Make your own bounded decision from all provided evidence.
```

### 9.5 保护规则摘要

```text
1. 所有模型输出必须能解析为 JSON。
2. 所有枚举字段必须落在允许取值内。
3. proposal_id 必须与当前候选框一致。
4. 补充证据只能作为 supporting evidence，不能直接决定保留或删除。
5. 目标域真实标注不能进入任何模型消费输入。
6. 模型输出非法或调用失败时回退到规则逻辑。
```

---

## 10. 项目总结

本项目展示了如何在视觉训练数据质量控制场景中，把本地部署的多模态模型接入一个固定、受约束的工作流。系统不依赖模型自由规划完整流程，而是在图像级先验、候选框级视觉验证、补充证据选择和最终决策节点给予模型受限自由度，并通过结构化输出、保护规则和确定性回退控制模型行为。

从结果看，相比仅依赖 `score≥0.20` 的固定阈值基线，该工作流显著减少错误伪标签，并提升 F1、Recall 和 mAP50。该 demo 的核心价值不是提出一个新的检测模型，而是展示如何围绕实际视觉数据任务设计一个可控的 MLLM 工具流，使模型能力能够在明确边界内稳定参与数据质量验证。

### 后续扩展方向

- 将类别参考样本从固定检索扩展为 embedding-based 检索；
- 引入更细粒度的失败模式分类；
- 将候选框级验证扩展到分割 mask 或图像级标签验证；
- 将过滤后的伪标签接入实际训练流程，进一步评估对下游模型训练质量的影响。
