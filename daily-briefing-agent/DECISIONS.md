# DECISIONS

> 语言：中文 | 最后更新：2026-05-23 | 作者：Jordan Chen（由 MyFlicker AI 辅助实现）

---

## 1. 架构图

```
                        ┌─────────────────────────────────────────────────┐
                        │              Scheduler（定时触发）                │
                        └──────────────────────┬──────────────────────────┘
                                               │ 每日 07:30 触发
                    ┌──────────────────────────▼─────────────────────────┐
                    │             ProfileStore（LLM 预处理）               │
                    │  profile.json → 结构化 interest_tags / blocked_tags  │
                    │  entity_tag_map（别名 → canonical slug）             │
                    │  tone_style / audio_length / delivery_notes         │
                    └──────┬──────────────────────────────────────────────┘
                           │ 注入偏好上下文（ProfileContext）
          ┌────────────────▼────────────────────────────────────────┐
          │                   并发初筛层（Phase 1）                    │
          │                                                          │
          │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
          │  │ Calendar     │  │ Email        │  │ News         │  │
          │  │ FilterAgent  │  │ FilterAgent  │  │ FilterAgent  │  │
          │  │ 冲突检测      │  │ label 前处理  │  │ 规则预过滤    │  │
          │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
          └─────────┼────────────────┼────────────────┼────────────┘
                    │                │                │
                    └────────────────▼────────────────┘
                              FilterAgentOutput × 3
                              （candidates + excluded + agent_notes）
                    ┌────────────────▼────────────────────────────────┐
                    │          知识图谱构建（Phase 2）                   │
                    │  GraphBuilder                                    │
                    │  Event 节点 ──entity_tags──▶ EntityTagNode        │
                    │  Event 节点 ──topic_tags──▶  TopicTagNode         │
                    │  priority_score = topic_score + entity_bonus     │
                    │                            + flag_bonus          │
                    └────────────────┬────────────────────────────────┘
                                     │ KnowledgeGraph（含 briefing_date）
                    ┌────────────────▼────────────────────────────────┐
                    │    TopicSelector + RankingAgent（Phase 3）        │
                    │  Python 确定性：按 effective_weight 贪心选 Topic   │
                    │  LLM 精排：topic_priority / event_priority        │
                    │  + narration_hint / topic_summary                │
                    └────────────────┬────────────────────────────────┘
                                     │ RankingAgentOutput
                    ┌────────────────▼────────────────────────────────┐
                    │            WritingAgent（Phase 4）                │
                    │  session_history → 选择今日开头/结尾风格            │
                    │  LLM 生成 plain text + covered_event_ids         │
                    │  内部 trim retry（超字数时 temperature=0.3 重写）   │
                    └────────────────┬────────────────────────────────┘
                                     │ WritingAgentOutput
                    ┌────────────────▼────────────────────────────────┐
                    │           ValidationAgent（Phase 5）              │
                    │  Layer 1 硬检查（代码）：Markdown / URL / 数字符号  │
                    │                       / 字数 / 开头重复           │
                    │  Layer 2 软检查（LLM）：tone / 事实准确 / 覆盖度    │
                    │  失败 → retry_hint → WritingAgent 重写            │
                    │  最多重试 2 次，超限强制输出                        │
                    └──────┬──────────────────────────────────────────┘
                     不合格 │  retry_hint             │ 合格 / max retry
          ┌────────────────▼──────┐       ┌──────────▼───────────────┐
          │  WritingAgent 重试     │       │  最终输出（Phase 6）       │
          │  （带 retry_hint）     │       │  briefing.txt            │
          └───────────────────────┘       │  briefing.json           │
                                          └──────────────────────────┘
```

---

## 2. 关键设计决策

### 2.1 整体架构：先分后总多 Agent 流水线

**决策**：采用"并发初筛 → 图谱构建 → 精排 → 写作 → 检验"五阶段流水线，不使用单步 prompt。

**理由**：
- 单步 prompt 将 61 条输入全量塞入 LLM，上下文消耗大，过滤逻辑难以干预和调试
- 多步流水线每阶段有明确输入/输出契约（Pydantic 模型），失败可单独重试，不影响其他阶段
- 初筛层三个 Agent 并发执行，实测整体耗时约 60–90 秒（主要瓶颈是 5 次串行 LLM 调用）

---

### 2.2 知识图谱数据模型：Entity Tag vs Topic Tag 分离

**决策**：将 tags 拆分为两个独立字段：

| 字段 | 含义 | 举例 | 计分方式 |
|------|------|------|---------|
| `entity_tags` | 具名实体（公司、人、具体法规名） | `cobalt-labs`, `psd3`, `maya-chen` | 平坦加成（不放大 in_degree）|
| `topic_tags` | 语义领域 | `fintech-regulation`, `payments-product-launch` | `in_degree × base_weight`（放大跨源信号）|

**理由**：`cobalt-labs` 在几乎所有事件里都出现，如果用 in_degree 放大会导致每件事都评为 P0。分离后：
- topic 的 in_degree 表示"这个话题今天被几个来源同时提及"，是真实的重要性信号
- entity 只给一个固定加成，表示"我关注这个实体"，不因事件多而失控放大

**优先级公式**：
```
priority_score = Σ topic_tag.effective_weight   (topic_score)
              + Σ tracked_entity_bonus           (entity_bonus, flat)
              + Σ FLAG_BONUS[flag]               (flag_bonus)
```

---

### 2.3 时间轴与会议冲突检测

**决策**：CalendarFilterAgent 在 LLM 调用前做区间重叠检测，结果写入 `special_flags: conflict-detected`。

```python
# 区间重叠：若 a.end > b.start，则 (a, b) 冲突
sorted_events = sorted(events, key=lambda e: e["start"])
for i in range(len(sorted_events) - 1):
    if sorted_events[i]["end"] > sorted_events[i+1]["start"]:
        mark_conflict(sorted_events[i], sorted_events[i+1])
```

冲突信息注入 LLM prompt，WritingAgent 收到含 `conflict-detected` flag 的事件时必须明确说出冲突双方和时间。

---

### 2.4 筛选与排序

**三层漏斗设计**：

| 层 | 负责方 | 方式 | 输出 |
|---|------|------|------|
| 初筛（硬过滤） | FilterAgent（代码） | 来源黑名单、关键词正则、label 规则 | `EXCLUDE`（不进图谱）|
| 初筛（软过滤） | FilterAgent（LLM） | 语义判断 + profile 注入 | `DEPRIORITIZE`（进图谱但低权）|
| 精排 | TopicSelector（代码）+ RankingAgent（LLM） | 按 topic effective_weight 贪心选 Top-N | P0/P1/P2 分级 |

**"被丢弃"的信息在 briefing.json 里有三层记录**：
1. `excluded_by: filter_agent` — FilterAgent 硬排除
2. `excluded_by: ranking_agent` — RankingAgent 预算不足跳过
3. `excluded_by: writing_agent` — 在精排候选里但因字数压缩未提及

---

### 2.5 去重策略

同一事件被多个来源报道时（典型：PSD3 同时出现在新闻和邮件里），通过 topic_tag 聚合：

- `TopicSelector` 将每个 event 分配到 effective_weight 最高的 topic
- 同一 topic 下的多源事件一起传给 WritingAgent
- WritingAgent 负责在叙述中自然合并（"The PSD3 final text dropped overnight... Read Rahul's summary before the 3 PM compliance sync"——邮件 + 日程 + 新闻合为一段）

---

### 2.6 内容排序策略

| 优先级 | 标准 | 典型内容 |
|--------|------|---------|
| P0 | action-required + 当天 deadline / from-ceo | Board deck review、Plaid credential rotation |
| P1 | tracked-entity + 直接关联今日日程 | PSD3 合规 sync、Stripe 合作午餐 |
| P2 | 高权重 topic，无当天 deadline | Cobalt GA 新闻、Lyra 竞品动态 |

**段落结构**（动态，非固定章节）：
1. 开头（1–2 句）：今日 opening style（每天变化）
2. P0 topics 按 effective_weight 降序，事件按 P0 → P1 → P2
3. P1 topics：压缩到 1–2 句
4. P2 topics：仅在字数允许时加入（一句话）
5. 收尾（1 句）：today's closing style（每天变化）

---

### 2.7 时长控制

**量化基准**（基于 TTS 朗读测试）：
- 正常英文 TTS 朗读速度：约 150 词/分钟 = 2.5 词/秒
- 目标 75s ≈ 187 词；可接受范围 60–90s = 150–225 词

**控制机制（双重保险）**：
1. WritingAgent prompt 明确告知词数上下限，LLM 生成时自我控制
2. WritingAgent 内部 trim retry：首稿超出 max_words 时，以 temperature=0.3 发送压缩指令，强调删 P2、压缩 P1
3. ValidationAgent Layer 1 硬检查词数，超限输出到 retry_hint，触发下一轮重写

---

### 2.8 TTS 友好度

**代码层强制检查（ValidationAgent Layer 1）**：
- 正则检测 Markdown 标记（`#`, `*`, `_`, `- `, `1. ` 等）
- 正则检测 URL 和 email 地址
- 正则检测未展开的数字符号：`%`, `$`, `€`, `K/M/B` 缩写

**LLM prompt 层规则**（WritingAgent system prompt）：
- 数字口语化：`"12%"` → `"twelve percent"`，`"$4B"` → `"four billion dollars"`
- 缩写展开：`"PSD3"` → `"PSD Three"`，`"GA"` → `"general availability"`
- 禁用开头词：So / Well / Actually / Basically

---

### 2.9 Agent 解耦与状态管理

**当前实现**：所有 Agent 是直接函数调用，状态通过 Pydantic 对象在内存中按顺序传递。无消息队列，无状态数据库。每次运行完全独立，无持久化中间状态（`session_history.json` 是唯一跨运行状态，仅用于开头风格轮换）。

**取舍**：简单直接，适合单次运行场景；代价是无法断点续跑、无法并行调度 Phase 3 以后的阶段。

---

### 2.10 重试与兜底机制

**已实现的重试逻辑**：

| 层级 | 触发条件 | 重试方式 | 上限 |
|------|---------|---------|------|
| WritingAgent 内部 | 首稿超出 max_words | trim retry（temperature=0.3，对话续写）| 1 次 |
| Validation → Writing | ValidationAgent 发现 error 或 LLM 建议 revision | 带 retry_hint 重新调用 WritingAgent | 2 次 |

**兜底**：`MAX_WRITE_RETRIES=2` 到达后，无论通过与否都强制输出当前草稿到 `briefing.txt`。briefing.json 的 `validation.passed` 字段记录最终状态，`validation.issues` 记录未解决的问题。

---

### 2.12 近几日风格存储

**实现**：`inputs/session_history.json` 记录最近 30 天每日的：
- `opening_style`（从 6 种风格中选）
- `closing_style`（从 5 种风格中选）
- `opening_fragment`（开头第一句，用于检测重复）
- `closing_fragment`（结尾最后一句）

**选择逻辑**：每天从可用 style 中排除最近 3 天使用过的，保证轮换。ValidationAgent 同样检查开头 60 字符是否与近期 fragment 重复，有重复则加入 retry_hint。

---

## 3. AI 工具使用记录

本项目使用 **MyFlicker（AI 编程助手）** 全程辅助开发，以下是真实的协作过程记录：

### 方案设计阶段（我主导）
- 独立阅读 inputs/ 数据，识别关键挑战：entity 标签被 in_degree 放大、新闻量过多、token 压力
- 主导决策：先分后总流水线架构、entity/topic tag 分离、三层漏斗过滤设计
- AI 角色：提供选项 A/B/C 供选择，解释不同方案的 trade-off，我做最终决定

### 代码实现阶段（AI 主导，我审查）
- AI 按设计文档逐 Agent 生成 Python 代码（ProfileStore → CalendarFilterAgent → EmailFilterAgent → NewsFilterAgent → GraphBuilder → TopicSelector → RankingAgent → WritingAgent → ValidationAgent）
- 每个 Agent 实现后立即跑测试，我审查输出质量和覆盖逻辑

### 发现并纠正的典型问题

**问题 1：Entity tag 污染 topic 评分**
- 现象：所有事件都评为 P0，因为 `cobalt-labs` in_degree=9，放大了每个事件的权重
- 我的判断：这是设计缺陷，不是 bug，需要重新设计数据模型
- 修复：主导拆分 `entity_tags` / `topic_tags`，AI 实现 schema 变更和所有相关代码

**问题 2：NewsFilterAgent 误排除有价值新闻**
- 现象：`news_024`（OpenAI × Walmart 企业 AI）因 `Walmart` 关键词被误排除；`news_015`（GCP 宕机）因 `Google Cloud outage` 被排除
- 我的审查：逐条对照用户 profile 判断误删，重新设计更精准的 pattern（要求 company + corporate-event 共现）
- AI 角色：实现修改后的 pattern

**问题 3：ValidationAgent 信息不对称导致误判**
- 现象：ValidationAgent 将 `em_001` 里的 "ARR projection chart" 判为幻觉；无法识别 `em_016` 里虚构的 deadline
- 根因：我发现 `ev.summary[:100]` 截断——`em_001` 的 summary 是 134 字符，`ARR projection chart` 在第 101 位之后
- 修复：改为全量 summary，加 `narration_hint`，prompt 明确说明 summary 完整

### 回顾
- AI 写代码很快，但核心设计判断（tag 分层、过滤策略、信息差异定位）需要人来主导
- AI 在 prompt 细节上有时会"过度遵守"某个规则导致新问题（如 entity-spotlight 风格导致开头总是 Cobalt GA）
- 最有价值的调试方法：用代码直接打印每层的输入/输出，而不是只看最终结果

---

## 4. 已知限制

1. **验证准确性**：ValidationAgent 的事实准确性检查依赖 LLM，无法保证 100%。已知弱点：当 narration_hint 和 summary 存在细微矛盾时，LLM reviewer 倾向于保守地报告 warning，可能导致有效内容被过度 retry。

2. **生成时间较长**：完整链路约需 60–90 秒（5 次 LLM 调用：ProfileStore + 3× FilterAgent 并行 + RankingAgent + WritingAgent + ValidationAgent），若触发重试最多可达 5 次 LLM 调用。不适合实时响应场景。

3. **输出不稳定性**：相同输入在不同运行间，因 LLM temperature > 0，会产生不同的 topic_priority 分配和叙述角度。有时 effective_weight 相近的两个 topic（如 `platform-engineering` eff=9.6 vs `action-item` eff=6.0）的优先级顺序会因当天 LLM "心情"不同而互换，导致一个 topic 的事件被完整提及，另一个被压缩或遗漏。

4. **低权重 topic 遗漏风险**：TopicSelector 贪心策略基于 effective_weight 选 topic，权重相近但事件密度低的 topic 会被挤出预算。例如 `board-prep` 只有一个事件，在 22 事件预算下容易被排在后面，导致相关事件未进入 WritingAgent 的输入集合。

5. **session_history 无外部持久化**：`inputs/session_history.json` 是普通文件，跨部署或环境重置时会丢失，需手动备份。

---

## 5. 如果再多两小时，我会做什么

1. **Agent 状态解耦**：将 FilterAgentOutput / KnowledgeGraph / RankingAgentOutput 序列化到文件（JSON），支持断点续跑和单阶段重跑，便于调试和测试。

2. **偏好动态更新**：当前每次运行都重新调用 ProfileStore LLM。可以对 profile.json 做 hash 缓存，未变化时复用上次解析结果，节省约 1 次 LLM 调用。

3. **WritingAgent 输出 covered_event_ids 可靠性**：当前依赖 LLM 自我报告，准确率约 85–90%。改进方向：对文本做关键词匹配（event title 中的专有名词 / 时间词），代码层交叉验证 LLM 报告的 id 是否真实出现在文本中。

4. **低权重 topic 兜底**：对含 `action-required` flag 的事件，即使其所属 topic 未被选入 TopicSelector，也强制单独作为 overflow 事件发给 RankingAgent，避免重要行动事项遗漏。（TopicSelector 已有 `overflow_events` 机制，需要强化触发条件。）

5. **ValidationAgent pass 标准调整**：当前只有 0 error 且 LLM verdict=pass 才通过，实际上 1–2 条 `tts_flow` warning 不影响最终质量，可以放宽为"0 error + LLM verdict ≠ needs_revision"即通过，减少无效重试。

6. **精细化排序与模型训练**：当前 topic effective_weight 基于规则打分，无法感知"今天这条新闻对这个用户真正有多重要"。改进方向：收集用户对历史简报内容的隐式反馈（如跳过、重播、标记），构建偏好标注数据集，训练一个轻量 ranking 模型（如 pairwise LTR 或 fine-tuned embedding similarity），替代现有规则公式，实现更准确的个性化内容召回与排序。

---

## 附：briefing.json 字段说明

| 字段 | 说明 |
|------|------|
| `briefing_date` | 简报日期，从 calendar.json 时间戳推断，非系统时钟 |
| `generation_timestamp` | 生成时刻（UTC） |
| `word_count` | 实际词数（按空格分割）|
| `estimated_seconds` | word_count / 2.5（TTS 150 wpm）|
| `estimation_method` | 估算方法说明 |
| `sections` | 章节切分：name / char_start / char_end（从文本 anchor 搜索计算）/ covered_event_ids |
| `covered_calendar_ids` | WritingAgent 实际提及的日程 id |
| `covered_email_ids` | WritingAgent 实际提及的邮件 id |
| `covered_news_ids` | WritingAgent 实际提及的新闻 id |
| `excluded_items` | 三层丢弃记录：`excluded_by` 字段区分 filter_agent / ranking_agent / writing_agent |
| `validation.passed` | 最终是否通过全部检查 |
| `validation.issues` | 未解决的 warning 列表（不影响输出，仅供审查）|
| `pipeline_stats` | 各阶段数量统计，供调试用 |
