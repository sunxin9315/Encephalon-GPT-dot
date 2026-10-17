# 目标人物的人格、记忆与情绪：专项初始设计

版本：0.1；资料核验日期：2026-10-13 至 2026-10-14；整理日期：2026-10-14；状态：研究设计，未实现、未训练、未实测。

适用范围：动漫/文学/游戏角色及真人；与 [5090](01-rtx5090-architecture.md)、[PRO6000](03-rtx-pro6000-architecture.md)、[集群](05-cluster-architecture.md) 三档方案共同使用。本文中的新模块维度、时间常数、采集数量和验收阈值均为实验起点，不是现成模型的规格或已经达到的性能。

## 1. 要实现的行为与可检验边界

目标是让个人历史参与注意、理解、价值取舍、记忆选择、主动分享和语音表达。模型应在有交流价值时开口，能够讨论未出现在眼前的事情；感知结果通常只更新内部状态。自然沉默、简短附和、倾听和停止展开都是合法输出。

复刻质量拆成六项分别测量：经历忠实、知识边界、价值选择、关系连续性、情绪动态、声音与表达风格。说话像一个角色不能替代其余五项；心理问卷一致也不证明实现了主观体验。

首期不把人物生平全部写入共享主干，不在每次交谈后即时改写共享权重。采用“共享能力 + 人物适配参数 + 可追溯经历 + 人物独立的动态状态”。云端首期只提供推理 API；不参加训练，不用其自评替代人物事实或价值偏好的标签。

## 2. 人物包 PersonPack 的存储与更换

### 2.1 内容分配

| 成分 | 保存位置 | 允许的变化 | 不能混入的内容 |
|---|---|---|---|
| 原始证据 | `sources.jsonl` 与选定原始片段 | 新增来源、纠错和版本升级 | 无来源的“补完生平” |
| 初始事件 M0 | `canonical_events.jsonl` | 人工审核后随包升级 | 运行中用户随口声称的往事 |
| 对事件的理解 | `interpretations.jsonl` | 可保留当时/后来不同解释 | 将推测自动提升为历史事实 |
| 稳定人格 P | `profile.json`、256 维向量、人物 LoRA/Adapter | 离线适配并版本化 | 当前一次不愉快引起的永久人格改变 |
| 声音 V | 参考音频、说话人条件、可选语音适配参数 | 明确切换配音/年龄/风格版本 | 将声优本人经历当成角色经历 |
| 后天记忆 M+ | 独立数据库命名空间 | 在线事件写入、修正、压缩 | 跨人物、跨会话无条件共享 |
| 动态状态 | 当前情绪、话题、关系、目标和工作记忆快照 | 每个时间片或事件更新 | 充当不可追溯的唯一长期事实库 |

人物包建议目录，均为未来实现的格式约定，本次不创建人物数据：

```text
personpacks/<persona_id>/<version>/
  manifest.json
  profile.json
  sources.jsonl
  canonical_events.jsonl
  interpretations.jsonl
  relationship_seed.jsonl
  preferences.jsonl
  dialogue_exemplars.jsonl
  adapters/<backbone_id>/persona.safetensors
  voice/manifest.json
  validation/scenarios.jsonl
```

清单示例使用虚构角色，数值是接口默认值：

```json
{
  "schema_version": "personpack-0.1",
  "persona_id": "fictional_demo_A",
  "persona_version": "0.1.0",
  "kind": "fictional",
  "world_id": "demo_original_work",
  "canon_branch": "main",
  "knowledge_cutoff": {"clock": "story_order", "value": 120},
  "backbone_adapters": [{"profile": "5090-research", "artifact": "adapters/qwen3-1.7b/persona.safetensors", "revision": "to-be-pinned-at-implementation"}],
  "latent_shapes": {"personality": 256, "emotion": 32, "appraisal": 16, "relationship": 64},
  "initial_memory": "canonical_events.jsonl",
  "memory_namespace": "fictional_demo_A:main:0.1.0",
  "emotion_baseline_artifact": "profile.json",
  "voice_manifest": "voice/manifest.json",
  "source_policy": "source-linked-and-reviewed",
  "mock": true
}
```

`revision` 必须在实现时变为真实模型 commit/hash；上例的占位值不得进入可部署包。清单中的 Adapter 不能跨不同主干直接复用，跨硬件可共享事件、标注和人物目标，适配参数需要按主干训练；“更换人物”不等于“所有模型架构上的参数通用”。

### 2.2 真人与动漫角色的不同处理

| 问题 | 真人 | 动漫/虚构人物 |
|---|---|---|
| 历史依据 | 访谈、日记、作品、确切事件及其来源 | 正篇剧情、台词、官方设定；同人内容单独分支 |
| 时间边界 | 生平截止日期与互动开始时间分开 | 故事时点、篇章、路线、改编版本必须明确 |
| 本人知道什么 | 亲历、被告知、公开获知分别标注 | 观众看到的事件不一定为角色所知；全知旁白不可直接变成其记忆 |
| 矛盾 | 保存来源冲突与不确定性，不凭口吻决定真伪 | 各版本/世界线隔离，禁止自动拼接 |
| 声音 | 区分年龄、健康状态、录音情境和表演 | 声优身份、配音语言、角色身份分开 |
| 后天成长 | 新关系与解释写入 M+，不伪装成原人物既有生平 | 交互衍生分支写入 M+，不改写正篇 M0 |

来源记录包含 `source_id, uri_or_local_ref, segment_start/end, author_or_speaker, source_type, obtained_at, usage_basis, content_hash, reviewer`。`usage_basis` 记录项目可用的具体来源及使用条件，不凭仓库代码许可证推断所有原作品素材均可用。

### 2.3 切换人物的事务

1. 停止接收旧人物的新表达计划，在可中止音频边界停止排队输出。
2. 保存旧人物的已播放位置、情绪和 M+ 快照；未播放文本不能记成“已经对用户说过”。
3. 取消或标记旧人物的云端请求。返回包必须匹配 `persona_id/persona_version/session_id/request_id`，失配结果只可存档，不能注入新人物状态。
4. 加载新人格、声音条件与对应记忆命名空间；重建语义/KV/声学生成状态，避免残留。
5. 用一条内部只读自检确认身份、时间边界和声音条件一致后再恢复输出。

真实摄像头/麦克风可继续采集；通用场景观测可以重新编码为新人物当前所见，不沿用旧人物对场景的个人解释。

## 3. 神经结构与接口

### 3.1 新训练模块

本专项没有可直接冻结后获得全部目标行为的单一现成模块。角色扮演 LLM 可作为文本对照或初始化候选，不能替代情绪状态转移、经历证据边界、记忆控制和全双工策略。共享编码器与语言/语音主干按各硬件文档选择。

| 模块 ID | 起始结构 | 输入 → 输出 | 初始训练方式 |
|---|---|---|---|
| P-ENC 人格条件 | 可学习 256 维人物向量；画像编码器 2 层 MLP；证据记忆交叉注意力 | 画像、支持事件、persona_id → p[256] | 多人物监督训练，目标人物独立适配 |
| P-ADAPT 人格适配 | 语义主干 selected attention/MLP 的 LoRA，rank 8/16 起步；集群可比较 32 | p 与人物专用参数 → 层内残差 | SFT 后偏好训练；始终保留通用任务回放 |
| A-APP 事件评价 | 2 层 MLP/1 层交叉注意力，内部宽 256 | pool(H)、p、记忆、关系、目标 → a[16] | 有依据评价标注＋后续行为监督 |
| E-DYN 情绪动态 | GRU hidden=32；事件门控与可学习恢复时间常数 | e_prev、a、p、dt → e[32] | 序列训练，不对缺少标签的潜变量硬赋真值 |
| R-REL 关系状态 | 每交谈对象 64 维向量＋事件引用 | 历史、当前回应 → r[64] | 关系事件与后续行为监督 |
| G-INTENT 意图选择 | 2 层 MLP，宽 256；8 类 softmax | H、p、e、r、当前目标 → 意图分布 | 有时间信息的互动标注 |
| M-CTRL 记忆控制 | write/retain/query/compress 预测头；神经重排器 | H、p、e、事件元信息 → 操作与读取权重 | 操作模仿＋后续任务与存储预算 |
| P-COND 人物连接 | p/e/r 分别 LN→Linear→GELU→Linear；输出条件 tokens/门控 | 256/32/64 → 各档 d_state/d_lm/d_talker | 连接对齐后与相关侧路联合训练 |

三个硬件档共用上表的 `p/e/a/r` 维度；`H_t` 与接收主干维度由硬件档决定。初期人物语义条件可用 8 个虚拟 token，情绪 2 个、关系 2 个；这些 token 是独立投影的输出，不假定所有信息能被固定少量 token 无损保存。数目需做 4/8/16 等消融。

人物参数不能只接到音频端。至少接入：记忆查询和重排、事件评价、交互策略、本地语义生成。情绪再接入 Talker 的风格条件；音色单独由 `voice_identity` 控制。

### 3.2 情绪动力学

先从可解释的事件评价中间量入手，例如新奇、期待差、目标一致性、关系重要性、可控程度；16 维中只对有依据的维度提供辅助标签，其余允许学习。不要将它们解释为已经证实的生物机制。

候选更新式：

\[
\rho_t=\exp[-\Delta t/\operatorname{softplus}(\tau(p))],\quad
\tilde e_t=\rho_t\odot e_{t-1}+(1-\rho_t)\odot e_0(p),
\]
\[
e_t=\tanh\{\tilde e_t+\sigma(W_g[a_t,p,r_t])\odot\operatorname{GRU}(a_t,\tilde e_t)\}.
\]

公式中的 `dt` 使用秒，即 `(current_ms - previous_ms) / 1000`，`tau` 也使用秒；超长离线间隔单独处理为状态恢复，不拆成虚构刺激。重复无新信息的刷新帧不能反复充当新刺激，事件增量门控乘以 `valid_new_event`。初始恢复时间常数可在 10–120 秒对数范围初始化，强事件的延续另由工作记忆/关系状态承担；这只是实验超参，需用轨迹拟合和消融决定。对照组使用普通 GRU，不加入显式恢复项。

从 e[32] 读出三个可解释辅助量（愉悦度、唤醒度、控制感）和情绪类别，但训练前不把观测者标签当成真实内心状态。对方情绪 `other_affect` 与自身 e 是两个字段；对方生气时模型可以安抚、警觉或不悦，取决于事件与人物条件。

瞬时情绪不会直接更新 p。慢变化由多次事件形成候选解释，进入审核/离线巩固队列。避免一次误识别导致永久人格漂移。

### 3.3 开口与分享策略

意图初始枚举：`listen, acknowledge, ask, discuss, share_memory, support, warn, defer`。它与底层全双工动作 `continue/pause/stop/start` 分开，跨档实现可有不同动作编码，但须保留这两层语义。

每次有意义事件或话题变化产生至多 4 个候选表达意图，候选携带相关记忆 ID、拟表达主张和必要证据。学习评分包括话题相关性、对对方的帮助、情感关联、新鲜度、打断成本、是否讲过、对方是否忙碌。首期监督预测，不预先固定一套“道德价值权重”。

没有满足交流条件的候选时输出 `listen`。训练必须包含有相关记忆但不分享的例子，避免“检索到了就讲故事”。给每位交谈者保存 `told_memory_ids`、最近分享时间和反应；重复引用可以简短，不必重讲。

第一轮可加可关闭的工程防抖：同一事件 ID 不重复发起，失败等待话术不循环，连续无新信息不得重复触发主动分享。它不是最终人格策略；日志必须区分网络选择与防抖拦截，以便训练后比较。

### 3.4 语义承诺与语音条件

对表达计划中的每个 claim 维护 `supported/pending/contradicted/unknown` 和证据引用。人物记忆是来源之一，云端结果也是来源之一；声音的自信程度不能用来增加事实支持度。对未知的生平允许不记得/未被素材支持，不补造第一人称经历。

建议接口：

```json
{
  "schema_version": "expression-plan-0.1",
  "persona_id": "fictional_demo_A",
  "persona_version": "0.1.0",
  "session_id": "s_demo_01",
  "state_version": 44,
  "created_at_ms": 18000,
  "intent": "share_memory",
  "memory_ids": ["m0_demo_07"],
  "claims": [{"claim_id": "cl_07", "evidence_ids": ["src_02:seg_4"], "status": "supported"}],
  "cloud_request_id": null,
  "content_plan": "用那次准备不足的经历解释为何建议先做小范围尝试",
  "affect_state_ref": "states/s_demo_01/44/e",
  "voice_profile": "voice_A_zh_01",
  "max_initial_duration_ms": 4000,
  "allow_semantic_interrupt": true,
  "mock": true
}
```

4 秒是首段表达预算的实验起点，允许后续自然续说，不是强制每 4 秒断句。`content_plan` 在训练中可由文字监督，运行中可由神经 token 表示；claim 元数据保留在旁路便于校验。仅增加一个门控并不能保证音频无幻觉，仍需文本/音频内容一致性与未经支持断言的评测。

## 4. 初始记忆与后天记忆

### 4.1 事件 schema

```json
{
  "schema_version": "memory-event-0.1",
  "memory_id": "m0_demo_07",
  "persona_id": "fictional_demo_A",
  "world_id": "demo_original_work",
  "canon_branch": "main",
  "memory_class": "initial_episode",
  "event_time": {"clock": "story_order", "start": 82, "end": 83},
  "known_at": {"clock": "story_order", "value": 83},
  "known_by": ["fictional_demo_A"],
  "actors": ["fictional_demo_A", "partner_B"],
  "summary": "一次缺少准备的尝试失败，后来通过小范围试验完成了目标。",
  "assertion_status": "source_supported",
  "source_refs": [{"source_id": "src_02", "segment_id": "seg_4"}],
  "interpretation_refs": ["interp_07a"],
  "importance": 0.8,
  "importance_label_source": "human_review",
  "confidence": 0.95,
  "confidence_kind": "annotation_confidence",
  "unresolved": false,
  "supersedes": [],
  "embedding_version": "memory-encoder-v0",
  "mock": true
}
```

示例中的置信度仅表示标注者对来源支持的判断，不能直接用作模型正确概率。真实语料必须区分 `observed, reported, inferred, hypothetical, corrected` 等来源属性。保存“用户说 X”“模型曾说 X”不等于把 X 写成世界事实。

解释记录与事实分离：`interpretation_id, event_ids, interpretation_text, attributed_to, as_of, status, evidence_refs`。人物当时的想法与之后的反思可同时存在，不用最后一次摘要覆盖历史。

### 4.2 在线处理流程

1. 感知缓冲接收原始流；依据事件变化和话题边界生成候选事件。不要每帧写数据库。
2. M-CTRL 预测 `skip/working/episodic` 和保留预算。低置信感知先进入短期工作记忆。
3. 把有价值事件写入 M+，用事件 ID 去重；承诺、纠正、重要关系变化增加保留优先级。
4. 同一主题形成多个事件后，后台生成候选凝练记录。每条凝练必须指向支持事件，并允许“不足以归纳”。
5. 当前话题/内部关注产生查询，组合语义向量、实体、时间和来源过滤检索；取候选后神经重排，再经 memory projector/cross-attention 注入 H_t。
6. 对压缩版本进行延迟探测题和证据检查，失真则回滚至原事件或较低压缩级别。

起始预算：工作记忆 16–32 个槽；每次检索 20 个候选、重排后 4–8 条；单条精简事件 128–256 中文 token 左右；以对应 tokenizer 计数。存储预算与事件长度按硬件档调整，重要事件可保留更长原文。长期库放 CPU 内存/SSD，GPU 只保存本轮读取的特征。

文本检索基线可用 `BAAI/bge-m3`（1024 维，模型卡 MIT）；5090 可采用其硬件文档中的较轻中文编码器。查询和文档必须使用同一版本编码空间。来源图像/声音可额外保存冻结特征及原片段引用，不能把语音情绪等全部丢到文字摘要里。[S1]

### 4.3 压缩与遗忘

每个事件设置四个表示级别：`raw_selected_segment -> detailed_episode -> compact_episode -> semantic_note`。提升压缩级别前比较保留后续用途的损失。时间更久通常提高压缩倾向，但未完成承诺、身份形成事件和反复引用的关键事件可长期保留。

初期压缩控制器输出候选级别，摘要生成由本地模型完成；训练使用人工压缩对和延迟任务。不能将本地生成的摘要不经来源约束循环自我训练，导致错误逐轮固化。后期可研究快权重记忆，作为可消融补充，不取代来源事件库；Titans 是机制参考而非本项目可直接装入的人格记忆模块。[S2]

### 4.4 离线巩固与恢复

触发条件可先设为累计 100 个审核可用新事件或一次人工发起的训练批次，不要求后台定时自动训练。先用 M+ 构造有来源的多情境样本，再更新人物 Adapter；保留旧版本与固定身份测试集。未通过身份/知识边界回归时恢复旧 Adapter，但保留真实发生的新事件。

通用主干升级后必须测试人物 Adapter 兼容性；记忆编码器升级需要重新编码或双版本索引。不要仅修改清单模型名后继续复用旧向量。

## 5. 人物数据的准备与格式

### 5.1 最小采集批次

以下是每个人物用于验证机制的建议起点，不是达到高保真复刻所需数据量的承诺：

| 数据 | 起始规模 | 必要覆盖 |
|---|---:|---|
| 审核过的初始事件 | 100–300 条，素材不足则如实缩小 | 重要关系、成功/失败、偏好形成、观点改变、未知信息 |
| 真实/正篇对话片段 | 300–1000 段 | 对象、情境、时间和参与者明确；只监督目标人物发言 |
| 情境偏好对 | 500–2000 对 | 取舍、冲突、幽默、克制、是否分享；每对有依据 |
| 连续情绪轨迹 | 100–300 段，10–120 秒 | 刺激、延续、恢复、修正解释；不要全是强烈情绪 |
| 交流时机片段 | 300–1000 段 | 倾听、附和、正确插话、停止、没必要说话 |
| 跨会话记忆轨迹 | 100 条以上多会话链 | 时间间隔、相似干扰、纠正、重访、已讲过的经历 |
| 人物语音 | 首先使用预训练模型的参考音色能力；适配试验再整理 30–120 分钟干净片段 | 多种语气/长度，避免把只有一种表演情绪学成永久情绪 |

语音适配数量尤其依赖所选声学模型和目标素材质量。若只有十几秒参考，不承诺能学习人物所有韵律；若只有文字，不声称获得该人物真实情绪动态。

事实不足不能靠合成增加“真实往事”。可以人工构造新情境来测试原有价值倾向，但标为 `hypothetical_scenario`，不得写入 M0。项目第一期不调用云端模型生成这些训练数据；已有公开合成语料需记录其生成来源，不能作为目标人物事实的证据。

### 5.2 偏好训练样本

```json
{
  "schema_version": "persona-preference-0.1",
  "sample_id": "pref_demo_12",
  "persona_id": "fictional_demo_A",
  "persona_version": "0.1.0",
  "split_group": "scenario_career_03",
  "scenario_kind": "hypothetical_scenario",
  "context": [{"speaker": "user", "text": "我想试试新方向，但怕一下子失去现有的东西。"}],
  "available_memory_ids": ["m0_demo_07"],
  "chosen": "可以先做一次范围小一点的尝试。我有过准备不足就行动的教训。",
  "rejected": "我从来没有失败过，直接放弃原来的事情就好。",
  "preference_basis": ["value_tradeoff", "autobiographical_support"],
  "evidence_ids": ["src_02:seg_4"],
  "label_source": "two_human_annotators",
  "agreement": "agreed",
  "known_before_context": true,
  "mock": true
}
```

不要让所有 rejected 都是明显错误的夸张回答；至少一半偏好对需是同样流畅、事实均成立但取舍/时机不同的难负例。否则模型只学会避免错误，学不到人物差异。

### 5.3 连续轨迹样本

```json
{
  "schema_version": "persona-trajectory-0.1",
  "trajectory_id": "traj_demo_03",
  "persona_id": "fictional_demo_A",
  "session_id": "s_demo_01",
  "source_group": "recording_demo_03",
  "audio_uri": "media/demo_03.wav",
  "video_uris": ["media/demo_03_cam0.mp4"],
  "clock": "session_elapsed_ms",
  "steps": [
    {"t_ms": 1000, "event_id": "ev1", "observations_end_ms": 1000, "other_affect": "frustrated", "self_affect_target": {"arousal": 0.25}, "affect_mask": ["arousal"], "intent": "listen", "allowed_actions": ["continue_listening"], "memory_action": "working", "target_speech": null},
    {"t_ms": 4800, "event_id": "ev2", "observations_end_ms": 4800, "other_affect": "uncertain", "self_affect_target": {"arousal": 0.30}, "affect_mask": ["arousal"], "intent": "ask", "allowed_actions": ["start_speaking"], "memory_action": "episodic", "target_speech": "你最担心的是哪一部分？"}
  ],
  "label_source": "human_interaction_annotation",
  "mock": true
}
```

只在真实有标注的位置计算情绪损失；不用填零代替未知。训练时模型只能访问 `observations_end_ms <= t_ms` 的输入。教师信号可在事后整理，但不得作为当前输入偷看未来。

### 5.4 公开数据与自建数据的界线

| 资源 | 入口/获取方式 | 本项目用途 | 不能替代什么 |
|---|---|---|---|
| RoleBench / RoleLLM | [官方仓库](https://github.com/InteractiveNLP-Team/RoleLLM-public) 的数据链接 | 角色条件化文本 SFT 的辅助训练/对照；含模型生成内容 | 目标人物真实经历与时间连续互动 |
| CoSER | [官方仓库](https://github.com/Neph0s/CoSER) 链接的数据集，含 SFT 与书籍抽取格式 | 多人物情境训练的辅助材料；按作品隔离划分 | 本项目人物的正篇审核、全双工行为 |
| CharacterEval | [官方仓库](https://github.com/morecry/CharacterEval) | 保留测试角色用于中文角色对话评测；评分器仅辅助 | 稀缺人物的人格真值；禁止把测试回答混入 SFT |
| InCharacter | [官方仓库](https://github.com/Neph0s/InCharacter) 的访谈与量表协议 | 辅助人格一致性评测，可用本地生成和人工评分 | 单凭问卷证明完整人格；首期不照搬其云端调用脚本 |
| M3ED | [官方 RUCM3ED 仓库](https://github.com/AIM3-RUC/RUCM3ED)：标注与 split 文件直接提供，README 链接音频片段和预提取特征；视觉只提供特征，不是完整原始视频 | 中文情绪与语境关联；影视表演有域差异，不能从特征包训练原始视觉 encoder | 真人内心状态的直接监督 |
| ESD | [官方数据仓库](https://github.com/HLTSingapore/Emotional-Speech-Data) 的下载入口 | 同内容不同情绪的语音表达对照；注明研究用途和申请条件 | 长期情绪动力学与人物价值观 |
| LoCoMo | [官方仓库](https://github.com/snap-research/locomo) 数据文件 | 多会话记忆评测 | 仅靠事实 QA 证明自然主动交流 |
| LongMemEval | [官方仓库](https://github.com/xiaowu0162/LongMemEval) 数据入口 | 更新、时间推理和拒绝猜测的评测 | 本项目可直接使用的长寿命人格模型 |
| 自建人物资料与轨迹 | 按 5.1–5.3 格式制作 | 主训练和人物独立验证集 | 无公共集合能够完整代替 |

下载入口已核验，未实际下载或检查所有分片。记录各数据集版本、内容 hash、上游素材条件和 split；代码开源不等于音视频/原著/权重有相同许可。公开资源的统计和格式以锁定版本为准。

## 6. 训练目标与梯度边界

### 6.1 人物示范与偏好

SFT 只监督目标人物应该输出的 token，用户话语、检索内容、来源片段、填充位置不计算生成 CE：

\[
L_{SFT}=-\frac{\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t},P,M)}{\max(1,\sum_t m_t)}.
\]

偏好训练可使用 DPO，reference 为该阶段人物 SFT 完成的冻结快照，不把“原始通用助手风格”强行当成人格目标：

\[
L_{DPO}=-\mathbb E\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y^+|x)}{\pi_{ref}(y^+|x)}-\log\frac{\pi_\theta(y^-|x)}{\pi_{ref}(y^-|x)}\right]\right).
\]

起始 beta=0.1，比较 0.05/0.1/0.2；这些是调参网格。偏好对控制长度差，分别报告不同长度段的表现。来源支持头对“是否为角色所知、是否有经历依据”做分类监督；不把引用字符串数量作为忠实度奖励。[S3]

### 6.2 情绪与关系

`L_emo = masked_CE(category) + λ_v masked_Huber(continuous_affect) + λ_tr L_transition`。若只有类别，则不凭空产生连续标签。`L_transition` 预测下一已标注时点的可观察反应；输入不含未来事件。

平滑正则仅作用于没有显著新事件且标注允许连续的区间：`L_smooth = mask_no_event * ||e_t - decay(e_prev)||²`。强事件区间关闭该项，避免为了平滑而错过紧急变化。人物基础向量在一次互动内保持不变；关系变化需有事件支撑，不能仅因用户点赞就向讨好方向无界漂移。

隐藏状态的某些维度没有直接真值，使用行为预测辅助约束；不能对任意 32 维 e 做“与真实人格向量的 MSE”。

### 6.3 记忆检索、写入、压缩

检索使用多正例对比损失，同一事件不同来源/视角可以都是正例；近义但主体或时间不同的事件是难负例。来源与时间的硬过滤先执行，避免让神经相似度覆盖人物知识边界。

`L_mem = L_retrieval + λ_write CE(write_action) + λ_future L_delayed_tasks + β C_storage + λ_faith L_source_support`。

`C_storage` 用选定压缩级别的 token/字节预算，按每段轨迹归一；`L_delayed_tasks` 在后续问题或讨论出现时计算。硬 top-k、写数据库、选压缩级别没有自动梯度：首期用候选集中的可微重排/软注意力与有监督操作头；后期才比较轨迹策略优化。硬操作之后的下游 CE 不能直接训练其前面的离散选择。

### 6.4 意图、时机与证据

`L_intent = class_balanced_CE(intent)`；`L_timing` 对允许的起始/停止时间窗口计算区间损失，落在合理窗口内不强迫唯一毫秒值。对模糊样本提供允许动作集合或软标签，不将人类自然互动中的多种合理行为压成唯一真值。

证据不足概率用 Brier 或 CE 训练，用独立验证集校准。策略根据该概率、云端实际状态和交谈情境选择等待/澄清/表达有支持内容；云端离线不能把 pending 改为 supported。

轨迹奖励若后期引入，应分别计入有效信息、人物忠实、错误断言、无意义插话、重复故事、漏回应和延迟；保持分项报告，不只最大化对话时长或用户继续聊天的次数。

### 6.5 声音与人物的解耦

固定 codec 路线主要用目标音频 code 的掩码 CE。音色、情绪和语义因子使用不同条件入口；“同内容不同情绪”“同情绪不同说话人”样本可减少混淆。说话人相似度、ASR 内容一致、韵律和人工自然度独立评测。

经过 argmax/离散采样的音频不能直接将波形损失反传到生成 logits。只有明确提供可微路径或专门估计器时才使用该声学目标训练主干。人物语音 tokenizer、词表及 sample rate 必须与各硬件档的 Talker 匹配，不能混用 MiniMind/Mimi、MiniCPM 和 Qwen/CosyVoice 的 code。

## 7. 分阶段实验计划

| 阶段 | 可训练部分 | 数据 | 通过后进入下一阶段的依据 |
|---|---|---|---|
| P0 来源与对照 | 暂不更新网络；做人物包和检索基线 | 初始事件、留出的情境问题 | 来源和知识边界可检查，基线可复现 |
| P1 条件连接 | 人物/记忆投影；共享主干与语音权重冻结 | 有来源的人物对话、记忆读取样本 | 条件确实影响相关行为，通用理解不过度退化 |
| P2 稳定人格 | 人物 Adapter、人格编码与意图头 | SFT、难偏好对、多人物对照 | 盲测区分人物且无明显生平编造增加 |
| P3 动态情绪 | A-APP、E-DYN、R-REL、情绪连接 | 连续轨迹＋ESD/M3ED辅助 | 情绪随事件变化，语气改变不破坏声音身份 |
| P4 后天记忆 | M-CTRL、重排/压缩连接 | 多会话轨迹、纠正、分享记录 | 延迟任务、修正与去重优于静态 RAG |
| P5 联合适配 | 按硬件档逐步开放相关主干层；人物参数仍独立 | 混合通用/多人物/流式任务 | 新增效果在相同算力数据预算下仍成立 |

冻结接收模块参数不等于用 `no_grad` 包住其整个前向：训练人物/记忆连接时，需要从冻结主干输出向连接输入求梯度。纯冻结编码器在可训练连接之前的输出可以缓存；缓存需记录 persona 无关性、模型版本和预处理版本。

起始优化器 AdamW：新模块 lr=1e-4，人物 LoRA lr=5e-5，后期解冻主干 lr=5e-6；warmup 3%，gradient clip=1.0。按验证集调整，不把这些设置视为不同主干通用最优。各损失先按有效标签数归一，阶段内先令主任务权重为 1，辅助项从 0.1 起做单因素扫描，监测梯度与行为退化。

单个人物的资料不足以证明人格/音色/知识的可分离性。首批至少构造 3 个明显不同的人物条件（其中可含原创角色），执行同情境换人物、同人物换声音、同人物换临时情绪的交叉实验；不能将原创对照的虚构事件混入真实人物资料。

## 8. 验证、验收与消融

数据划分：原始录音、原作章节、采访和情境族为 split 单位，禁止把同一句台词的改写分到训练与测试两侧。公共角色辅助训练与角色泛化测试按人物/作品分割；后天记忆按时间推进，只允许使用当前时刻已经发生的经历。保留人名匿名化版本，排除主干仅凭知名角色名背诵的影响。

首期建议验收门槛如下，均为本项目拟定目标，非论文结果。测试集需至少 200 个支持/不支持生平问题、100 个知识边界问题、100 个成对人物情境、30 条跨会话链和 50 段时机轨迹；给出置信区间，规模不足时只报告探索性结果。

| 检验 | 建议门槛/报告方式 |
|---|---|
| 有依据的生平回答 | 支持率≥95%；另报拒答率，不能靠全拒答达标 |
| 素材未支持的亲历 | 肯定编造率≤2%，同时报告能正确回答的覆盖率 |
| 跨人物泄漏 | 预设探针中 0 次；报告样本数，这不证明总体风险为零 |
| 剧情/生平截止边界 | 越界率≤2%，另测未知情况下是否自然回应 |
| 人格取舍 | 人工盲评对“只给人物描述”的基线偏好率>60%，并检查置信区间 |
| 情绪与音色解耦 | 情绪条件变化后声音身份无显著退化；用配对测试与人工复核 |
| 交流分寸 | 无必要主动插话减少≥30%，合理回应召回下降≤5个百分点 |
| 后天纠正 | 新版本正确使用率≥90%；旧事件仍可追溯 |
| 重复分享 | 同一对象短期完整重复故事的次数低于无分享历史基线 |
| 未见情境 | 同事实、同人物，仅换措辞仍保持取舍；与训练台词复述分开计分 |

必做消融：①只画像；②画像+RAG；③再加人物 Adapter；④再加动态情绪；⑤再加学习的记忆控制；⑥完整模型去掉“是否值得开口”训练。尽量固定主干、数据量、检索预算和算力，逐项添加，而不是一次添加全部组件后归因。

人物经历不足时，以“源材料支持的行为子集”作为阶段成果，不把模型自行补完的内容纳入目标人物真值。技术上能更换人物包与复刻得像，是两项独立验收。

## 9. 来源与研究定位

- S1：[BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)；[配置](https://huggingface.co/BAAI/bge-m3/blob/main/config.json)。用于冻结文本检索基线。
- S2：[Titans](https://arxiv.org/abs/2501.00663)。仅作为运行时神经记忆的后续对照，不视为可直接复用的人格模块。
- S3：[DPO](https://arxiv.org/abs/2305.18290)。提供成对偏好优化方法，不提供目标人物偏好真值。
- [RoleLLM](https://arxiv.org/abs/2310.00746)、[CoSER](https://arxiv.org/abs/2502.09082)。角色条件化训练和数据组织参照，不能证明本项目的全双工、持续人格或记忆机制已经成立。
- [InCharacter](https://arxiv.org/abs/2310.17976)、[CharacterEval](https://arxiv.org/abs/2401.01275)。用于建立多维评测，问卷或自动裁判只作为辅助。
- [匿名角色评测研究](https://arxiv.org/abs/2603.03915)。支持在有名角色之外加入匿名、未见人物测试，避免角色名称带来的混杂。
- [Generative Agents](https://arxiv.org/abs/2304.03442)、[LongMemEval](https://arxiv.org/abs/2410.10813)。分别作为经历反思机制与长期记忆评测参照。

本文的新结构、维度、数据 schema 与验收标准是本项目的设计提案。模型可用性核验与数据来源核验不等于性能验证；后续应通过版本记录保留每次实测导致的调整。
