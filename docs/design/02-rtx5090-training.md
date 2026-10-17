# 单卡 RTX 5090：待训练部分、数据格式与优化目标

版本：v0.1；公开来源核验日期：2026-10-13；文档整理与一致性审查：2026-10-14。配套 [01-rtx5090-architecture.md](01-rtx5090-architecture.md)；人物专项见 [07-character-memory-emotion.md](07-character-memory-emotion.md)。本文给出可开始实施的训练契约与实验预算，所有规模、超参数和验收阈值是建议值，并非训练结果。

## 1. 固定实验边界

- B0：MiniMind-O 原配 Thinker–Talker，已有权重初期全部冻结。
- B1：Qwen3-1.7B＋冻结专用感知＋MiniMind-O Talker 初始化；新的语义/音频时间连接需要训练。不能把原配权重当作已适配 Qwen 的 checkpoint。
- core 为 64×512；人物 p256、情绪 e32、关系 r64、事件评价 a16。音频目标为 Mimi 同版本的 8 路码，12.5 Hz；人物包、记忆与状态按 persona_id/version 隔离。
- 云端只在推理阶段调用 API，不对其训练，也不调用来产生训练样本。云端等待训练使用人工审核的本地 fixtures 和可控网络状态模拟器。
- 首期不训练传感器执行器，不重训完整音视频基础模型，不使用评测集答案作训练标签。

“冻结”只表示不更新该参数：位于可训练连接层之后的冻结 Thinker/Talker 仍须对输入求梯度。只有连接层之前的纯冻结 encoder 特征可以用 no_grad 并缓存。不能把整个冻结主干包在 no_grad 中再期待训练其输入投影。量化训练须先验证输入梯度和算子支持；推理专用 backend 不能直接代替训练 backend。

## 2. 模块与可训练参数清单

| 模块 | S1 接口阶段 | S2 机制阶段 | S3 人物/语音阶段 | S4 联合阶段 |
|---|---|---|---|---|
| SenseVoice、SigLIP2、BEATs、CAM++、Mimi | 全冻结，encoder 可缓存 | 全冻结 | 全冻结 | B0 可逐路解冻；B1 默认仍冻结，另做小批量分组实验 |
| 感知 MLP 与 resampler | 训练新增参数；B0 原连接保持冻结作为参照 | 继续训练 | 小学习率或冻结 | 训练 |
| H_t core、快速声学 Conv/GRU | 尚不启用复杂闭环；只验证形状 | 全训练，8 秒 TBPTT | 继续训练 | 训练 |
| 状态→语义 cross-attention | 训练新增层 | 训练 | 训练 | 训练 |
| Qwen/MiniMind 语义主干 | 原权重冻结 | 原权重冻结 | persona LoRA rank=8；原权重冻结 | 先 LoRA/最后若干层，B0 才优先尝试全解冻 |
| 事件评价、情绪、关系与价值头 | 不在本阶段依赖其输出 | 训练通用状态更新 | 加人物条件训练 | 训练并保持多人物混合 |
| 记忆 query/写入/压缩选择/摘要 adapter | query 对齐可先训练 | 训练 | 训练人物经历使用 | 训练 |
| claim、证据、动作与等待策略头 | 独立小标签集试训 | 全训练 | 结合人格和关系 | 训练 |
| commitment bridge | B0 的新条件旁路训练 | 与策略连接 | B1 必须重新对齐；训练语义到音频 resampler | 训练 |
| 原 Talker | 冻结 | 冻结 | 先冻结试验；必要时解冻 embed_proj/前两层及 style LoRA | 按语音回归结果解冻更多层 |

新参数合计先按约 40–80M 设计，人物 LoRA 按实际 target_modules 统计。参数账本需记录 name、shape、numel、dtype、requires_grad、optimizer group；不以“只训练 adapter”代替完整统计。

## 3. 公开数据：下载入口、用途与限制

只下载本阶段需要的分片，原始数据保持只读。每个分片的 manifest 记录来源 URL、版本、SHA256、许可证、派生步骤与 split。以下下载入口已核验，未在本任务中执行下载；公开网页可访问不保证所有媒体链接永久有效。

| 数据 | 获得方式与公开状态 | 本档用途 | 划分与使用边界 |
|---|---|---|---|
| [MiniMind-O dataset](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset) | HF/ModelScope 的 Parquet 分片，官方 [说明](https://github.com/jingyaogong/minimind-o) 列出 T2A/A2A/I2T；先选对应任务小分片 | B0 复现、冻结连接层对齐、B1 的语音桥接试验 | mini 仅英文且无视觉；中文须 full 中的中文子集。卡片同时列 Apache-2.0/GPL-3.0，含不同上游与已有合成音频，不可把混合数据统一声明为单一宽松许可；记录分片来源。下载既有数据不等于本项目调用云端生成数据 |
| [AISHELL-1，OpenSLR 33](https://www.openslr.org/33) | 页面提供 data_aishell.tgz、resource_aishell.tgz 及校验文件 | 中文语音到语义连接、转写回归、讲话人条件基础任务 | 页面标 Apache-2.0；保留官方 train/dev/test，说话人无交叉；朗读语音不能替代全双工谈话 |
| [AliMeeting，OpenSLR 119](https://www.openslr.org/119) | Train_Ali_far/near 与 Eval/Test 的直接压缩包链接 | 8 通道阵列、重叠讲话、说话人和活动时序、近远场对照 | CC BY-SA 4.0；仅 Train 用于训练。它没有“AI 此时应否插话”的现成标签，须二次标注；Eval/Test 不混入训练 |
| [AMI](https://groups.inf.ed.ac.uk/ami/download/) | 官方 chooser 按 meeting 选择麦克风阵列、headset、低分辨率视频；标注 ZIP 单独下载 | 同步音视频、多说话人、视角—说话人关联、自然附和与沉默 | 公开 signals/transcription 标 CC BY 4.0，部分标注单列；全尺寸视频不在普通下载页。按整场及人员划分，不能随机切相邻帧跨 split |
| [COCO](https://cocodataset.org/#download) | 官方 train2017/val2017 图像与 captions/instances | 视觉 connector 静态对齐与基础回归 | captions 及图片各自条款按官方说明和图片来源保存；仅基础视觉使用，不把 caption 数据训练成“看到什么都要说”策略 |
| [AudioSet](https://research.google.com/audioset/download.html) | 官方 CSV 标注、ontology、特征；原始音频依赖对应公开视频获得 | BEATs 后的事件分类头、环境声与讲话共存 | 标签和下载页所列内容按官方许可；CSV 不是完整音频包，链接可失效；已有 YouTube 特征不能代替 BEATs 原始波形训练输入。记录缺失率；不把未标注类别一概作负样本 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 官方仓库音频、metadata 与 ZIP | 低成本环境声特征/事件头试验和五折评测 | 数据按仓库许可（整体含非商业限制，逐条来源另存）；同源片段不能跨官方 fold；使用此集合需区分研究试验与后续发布训练来源 |
| [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) | 官方 Google Drive；要求填写许可并按作者流程提交 | 中文/英文相同内容不同情绪的声学条件训练 | 仅研究用途，需要办理许可；本任务未替用户提交或发邮件。读情绪语音只监督表达，不把表演标签当真实内在情绪 |
| [LoCoMo](https://github.com/snap-research/locomo) | 作者仓库的数据文件与评测说明 | 多会话检索、时间与人物记忆回归 | 本档默认只作评测；数据许可以数据文件及作者说明为准，不能仅用仓库代码许可覆盖素材 |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 官方 README 给出数据下载与样本格式 | 长时事实更新、信息缺失和跨会话推理评测 | 本档仅作保留评测；不使用答案来训练记忆控制器，不把公开英文评测直接代表中文角色交流 |

如选择 ESD/ESC-50 等具有特定用途限制的来源，必须把来源标签传递到派生特征与 checkpoint 记录；不能因音频变成 Mimi codes 就丢弃其来源约束。人物素材的授权、剧情版本与证据级别见人物专项。

## 4. 必须自建的数据

| 自建集合 | 首轮建议规模 | 必备标注 |
|---|---:|---|
| 人物初始经历 M0 | 每人物 100–500 条有来源事件起；覆盖不同关系、冲突、变化 | 原文/音视频出处，剧情或人生时间，个人解释，支持/推测/未知，不能编造补齐 |
| 价值取舍与分享偏好 | 每人物 500–2,000 组情境及候选反应 | 哪个更符合素材、理由和证据；允许无法判定；包含“有经历但不应此刻分享” |
| 完整互动轨迹 | 起步 20–50 小时；2 人物先做接口 smoke test，身份解耦评测至少 3 条件（A/B/中性）；先标 2,000–5,000 个关键时刻 | 说/停/继续/听/澄清/等待、合适动作集合、可判别时间、已播放范围 |
| 云端状态 fixtures | 1,000 个基础场景，每个本地随机化 5–10 种延迟/错误/断线 | 哪些结论需确认、正确证据、错误返回、返回适用范围；不得把状态随机化副本当独立测试样本 |
| 长期经历序列 | 200–500 个跨 5–20 次会话的轨迹 | 承诺、偏好变化、纠正、分享记录、未来查询/主动使用机会、可丢弃细节 |
| 目标声音和语气 | 在权利允许的素材中先整理 30–120 分钟干净多情境音频；若素材不足如实标记 | 文本、时长、讲话人、情绪/韵律、噪声、参考音频；此规模只是适配试验，不保证复刻质量 |
| 缺失与恢复场景 | 在独立干净轨迹上有控制地派生 | 断流开始/结束、质量、available_at、应保留的不确定性 |

以上规模用于估计首轮标注工作，不代表样本复杂度已被证明。先取各集合 5% 做过拟合/数据正确性实验，再扩大。自建人物可包括授权真人、明确原创角色或有权使用的角色素材；动漫须注明作品、剧情时间点和人物已知范围。

标注流程：两名标注者独立标关键决策，分歧保留为动作概率或可接受集合；事实类由原始来源复核。偏好分歧不强行制造唯一人格真值。自动对齐器、冻结本地模型只能提供候选，抽样人工审核；禁止用当前学生输出给自己制造高置信真值。

## 5. 数据目录与统一字段

建议独立数据目录（本任务仅定义，不生成实现）：

~~~text
data/
  manifests/{dataset_id}/{revision}.jsonl
  media/{source_id}/
  features/{encoder_revision}/{source_id}.safetensors
  episodes/{split}/{episode_id}.jsonl
  people/{persona_id}/{version}/
  fixtures/cloud/{scenario_id}.json
  labels/{annotation_version}/
  splits/{experiment_id}.json
~~~

所有记录都必须有 schema_version、sample_id、source、split_group、persona_id/version（无人物任务可为 null）、session_id、event_start/end_ms、available_at_ms、validity。媒体用 URI/相对路径引用，不在 JSON 塞入巨量浮点数组。tensor 文件包含 encoder_revision、preprocess_revision、dtype、shape、lengths、frame_times、checksum。

输入长度、目标长度、loss_mask、可见性 mask 分开保存。available_at 指该特征真正可用于决策的时间；event_end 早于 available_at 时，不能训练时提前提供特征。云端只记录事实状态和已发生耗时，不向模型泄露模拟器预知的未来返回时间。

### 5.1 连接层与语音训练样本

下面是缩略但合法的 JSON；码序列放在外部文件，示例路径仅说明格式。

~~~json
{
  "schema_version": "encephalon.sample.v1",
  "sample_id": "align-0001",
  "task": "audio_to_text_audio",
  "source": {"dataset": "local_authorized", "record_id": "rec-31", "license_record": "licenses/rec-31.json"},
  "split_group": "recording-31",
  "persona_id": "character-a",
  "persona_version": "1.0",
  "session_id": "session-31",
  "state_version": 0,
  "request_id": null,
  "event_start_ms": 0,
  "event_end_ms": 2400,
  "available_at_ms": 2600,
  "validity": "present",
  "input": {
    "audio_uri": "media/rec-31/input.wav",
    "sample_rate": 16000,
    "channels": 4,
    "speech_features_uri": "features/sensevoice-r1/rec-31.safetensors",
    "transcript": "你为什么后来改变了想法？",
    "views": [],
    "history": []
  },
  "target": {
    "text": "因为后来的经历让我发现，当时忽略了别人的处境。",
    "audio_uri": "media/rec-31/answer.wav",
    "codec_uri": "features/mimi-r1/rec-31-answer.safetensors",
    "codec_shape": [8, 75],
    "codec_frame_rate": 12.5,
    "speaker_embedding_uri": "features/campplus-r1/character-a.safetensors",
    "reference_audio_uri": "media/character-a/reference.wav",
    "support_event_ids": ["event-a-17"]
  },
  "supervision": {"text_loss": true, "codec_loss": true, "emotion_loss": false}
}
~~~

原 MiniMind-O Parquet 使用 conversations、question_audios、answer_audios、image_bytes、ref_audios、spk_emb 等字段，[官方 dataset 实现](https://github.com/jingyaogong/minimind-o/blob/master/dataset/omni_dataset.py)说明了读取和延迟排布。本项目导入时映射到上面 schema，保留原行 ID。不能把该实现对多通道平均的行为原封不动当作阵列融合。

参考音频必须和目标音频分离；若用同一句目标作 reference，容易让模型复制目标，夸大零样本音色效果。B1 重新用 Qwen tokenizer 编码文本；原 MiniMind token ID 不可直接复用。

### 5.2 全双工与云端决策样本

~~~json
{
  "schema_version": "encephalon.episode_tick.v1",
  "sample_id": "ep-18-t15200",
  "source": {"dataset": "local_authorized", "record_id": "ep-18", "license_record": "licenses/ep-18.json"},
  "split_group": "scenario-18",
  "persona_id": "character-a",
  "persona_version": "1.0",
  "session_id": "ep-18",
  "state_version": 41,
  "request_id": "req-18",
  "tick_ms": 15200,
  "event_start_ms": 14500,
  "event_end_ms": 15200,
  "available_at_ms": 15200,
  "validity": "present",
  "observations": [
    {"source_id": "mic-array-0", "event_start_ms": 14500, "event_end_ms": 15100, "available_at_ms": 15200, "validity": "present", "quality": 0.84, "revision": 2, "text": "等一下，我刚才说的年份不对", "speaker_id": "user-1"}
  ],
  "playback": {"utterance_id": "utt-9", "generated_until_ms": 15800, "queued_until_ms": 15440, "played_until_ms": 15200},
  "cloud": {"status": "running", "elapsed_ms": 1800, "request_state_version": 39, "result": null},
  "labels": {
    "acceptable_actions": ["stop_and_listen"],
    "first_decidable_ms": 15200,
    "preferred_action_ms": 15240,
    "action_deadline_ms": 15400,
    "needs_cloud": true,
    "invalidated_claim_ids": ["claim-5"],
    "write_memory": false,
    "supervision_confidence": 0.95
  }
}
~~~

动作枚举至少包括 listen/continue_speaking/stop_and_listen/backchannel/ask_clarification/speak_committed/defer。request_cloud 是可与动作并存的独立二值头。必须标注同样一句“嗯”在附和、犹豫和打断语境中的不同处理；VAD 时间只作观察，不自动成为策略标签。上例的语义特征在 15200 ms 才可用，因此可判别/期望动作时间不能标到该时刻之前；若另有更早可用的快速分支，必须在 observations 中显式给出对应特征与 available_at。

### 5.3 记忆与结论样本

~~~json
{
  "schema_version": "encephalon.memory_trial.v1",
  "sample_id": "memory-trial-21",
  "source": {"dataset": "local_authorized", "record_id": "life-story-21", "license_record": "licenses/life-story-21.json"},
  "split_group": "life-story-21",
  "persona_id": "character-a",
  "persona_version": "1.0",
  "session_id": "memory-session-21",
  "state_version": 31,
  "request_id": null,
  "query": "朋友再次提到合作，我想起什么以及是否值得说？",
  "event_start_ms": 859000,
  "event_end_ms": 860000,
  "available_at_ms": 860000,
  "validity": "present",
  "candidates": [
    {"event_id": "event-a-17", "source_ref": "interview-2:180-220s", "event_time": "2019", "known_at_ms": 0, "evidence_status": "supported", "summary": "一次合作中未及时沟通，之后更重视提前说明风险。", "shared_with": ["user-1"], "token_cost": 40},
    {"event_id": "event-a-39", "source_ref": "session-10:turn-8", "event_time": "2026-10-12", "known_at_ms": 500000, "evidence_status": "reported_by_user", "summary": "对方当前更担心失去自主性，而不是收入。", "shared_with": [], "token_cost": 30}
  ],
  "labels": {
    "relevant_event_ids": ["event-a-17", "event-a-39"],
    "prefer_brief_reference": true,
    "full_story_should_be_repeated": false,
    "budget_tokens": 80,
    "future_probe": "对方目前最在意什么？",
    "future_answer": "自主性",
    "answer_support_ids": ["event-a-39"]
  }
}
~~~

claim 训练记录另外保存 claim_id、candidate_text、claim_type（事实/经历/偏好/推测）、support_ids、contradiction_ids、needs_cloud、expressibility（assert/qualify/hold）、already_spoken、corrected_by、标签来源。价值偏好不强行标成客观真/假；证据不足只用于需要依据的事实性断言。

### 5.4 情绪与人物样本

字段：persona_id/version、当前话题、角色已知事件集合、relationship_id、相关 M0/M_after 事件、评价 a 的已知标签及 16 维 mask、情绪读出标签及 mask、前后状态时间、候选回复、偏好分布、证据出处、音频韵律标签、allow_silence。e32 是潜在状态，没有全量人工“真值”；只能用部分可解释读出和行为后果监督，不能虚构 32 维标签。

人物专项中的 PersonPack JSON 为权威格式。此处的 a16/r64/p256 是训练起点；更换维度必须同步 architecture manifest，而不是静默修改数据。

## 6. 损失定义与梯度路径

### 6.1 公共归一化

每个任务先对单样本有效位置归一化，再对该任务有效样本平均，避免长语音/长历史获得不受控权重：

    Mean_m(ℓ) = Σ_i m_i w_i ℓ_i / max(Σ_i m_i w_i, ε)

m 表示“该目标有有效监督标签”，w 是标注可信度或已声明的类别权重。没有该任务标签的样本不计该项，不把 null 当 0。输入可见性另用 available_at mask；未来预测标签可以位于未来，但只能用于 loss，不能提前注入模型输入。训练与测试均屏蔽 padding、参考音频、输入提示等不属于目标的区域。

### 6.2 连接与语义目标

有监督文本目标：

    L_text = Mean_m[-log pθ(y_i | observed_≤t, y_<i, PersonPack)]

仅对目标 assistant/指定监督转写区域计算；不能把输入用户文本也当人物输出训练。语音输入对齐阶段可单独训练转写或语义问答，后续才混合角色表达。

冻结特征与共享表征对齐采用 InfoNCE：

    L_align = -(1/N) Σ_i log[exp(sim(z_i,t_i)/τ) / Σ_j exp(sim(z_i,t_j)/τ)]

z 为可训练 connector 输出池化，t 为冻结目标编码或经过审核的配对文本特征；τ=0.07 起。负样本排除同一事件的邻近片段和重复字幕，避免把本来等价的样本当负例。L_text 梯度穿过冻结主干流到 connector；冻结主干权重不更新。

多视角对齐使用同一对象/事件的配对标签，仅在可靠同步/匹配时加入；禁止对所有不同视角强迫完全相同表征，以免丢失互补信息。

环境事件头使用多标签 BCE：

    L_event = Mean_known[-y log σ(l) -(1-y)log(1-σ(l))]

AudioSet 的弱标签缺失类别不能直接当确定负例；首期用可信正负类/人工复核子集，或明确标记 PU 学习实验。不同分类集合的 ontology 先映射，未映射项 mask。

### 6.3 持续状态与短时预测

H_t 的主要监督来自文本、交互、记忆和证据任务，另加有限辅助目标：

    L_state = L_event + λ_id L_speaker + λ_future Mean[Huber(Pred(H_t), stopgrad(z_(t+δ)))]

δ 先取 0.32/1.0 秒；预测目标可来自后续已记录特征，但它只能放在 loss 端，不能作为 H_t 的输入。未来不确定事件标记缺失/低权重，不把全场视频编码成学生当前状态。speaker 头训练候选匹配/unknown/overlap，使用带说话人标签的音频；方向回归采用有真实标定或已知混音方向的数据。

8 秒 TBPTT 窗口间 detach H；同一 episode 顺序传递状态。训练初期同时混合从零状态开始的短片段，避免只会依赖理想 warm-start。截断反传不代表旧事件在训练中永远无法影响后续，长期作用主要通过显式记忆和跨会话任务监督。

### 6.4 交互、等待与时机

存在多种合理动作时，用标注者分布 q 而非唯一标签：

    L_action = Mean_t[-Σ_a q_t(a) log πθ(a|H_t,p,e,cloud_status)]
    L_call = Mean_t[BCE(p_call,t, y_call,t)]

时机损失采用离散 hazard。若事件应在第 k 个 tick 执行动作、此前均不执行：

    L_time = -log h_k - Σ_(j<k) log(1-h_j)

窗口中没有执行事件则作为右删失，只保留已观察区间的 Σ-log(1-h)。对允许时间区间可把概率质量对区间求和再取 -log。不要同时把所有未说话 tick 当无限重复的负样本；按事件和背景时段平衡采样。

等待行为依据真实状态与已等待时长；模拟器最终返回时间不作为输入。如果供应商没有公开远端进度，queued/running 只监督“本地待派发/已发出在途”，不得生成虚假的远端排队、完成比例标签。行为标签由有依据的场景与人类标注产生，不由“延迟超过几秒就一定说话”的固定规则替代。初期优化 L_action+L_time+L_call，暂不直接训练在线 RL。

### 6.5 证据、结论与表达控制

对于有可核验标签的 claim，y_unsup=1 表示当前依据不足以肯定断言：

    L_claim = Mean[(p_unsup-y_unsup)^2]
              + CE(expressibility, assert/qualify/hold)
              + BCE(p_requires_cloud, y_requires_cloud)

含来源检索的证据支持分类可加 CE(supported/contradicted/insufficient)。标签必须基于当时可见证据，不以云端成功状态替代 supported。校准集单独用于温度/阈值选择；Brier/ECE/风险—覆盖率在测试集报告。

训练 commitment bridge 时，初期输入标注的可表达 claim；之后混合模型自身的候选和标注纠错。若用软门 g_c∈[0,1]，增加：

    L_gate = Mean_c[BCE(g_c, y_allowed,c)]

输出文本的事实支持另做人类/固定标注集评价。不能认为 g=0 就严格阻止语义主干凭内部知识说出某件事；需要进行未获支持内容诱发测试。推理 hard selection 不可微，其学习由上述分类与模仿目标完成。

### 6.6 记忆读写、压缩与预算

读取：固定 BGE 文档向量，训练由 H_t 生成的 query 与候选重排，使用多正例 InfoNCE 或 listwise CE。训练时提供固定候选集合并用 soft attention；部署 FAISS top-k 是离散检索，生成 loss 不会自动通过索引更新写入策略。

写入和压缩级别：

    L_write = BCE(p_write,y_write) + CE(level, raw/event/semantic/drop)
    L_budget = max(0, Σ_i p_write,i × cost_i - budget)^2 / max(budget²,1)
    L_summary = L_text(summary) + λ_fact L_retained_facts
    L_memory = L_retrieval + L_write + λ_b L_budget + λ_s L_summary + λ_f L_future_task

cost 使用 token 或实际字节预算，不混用未说明的单位。L_retained_facts 对人物、时间、因果、立场和支持来源做字段级 CE/BCE；自由摘要的真实性另行抽验。future_task 使用独立后续回忆、承诺和主动使用机会；不能把未来答案泄露给压缩器输入。

L_future_task 仅在 soft candidate/神经摘要等可微支路回传。真正的磁盘写入、hard top-k 或已采样摘要不具备该路径，需用读写标注单独训练；后续若做策略梯度，应明确另立实验，不写成普通端到端反传。

### 6.7 人格、情绪与关系

    L_person = L_text(person demonstrations) + λ_pref L_DPO + λ_source L_source
    L_affect = Mean_label[Huber(readout(e_t), affect_label)]
               + λ_a Mean_known[Huber(readout(a_t), appraisal_label)]
               + λ_rel L_relation + λ_stable L_stable

readout 可先有 valence/arousal/dominance 三维辅助输出，但潜状态 e32 不必被限制为三维情感。L_stable 只在无重要新事件、无关系变化的区间抑制无故跳变；刺激出现时 mask=0，避免训练出迟钝情绪。关系标签以事件中的行为与明确陈述为据，不把亲密程度模型输出当真实关系标签。

DPO 对相同人物、相同情境的两段候选 y+ / y-：

    L_DPO = -log σ(β[(log πθ(y+|x)-log πref(y+|x))
                      -(log πθ(y-|x)-log πref(y-|x))])

πref 是本地冻结的 SFT 参考，不是云端。只对回复 token 求 log-prob，参考与待训模型用相同 tokenizer；候选长度尽量匹配，监测长度偏置。β=0.1 为试验起点，需校准；不把“更讨好用户”当人物忠实偏好。L_source 用 source-ID 检索/支持监督约束经历依据，不要求暴露内部推理过程。

### 6.8 Talker 与桥接

codec 目标来自匹配版本的 Mimi 编码。设 c_(k,t) 为第 k 个码本的真实目标，m 为有效目标帧且已去除参考与 padding：

    L_audio = (1/K) Σ_k [Σ_t m_(k,t) w_(k,t) CE(logits_(k,t),c_(k,t))
                         / max(Σ_t m_(k,t) w_(k,t),ε)]

K=8，延迟排布 d_k 与模型版本绑定；EOS 可设较大权重但先从 1 起，比较 2/5，不能直接照搬另一任务权重。必须验证停止符与普通码本值空间不混淆。参考音频、说话人占位、输入语音等只作条件。

B1 可增加同一句目标在 B0 原配网络中的条件表示作为**本地冻结教师**：

    L_bridge = Mean_valid[1-cos(C_Qwen(h), stopgrad(C_B0(h0)))]

不同 tokenizer 的隐藏位置不能逐 token 硬配；对相同文本片段按字符跨度/已知语音时间映射并池化，再比较。不存在可靠对齐时关闭该辅助损失，仅使用经过时序 resampler 的 L_audio。教师隐藏可缓存，不能覆盖目标人物语义监督。

L_audio 可穿过冻结 Talker 回到可训练 bridge，但不更新冻结 Talker 参数；如果解冻对应层，则同步更新这些层。Mimi 的离散码目标是常量，不需反传。音色相似度、CER、韵律自然度首期用作评测：采样码/argmax→codec→波形之后不能默认普通梯度可传回 logits。若以后加入可微声学损失，另行明确连续替代或采样估计方案。

## 7. 训练阶段与单卡起始超参数

统一使用 AdamW、BF16（数值敏感 encoder 保持 FP32）、梯度裁剪 1.0、warmup 3%、固定 seed 多次复验；参数只是启动配置。

| 阶段 | 训练内容与数据 | 起始超参数 | 进入下一阶段条件 |
|---|---|---|---|
| S0 数据与原配复现 | 不新增复杂模块；B0 官方格式小子集与自有 200 条人工检查集 | 先纯推理再小集过拟合；无须跑完整公开集合 | 时钟/目标/码本与源文件一致；原配基线可复现 |
| S1 冻结接口 | MLP/resampler/cross-attention；AISHELL、COCO、MiniMind 对应子集 | microbatch=1–2，序列 1024，累积 16；新参数 lr=1e-4 | 验证 loss 降低、输出实际依赖输入；原任务回归可接受 |
| S2 持续机制 | core、快速状态、policy/claim/读写头；AliMeeting/AMI 的二次标注＋本地 fixtures | 8 秒 TBPTT，轨迹 batch=1–2，累积 8–16；lr=1e-4 | 语义打断优于 VAD 对照，同时报告漏打断和沉默问题 |
| S3a 人物 | p/人物 LoRA/评价情绪/记忆；M0＋人物情境偏好＋多会话 | 主干冻结；新头 lr=5e-5，LoRA lr=1e-4；2–4 人物混合 | 未见情境忠实度改善，经历幻造与人物串扰不升高 |
| S3b B1 声音 | 新 bridge/style；MiniMind 音频子集＋授权目标声音 | microbatch=1，音频先 3–8 秒；lr=1e-4，新解冻 Talker 层 lr=1e-5 | CER、参考独立音色相似度、重复漏词和 RTF 达标 |
| S4 联合 | 连接＋状态＋策略＋少量主干/Talker 解冻；混合通用回归与完整人物轨迹 | 长度 1024–2048，checkpointing，累积 16；新参数 3e-5，旧层 1e-6–1e-5 | 主任务增益超过回归损失，显存与实时预算未失控 |
| S5 运行积累 | 更新外部事件和状态；离线审核后低频人物 adapter 巩固 | 不在线逐条训练共享主干；保留旧 checkpoint 与回放集 | 新经历纳入且旧人物核心判断没有无依据漂移 |

总损失按任务有效样本先归一化后再加权。首轮可采用：L_text=1，L_audio=1，L_action=1，L_claim=1，L_memory=0.5，L_affect=0.2，L_align=0.1，L_bridge=0.1，L_budget=0.01；只启用该阶段有标签的项。先观察各项梯度范数和验证曲线，再调权重，不把初始数值当理论结论。S4 加通用回归数据的 L_text/L_audio 保持能力；参考 logits 的 KL 只针对同一输入、相同 tokenizer 的本地冻结参考。

常规 B1 全参数 Adam 的权重、梯度、主副本和动量合计就可能占满 32 GB，故 S4 不承诺所有本地模块同时全解冻。逐组解冻和 CPU offload 是替代试验，需单独报告吞吐与梯度覆盖；B0 加入本方案新模块后约 0.65–0.75B，可作为完整联合解冻的小规模研究平台，仍需实测感知激活。冻结 encoder 的 no_grad/注册方式、codec 的量化梯度估计及 optimizer 参数覆盖都须在该阶段重审，不是切换原仓库 all 选项即可。缓存被冻结特征的阶段，若改变 encoder 权重或预处理，必须失效并重算缓存。

## 8. 数据划分、反作弊与评测

1. 公开集合沿用官方 split。自建数据按完整人物素材源、人物、场景、拍摄会话和时间分组，基础场景的不同延迟副本必须在同一 split；常规比例可从 80/10/10 开始。
2. 做两类人物评测：已训练人物的新情境；完全保留人物的迁移。素材少时用事件留一，不能随机打散同一段访谈的句子制造高分。
3. 人物长期评测设“知识截止点”：角色只能访问该剧情/生涯时刻已知的事，后续剧情只能出现在评测目标或新观察中。
4. 记忆分别测读回、修正、主动恰当使用、避免重复分享和适当沉默。LoCoMo/LongMemEval 只覆盖其中部分，不代表自然交流已达标。
5. 全双工至少报告错误打断、漏打断、附和识别、旁人误响应、语义可判别后停止延迟及首音后停止延迟；不能把两个时钟混为一个低延迟数字。
6. 云端结论评估报告 Brier、ECE、风险—覆盖率、过度调用、无效等待、过期内容采纳与无依据自信断言。并列“从不回答”“总等云端”基线，防止策略投机。
7. 音频分别报告 CER/WER、说话人相似度、人工自然度/人物一致性、长句漏词与循环、流式 RTF 和停止缓存延迟；相似度 encoder 不应与训练监督来源完全相同而没有独立验证。
8. 视觉只作必要上下文的使用率与任务辅助效果评测，同时测“没有必要时不播报”。不把 caption 得分当主动交流质量。
9. 故障注入：遮挡/麦克风丢包/云端断线/记忆超时/旧结果乱序/人物切换。报告哪些剩余能力保留，核心进程故障另测恢复机制。
10. 延迟/显存统计包括预处理、GPU排队、网络、音频设备；至少跑 30 分钟，记录 p50/p95/p99 与峰值，不只测单模型前向。

对照矩阵最少包含 B0 原配、B0＋状态、B1＋MLP、B1＋状态/claim/记忆、去人格记忆、只 RAG、去云端状态和随机人物条件。每次只改变待研究机制，保持主干、训练数据规模和算力预算可比。

## 9. 每次训练必须产出的研究记录

保存模型与数据 revision、PersonPack 版本、全部 loss 权重、可训练参数列表、梯度路径说明、峰值显存、样本可见性检查、训练/验证曲线、失败样例以及所有对照配置。评测报告逐项标注“实测”“由公式估算”“未验证”。没有足够证据的结果记录为待验证，不能仅凭一次流畅演示宣称已实现持续人格、全双工或消除幻觉。
