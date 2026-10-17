# 大型服务器集群：待训练模块、数据格式与优化目标

> 版本：0.1；整理日期：2026-10-14；资料核验日期：2026-10-13至2026-10-14。对应 [05-cluster-architecture.md](05-cluster-architecture.md) 的 C0/C1/C2/C3 路线。本文定义初期训练契约，不包含训练代码，也不启动模型或数据下载；核验仅限官方页面、源码/配置和论文，未实际测试下载链路或训练运行。规模、学习率和验收阈值均为可调整的起始实验设置。人物部分同时遵循 [07-character-memory-emotion.md](07-character-memory-emotion.md)。

## 1. 训练范围与阶段顺序

主线先冻结 `Qwen/Qwen3-Omni-30B-A3B-Instruct` 的原配预训练模块，训练新增机制，再适配目标声音，最后联合解冻。远程第三方模型在所有阶段都仅作运行时推理 API；本方案不通过该 API 产生训练集，不训练 API 模型，也不把其返回无条件视为真值。

| 阶段 | 可训练参数 | 冻结部分 | 数据/规模起点 | 进入下一阶段的条件 |
|---|---|---|---|---|
| S0：原配复现 | 无 | 全部 | 500 条能力检查；200 段音视频；100 条人物声学对照 | 原配加载、时间轴、音频播放与记录正确；保留未修改的结果 |
| S1：接口对齐 | 旁路投影、resampler、空间音频连接器、状态读取 Adapter | 原配编码器/Thinker/Talker/codec | 10万条接口样本起步，扩到100万；固定状态槽先只充当输入容器 | 文本任务有效、连接非随机、移除新模态影响符合预期 |
| S2：持续状态与策略 | H_t 更新、事件边界、说话人关联、interaction/claim/cloud heads、记忆选择器；S1 连接层 | 原配主干和声学模块 | 自建100小时互动＋10万标注决策点；后续300–1000小时 | 同时通过正确打断、继续、沉默、云端延迟与缺失模态测试 |
| S3：人物经历与动态情绪 | 人物 embedding/LoRA、评价与情绪网络、关系更新、记忆融合 | 通用主干优先冻结 | 先3个人物包，每人物100–500条有据事件、1千–5千个情境/偏好样本 | 人物偏好与经历可区分，换包不串记忆，不靠口头禅刷相似度 |
| S4：神经声音桥 C2 | 2048→896 桥、单调对齐、声学 cross-attention；其后语音 LM LoRA/后部层 | 初期 flow/HiFT、speech tokenizer、speaker encoder | 公共/自有中文100–500小时＋英文100小时；每目标人物30–120分钟高质量起步，5–20小时为增强计划 | 与原生文本TTS对照相比，可懂度、音色、时序达到约定容差 |
| S5：联合适配 | 新模块＋Thinker LoRA，继而后部层/全Thinker；必要时语音LM/感知末层 | codec/tokenizer先保持版本固定 | 混合任务：20%接口保持、30%连续互动、20%人物、20%语音、10%长期记忆起始配比 | 新机制获益，通用能力与实时性未超出退化预算 |
| S6：全本地模型研究 | 有证据需要时开放感知、Thinker、声学生成；codec另设版本化实验 | 第三方API始终外部；量化/检索/媒体I/O为离散边界 | 依据S5瓶颈扩数据，不能仅扩大epoch | 旧新码本、记忆编码版本兼容性经过迁移验证 |

这里“全本地模型”不意味着直接把所有 `requires_grad` 改成 true 就完成训练。ONNX speech tokenizer 没有自动梯度；欲训练它需获得/实现对应可训练网络和独立声学目标，再同步重编码数据。C1 原配 Qwen Talker 的目标真实音频 codes 获取条件未核实完毕，不把该分支的任意音色全参训练排进必达计划；目标人物声音由 S4/C2 的公开 CosyVoice3 训练组件承担。

## 2. 数据资产组织和统一时间契约

建议存储单位是**会话/事件，而非独立问答行**。原始素材只保存一次，JSONL/Parquet 清单引用音视频和张量；一段素材可产生不同训练视图，但不能跨 train/val/test 泄露。

推荐逻辑目录（此处是拟定格式，不表示文件已存在）：

```text
dataset_release/
  manifest.json
  sources.jsonl
  sessions/*.json
  labels/*.jsonl
  media/{source_id}/
  features/{encoder_revision}/{sample_id}.safetensors
  speech_tokens/{tokenizer_revision}/{utterance_id}.npy
  splits/{train,val,test}.json
  annotations/{round_id}/
```

每一条训练记录必须携带：`schema_version/sample_id/source_id/persona_id/persona_version/session_id/event_id/request_id`、`time/available_at`、`validity`、`label_mask`、`split_group`、来源许可及标注来源。不得用云端是否在线作为 `claim_supported` 的标签。

### 2.1 统一样本信封

以下 JSON 均为合法格式的短例；数组长度很小时只是样例记录，不代表实际训练维度。大张量以引用和 shape 表示，避免伪造 256 维人格真值。2.2–2.6 节给出任务 payload，落盘时通过 `sample_id` 与本节统一信封关联，继承必填来源、人物、时间和划分字段；它们不是省略这些元数据的独立数据标准。

```json
{
  "schema_version": "0.1",
  "sample_id": "sample_00041",
  "task": "stream_policy",
  "source_id": "capture_0008",
  "persona_id": "person_a",
  "persona_version": "1.0",
  "session_id": "session_004",
  "event_id": "event_017",
  "request_id": "req_003",
  "split_group": "participant_group_02_recording_day_03",
  "time": {"start_ms": 10000, "end_ms": 18000, "decision_ms": 14200},
  "available_at_ms": 14200,
  "validity": {"audio": true, "camera_front": true, "camera_side": false},
  "source_terms": {"kind": "self_collected", "record_id": "release_2026_01"},
  "annotation": {"method": "human_dual_review", "version": "2"},
  "label_mask": {"action": true, "affect_readout": false, "speaker": true},
  "payload_ref": "labels/session_004_event_017.json"
}
```

`decision_ms` 时只允许读取 `available_at_ms <= decision_ms` 的输入。离线整句字幕、未来情绪、以后才知道的云端答案不得提前进入 state。目标回应/未来事件可以作为 loss 标签，但不能混进条件特征。

### 2.2 感知连接器与空间融合样本

```json
{
  "sample_id": "align_0021",
  "source_id": "meeting_006",
  "time_range_ms": [12000, 16000],
  "audio": {"uri": "media/meeting_006/array.wav", "sample_rate": 48000, "channels": 6, "channel_mask": [1, 1, 1, 1, 1, 1], "geometry_ref": "media/meeting_006/mics.json"},
  "views": [
    {"view_id": "front", "uri": "media/meeting_006/front.mp4", "frame_times_ms": [12000, 12500, 13000], "calibration_ref": "media/meeting_006/front.json"},
    {"view_id": "side", "uri": "media/meeting_006/side.mp4", "frame_times_ms": [12000, 12500, 13000], "calibration_ref": null}
  ],
  "transcript": {"text": "先别回答，我修改一下刚才的条件。", "speaker_id": "spk_02", "word_alignment_ref": "labels/align_0021_words.json"},
  "environment_events": [{"label": "door_close", "start_ms": 12300, "end_ms": 12600}],
  "speaker_activity": [{"speaker_id": "spk_02", "start_ms": 12200, "end_ms": 15300}],
  "supervision": {"task": "answer_from_observation", "target_text": "说话者正在修改先前条件，应继续听。", "target_evidence_refs": ["transcript"], "doa_degrees": null, "doa_valid": false}
}
```

可供本阶段使用的标签：文字转写、音频事件多标签、跨视角同事件/同对象配对、speaker 活跃段、可选方向与遮挡。语义描述任务用于接口验证，其答案不自动成为实时系统必须说出的句子。视听不同步、摄像头丢帧、混响、输入/播放回声都应进入增强；增广后的标签时间必须同步变换。

### 2.3 持续状态、全双工与云端等待轨迹

```json
{
  "sample_id": "duplex_0103",
  "persona_id": "person_a",
  "persona_version": "1.0",
  "session_id": "session_004",
  "state_init_ref": "states/session_004_10000.safetensors",
  "events": [
    {"event_id": "ev_101", "kind": "self_playback", "available_at_ms": 12000, "text": "如果按刚才的预算", "played_until_ms": 12700, "plan_id": "plan_09"},
    {"event_id": "ev_102", "kind": "user_audio", "available_at_ms": 12800, "audio_ref": "media/duplex_0103.wav", "speaker_id": "spk_02", "transcript_visible_until_ms": 12600},
    {"event_id": "ev_103", "kind": "api_status", "available_at_ms": 12820, "request_id": "req_03", "status": "RUNNING", "status_scope": "local_request_inflight", "context_version": 7}
  ],
  "decisions": [
    {"t_ms": 12900, "action": "PAUSE", "acceptable_actions": ["PAUSE", "STOP"], "intent": "listen_to_revision", "evidence_available_ms": 12800, "target_stop_by_ms": 13200, "loss_mask": true},
    {"t_ms": 14400, "action": "CLARIFY", "acceptable_actions": ["CLARIFY"], "intent": "resolve_missing_constraint", "evidence_available_ms": 14000, "target_text": "新的预算上限是多少？", "loss_mask": true}
  ],
  "claims": [{"claim_id": "claim_44", "text": "方案满足新的预算", "support": "INSUFFICIENT", "external_confirmation_needed": true, "evidence_refs": [], "valid_until_ms": null}],
  "cloud_fixture": {"request_id": "req_03", "return_at_ms": 18000, "context_version": 7, "result": "旧预算下满足条件", "evidence_refs": ["fixture_table_02"], "valid_for_current_context": false},
  "outcome": {"interruption_was_correct": true, "old_result_spoken": false, "filler_repetitions": 0}
}
```

必须补充配对轨迹：同样“嗯”在附和与请求插话中不同；同样“等等”由对话者或旁人发出不同；云端相同内容但延迟/失败/过期不同；人物相同经历但已经向本用户说过/没说过不同。训练中可模拟 API 状态机及人工编写的有证据返回，模拟返回不伪装成真实云端模型输出。模型在决策时不能读取 fixture 中未来的 `return_at_ms`；这个字段仅供环境模拟器调度。供应商没有回执时只标注本地队列/请求在途，不创建虚假的后端进度监督。

### 2.4 人物、价值判断和情绪轨迹

```json
{
  "sample_id": "persona_pref_0048",
  "persona_id": "person_a",
  "persona_version": "1.0",
  "canon_scope": "authorized_interview_2024",
  "memory_snapshot_ref": "personpacks/person_a/memory_v1.jsonl",
  "source_refs": [{"source_id": "interview_07", "start_ms": 120000, "end_ms": 143000}],
  "situation": "朋友在已有承诺和一个临时机会之间犹豫。",
  "relevant_memory_ids": ["ep_0031"],
  "candidate_a": "先确认原来的承诺能否妥善调整，再决定是否接受新机会。",
  "candidate_b": "新机会只要更有利，就不用考虑之前的约定。",
  "preferred": "a",
  "preference_basis": "原素材中明确谈到对承诺和协商的重视",
  "label_confidence": 0.9,
  "affect_readout_before": [0.1, 0.2, 0.0],
  "affect_readout_after": [0.0, 0.4, 0.1],
  "affect_valid": false,
  "share_memory_now": true,
  "previously_shared_with_listener": false,
  "chosen_intent": "discuss_using_experience"
}
```

这里示例情绪数值没有人工依据，因此 `affect_valid=false`，不计算情绪回归损失；真实训练应填有依据的标签或保留缺失。初始 256/32/64/16 维人格、情绪、关系、评价隐变量是潜表示，不是标注人员凭空填写的向量。

动漫角色要带作品、版本、剧情时间与角色当时知情边界；真人要带素材来源和时间。叙述者知道的剧情不能直接当角色亲历；演员访谈中的价值判断不能直接当所饰角色的价值观。数据不足的倾向标作推测，降低训练权重，不扩写成事实经历。

### 2.5 长期记忆、压缩与纠正样本

```json
{
  "sample_id": "memory_060",
  "persona_id": "person_a",
  "history_events": [
    {"event_id": "m01", "event_time": "2026-01-02", "text": "对方希望月底讨论旅行，但日期尚未确定。", "source_ref": "session_010_turn_03", "support": "direct_statement"},
    {"event_id": "m02", "event_time": "2026-01-10", "text": "对方更正为下月出行。", "source_ref": "session_018_turn_05", "supersedes": "m01", "support": "direct_statement"}
  ],
  "budget_tokens": 96,
  "target_compression": {"text": "旅行计划已改为下月，具体日期仍未确定。", "must_keep": ["date_not_fixed", "updated_month"], "source_refs": ["m01", "m02"], "obsolete_claims": ["travel_this_month"]},
  "future_context": "对方再次提到准备旅行用品。",
  "retrieval_positive_ids": ["m02"],
  "retrieval_hard_negative_ids": ["other_person_trip"],
  "preferred_action": "discuss_preparation_without_inventing_date",
  "future_target_text": "上次说改到下月了，具体日期定下来了吗？"
}
```

摘要忠实度包括人名、日期、否定、条件、因果、证据强度和修正历史；未来任务不只做问答，还做自然讨论中的适当引用、避免重复分享、履行承诺。记忆不能因为经常被检索就自动变真。

### 2.6 神经声学桥样本

```json
{
  "sample_id": "voice_bridge_205",
  "persona_id": "person_a",
  "speaker_id": "voice_a",
  "source_id": "authorized_voice_session_02",
  "target_text": "我以前也遇到过类似的选择。",
  "target_audio": {"uri": "media/voice_a/u205.wav", "sample_rate": 24000},
  "reference_audio": {"uri": "media/voice_a/ref07.wav", "sample_rate": 24000, "same_as_target": false},
  "tokenizer": {"model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "file": "speech_tokenizer_v3.onnx", "revision": "pinned_revision_required", "frame_rate_hz": 25, "vocab_size": 6561},
  "target_speech_tokens_ref": "speech_tokens/cv3/u205.npy",
  "text_to_audio_alignment_ref": "labels/u205_spans.json",
  "semantic_hidden_ref": "features/thinker_revision/u205.safetensors",
  "semantic_hidden_shape": [18, 2048],
  "style": {"emotion": "reflective", "speed": "normal", "emotion_valid": true},
  "commitment_spans": [{"text_start": 0, "text_end": 13, "claim_ids": ["autobiography_003"], "allowed": true}],
  "loss_region": {"reference": false, "prompt": false, "target": true, "padding": false}
}
```

初始接口对齐阶段的 `semantic_hidden_ref` 可以来自冻结 Thinker 对真实文本的 teacher forcing；到持续交互阶段必须按各时刻当时可见的感知和已生成前缀在线重算，不能喂入整句未来语义。冻结特征缓存只在编码器与条件不变时有效。进入联合解冻后相应缓存必须禁用或重建。

## 3. 公开数据：下载入口、能训练什么、仍缺什么

下表“可下载”包含需要申请的公开研究数据，不表示已在本项目下载，也不表示许可覆盖所有用途。代码许可证与媒体/数据条款分别记录；版权来自影视的样本不能因仓库有开源许可证就视为全部授权。

| 数据与官方获取入口 | 获取/使用边界 | 本项目用途 | 无法替代的自建内容 |
|---|---|---|---|
| [AMI](https://groups.inf.ed.ac.uk/ami/download/) | 公开下载标注、阵列音频、低分辨率视频；页面说明 CC BY 4.0，完整尺寸视频的获取方式不同 | 空间音频、speaker、重叠讲话、多视角时间同步；使用训练会议，评测会议隔离 | 人机互动的暂停/继续意图、目标人物经历、云端状态 |
| [Ego-Exo4D](https://docs.ego-exo4d-data.org/) | 官方 Getting Started 申请访问、接受数据协议后用 downloader；同步多视角和7通道音频，不是无需协议的直接包 | 视角关联、事件时间、空间音频和预留传感器接口 | 日常交谈分寸、人格/价值判断 |
| [WenetSpeech](https://wenet-e2e.github.io/WenetSpeech/)、[下载仓库](https://github.com/wenet-e2e/WenetSpeech) | 官方页要求表单申请；明确非商业用途及原音频权利归属；不能用repo Apache替代数据条款 | 中文音频连接器；先S/M子集，再按需扩L | 情绪动力学、完整实时互动、人物目标音色 |
| [AudioSet](https://research.google.com/audioset/download.html) | 直接提供标签、YouTube ID与时间和预提取特征；不提供完整原始波形包；标签数据CC BY 4.0，ontology另有条款 | 环境声事件头；有合法可得波形才能重编码本项目特征 | 低延迟强标注、具体设备空间定位 |
| [AISHELL-3 / SLR93](https://www.openslr.org/93/) | 直接音频/文字包；Apache-2.0；约85小时中文多说话人中性语音 | C2中文speech token与音色稳定性 | 丰富情绪、目标人物独有风格 |
| [LibriTTS-R / SLR141](https://www.openslr.org/141/) | 直接 train/dev/test 包；CC BY 4.0；24kHz朗读语音 | C2英文桥接与内容一致性保持 | 自然对话情绪、目标人物经历 |
| [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) | README有下载入口；需完成协议，数据仅研究用途，不能把repo MIT当数据许可 | 中英相同内容的情绪语音对照；speaker与风格解耦 | 情绪产生的原因、目标人物情绪演化 |
| [MELD 官方仓库](https://github.com/declare-lab/MELD)、[官方数据页](https://huggingface.co/datasets/declare-lab/MELD) | 作者给下载包/标注；影视来源，数据与素材条款单独核验 | 对话上下文中的他人情绪识别；独立train/dev/test | 模型自身情绪、中文人物价值取舍 |
| [中文情绪 M3ED / RUCM3ED](https://github.com/AIM3-RUC/RUCM3ED) | 官方仓库有情绪/关系标注与splitInfo，并给音频及特征下载入口；因版权问题视觉只发布特征，未提供完整原始视觉视频；使用条款需向发布方核验 | 中文情绪和关系读出头的可选补充；音频可按取得的素材重新编码，视觉特征只能用于对应特征空间的辅助头 | 不能用其视觉特征训练本项目原始视频编码器；不能替代人物自身的情绪动力学 |
| [MiniMind-O dataset](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset) | 公开 parquet；来源混合，需记录上游数据与合成来源条款 | MiniMind/Mimi分支接口复现；mini子集主要英文无视觉 | 不能当CosyVoice3或Qwen3-Omni码本标签；不能用mini证明中文人物语音成熟 |
| [LoCoMo](https://github.com/snap-research/locomo) | 仓库含 `data/locomo10.json`；模型生成与人工整理混合的评测数据 | **仅保留评测**长期事件与时间推理；不运行其API生成脚本 | 本项目的训练经历、人物素材、真实关系演化 |
| [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval) | 官方发布文件，模型卡MIT | **仅保留评测**更新、时间与跨会话记忆 | 主动使用记忆和不重复分享的训练 |

中文人物动态情绪数据以自建为主，不能把“他人语音被标注为愤怒”直接作为“模型此刻应愤怒”的标签。上表 M3ED 指 RUCM3ED 中文情绪对话数据，与 m3ed.io 的同名机器人数据不同。其预提取视觉特征不能假定与 Qwen3-Omni 编码空间兼容，接入前需要专门投影与训练。

大规模公开图文指令不是首期硬性依赖：C0已包含视觉能力，S1先用AMI/Ego-Exo训练分片和自建场景做连接验证；只有自训连接明显损失通用视觉能力时，再添加经来源核验的图文训练语料。

### 3.1 必须自行准备的六类数据

1. **目标人物资料库**：可追溯事件、原话、观点、价值冲突、后期改观；每条分“明确事实”“有依据推测”“未知”。起步3人物而非只1人物，便于验证可替换性。
2. **人物条件偏好对**：同情境多种反应，有些表面语气相似但价值判断不同；加入沉默、简短回应、不给建议等候选。最初每人物1千组，双人复核20%。
3. **连续互动轨迹**：时间同步的输入、表达计划、实际播放、插话、交谈对象、speaker、可接受动作集合。初始100小时中至少20%含重叠、20%有可接受沉默、20%有重要环境变化，可重叠统计。
4. **云端协作 fixtures**：人工有证据结论与失败/延迟/过期回执；10万决策点中至少各覆盖1千个timeout、unavailable、stale、conflict案例。断网后相同事实标签保持不变。
5. **后天长期经历**：每人物至少20条跨10次会话的系列，含承诺、纠正、反复提及、关系变化、久远但重要的事件；增加与当前无关却词面相似的干扰记忆。
6. **目标人物声学**：有文本和说话人身份的干净片段、独立参考语音、不同情绪/语速，同一说话人的多个录音条件。动漫音轨中的配乐、混响和他人声需分离或剔除，并记录处理痕迹；素材不足时先限制风格范围。

合成仅用于可控因果/时间/故障增强；不把合成角色经历写入真实人物事实库。远程API不产出训练样本；可以用本地已有工具做对齐、特征提取和标注辅助，但人类依据与自动推测要区分。

## 4. 每个待训练模块的损失与梯度去向

### 4.1 共同记号和归一化

对任务 `j`，标签有效掩码为 `m_ij`，标注可信度 `w_i` 在 [0,1] 内：

\[
\bar L_j=\frac{\sum_i m_{ij}w_i\ell_{ij}}{\max(1,\sum_i m_{ij}w_i)}.
\]

文本、语音分别按有效目标 token 归一化，再组合任务均值；不按全批原始 token 数混算。公共多说话人数据按 speaker 重采样，人物数据按 persona 重采样，避免大数据人物覆盖小数据人物。所有任务在分布式进程间同时归约分子分母，不平均各卡不等长的局部均值。

| 待训练部分 | 样本格式 | 主损失 | 梯度到达与边界 |
|---|---|---|---|
| audio/vision MLP、resampler、状态读取Adapter | 2.2 | 文本任务CE＋配对InfoNCE＋时间对齐辅助 | 经冻结Thinker对连接层输入求导；冻结编码器输出可缓存 |
| 空间音频连接、speaker关联头 | 2.2 | 多标签BCE＋speaker对比＋有标注DOA的圆周损失 | 输入编码器冻结；未知/混合speaker不强行单标签 |
| H_t、事件边界、快交互头 | 2.3 | 行为CE、时间损失、事件边界BCE、可选未来事件预测 | TBPTT窗口内反传；窗口边界detach明确记录 |
| claim支持度、外部请求、云端状态使用 | 2.3 | 支持度CE/Brier＋请求BCE＋状态条件行为CE | 返回fixture是常量观测；不反传API |
| 承诺选择与语音提交 | 2.3/2.6 | 可表达span BCE＋错误肯定断言惩罚＋正确表达覆盖约束 | 离散提交用监督；不通过实际播放反传 |
| 人格、评价、情绪、关系 | 2.4 | 人物SFT＋偏好DPO＋有标注情绪/评价回归＋条件时间正则 | 更新人物Adapter和共享评价网络；不监督未标注潜维度 |
| 记忆query/重排/写入/压缩级别、文字bank投影 | 2.5 | 检索NCE、写入与级别CE、预算惩罚、摘要CE、忠实度标签 | top-k/磁盘写入离散；选择器通过标签和后续策略学习；reader梯度到达选中文字的bank投影 |
| C2神经语音桥与语音LM适配 | 2.6 | speech-token CE＋对齐损失＋可选声学条件损失 | teacher forcing到桥和LM；tokenizer标签生成冻结 |
| flow/HiFT（后期可选） | 2.6 | 原生flow matching；vocoder原生谱/对抗目标 | 需可微训练实现；不把ONNX推理图当训练图 |
| 联合解冻的Thinker/感知 | 所有已适配格式 | 多任务损失＋原能力保持KL/回放；MoE负载均衡 | 仅对明确开放部分反传；编码器缓存同步失效 |

### 4.2 连接与时间对齐

任务文本监督：

\[
L_{text}=-\frac{1}{N_y}\sum_{u\in target}\log p(y_u\mid y_{<u},x_{\le t},H_t,P).
\]

系统提示、用户输入、参考音频、padding和仅用于条件的记忆不作为回答重构目标。口语语气信息由声学/情绪任务保留，不能只用ASR目标训练所有连接层。

InfoNCE：

\[
L_{align}=-\frac1B\sum_i\log\frac{\exp(s(z_i,t_i)/\tau)}{\sum_{k\in\mathcal N_i}\exp(s(z_i,t_k)/\tau)}.
\]

初始 τ=0.07；分母候选集合包含配对正例和负例。同一个事件的其他视角/字幕必须标为多正例或从负例中排除。空间方向有标签时用 `1 - cos(预测方向 - 真实方向)`，而不是跨越±180°时错误的普通MSE。时间对齐头预测音视频偏移桶，正确同步作0桶，合成偏移为可控标签；只在确实存在对应声源时计损失。

S1起始组合 `L_text + 0.1 L_align + 0.1 L_sync + 0.1 L_event`；先检查每项梯度量级，不能用对比损失很好看替代下游理解提升。

### 4.3 状态、全双工行为与等待

若时刻 `t` 有多个合理动作，使用可接受动作集合 `A_t`：

\[
L_{act}=-\frac1{N_t}\sum_t\log\sum_{a\in\mathcal A_t}\pi(a\mid H_t,P,e_t,cloud_t).
\]

停止时刻只在足以确定需要停止的样本上计算：

\[
L_{time}=\mathrm{Huber}((\hat t_{stop}-t_{stop}^{*})/100\mathrm{ms}).
\]

再用“未来500ms是否适合开始回应/停止”的BCE辅助头处理没有单点时间标注的样本；没有标注的静默时段不自动视为负例。评价按“足够证据可见”到停止的时间，同时报告从声学起点到停止的总延迟。

事件边界BCE预测什么时候封存一段经历。H_t的下一事件预测仅是辅助目标，不能迫使状态保留全部像素；先用事件类别和待办变化，不做昂贵的视频重构。

S2起始 `L_act + 0.2 L_time + 0.2 L_boundary + 0.5 L_claim + 0.2 L_request + 0.2 L_commit`。模拟云端延迟取0.2/1/3/8秒与timeout，训练批中同时包含无需云端的样本。WAIT_BRIEF不是必选动作；对有用澄清、一次简短说明和适当沉默都给合理标签。

### 4.4 结论校准与承诺

对每条claim先预测 SUPPORT、CONTRADICT、INSUFFICIENT、SUBJECTIVE 的分布；同时对可核验事实预测“当前证据不足”的概率 `q_i`，使用Brier：

\[
L_{claim}=CE(s_i,\hat s_i)+\frac1N\sum_i(q_i-y_i^{insufficient})^2.
\]

`SUBJECTIVE`不进入事实正误的Brier分母。云端结果与证据冲突时，标签依据经审查的证据，不依据模型大小；已过时但以前正确的事实需单独标时效失效。

承诺头对span给出 `allow/qualified/defer`。首期用交叉熵＋配对样本训练；离散门控不可直接对音频播放求导。联合阶段可在teacher-forced候选计划上惩罚未经支持却被肯定表达的概率，同时维持可回答样本的覆盖率下限。独立记录 `unsupported_assertion_rate`、`supported_answer_coverage` 和 `unnecessary_wait_rate`，防止“永远不说话”获得高分。

校准只使用验证集选择温度或阈值，测试集报告可靠性曲线、Brier以及风险—覆盖曲线，不直接把softmax最大值命名为“真实可信度”。

### 4.5 人格、情绪与人物隔离

人物监督学习与DPO采用：

\[
L_{DPO}=-\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y^+|x,P)}{\pi_{ref}(y^+|x,P)}-\log\frac{\pi_\theta(y^-|x,P)}{\pi_{ref}(y^-|x,P)}\right]\right).
\]

`ref` 是人物SFT后冻结的本地参考模型，β先取0.1；两候选尽量控制长度差，避免只学更短或更长。每个人物独立偏好标签；不以统一“更礼貌、更积极”代替目标人物价值取舍。[DPO 论文](https://arxiv.org/abs/2305.18290)

情绪隐状态32维，只对有标签读出使用Huber/分类损失。状态动力学使用目标人物情绪基线、事件评价和时间间隔；无新重要事件时可加较弱 “回归基线” 正则，有显著事件时关闭该平滑项：

\[
L_{affect}=m_{label}\,\mathrm{Huber}(D(e_t),v_t)+\lambda_0 m_{noevent}\Vert e_t-\hat e_t^{decay}\Vert^2.
\]

`D`输出3维可解释读出；评价16维可部分标注，关系64维通过关系任务读出监督。没有标签的维度通过行为/记忆/声学任务间接学习，不强行拟合虚构真值。

人物隔离用反事实批次：同场景换P、同P换经历、同人格换声线。训练时使用人物命名空间掩码；评测分别检验应改变的价值判断和不应改变的客观事实。不要对全部输出加“人物间必须不同”的损失，否则会制造无意义差异。

S3起始 `L_text + 0.2 L_DPO + 0.2 L_affect + 0.2 L_appraisal + 0.2 L_memory_source`。`L_appraisal` 对有标注的评价读出计算逐维Huber，缺失维度掩码为0；`L_memory_source` 为候选记忆是否支持该人物经历陈述的三分类CE（支持/矛盾/证据不足），并以正确source_id检索监督辅助。DPO在完成SFT后启用。具体事件日期优先来自记忆检索，不能因人物LoRA学到流畅故事就认定有事实依据。

### 4.6 记忆读写和压缩

检索使用正例/难负例InfoNCE；写入与压缩等级采用人工标注的交叉熵。压缩等级首设 KEEP_DETAIL、EVENT_SUMMARY、SEMANTIC_FACT、DROP，每级有token成本 `c_k`：

\[
L_{budget}=\left[\max\left(0,\frac{\sum_{i,k}p_{ik}c_k-B}{B}\right)\right]^2.
\]

这里对预测概率的期望成本可微；推理时选择某级与写入磁盘是离散操作。摘要用来源支持的目标CE，并用多任务标签保留人物、时间、否定、条件和纠正关系。未来讨论/任务的CE监督读取后的有效性：

\[
L_{mem}=L_{retrieval}+L_{write}+L_{level}+L_{summary}+0.1L_{budget}+0.5L_{future}.
\]

首期使用真实相关记忆与模型检索结果混合训练reader；不能声称 `L_future` 已经穿过硬top-k更新检索器。检索器通过其正负例监督更新。后期如用策略梯度训练写入/选择，需独立对照和方差控制，不默认为必要起点。

### 4.7 声学桥、语音LM与波形网络

三个码本体系的边界：

| 分支 | 可获得的目标token生成方式 | head/码组 | 时序与梯度 |
|---|---|---|---|
| Qwen3-Omni原配基线 | 本次核验未确认公开独立音频编码器足以生成任意真实训练音频目标codes；不承诺此训练路径 | config含16码组；Talker主词表3072，辅助预测器2048，Code2Wav另有语义码本4096；不可统一写成单一词表 | 以原processor/code2wav时序为准，不能从位置编码13直接推断codec帧率；首期原配声学冻结 |
| CosyVoice3/C2 | 公开 `speech_tokenizer_v3.onnx` 编码目标波形，冻结后离线存储 | 单路speech token词表6561；特殊符号以原head配置为准 | 25Hz token、24kHz输出；teacher forcing，不通过采样回传 |
| MiniMind-O/Mimi消融 | 原配Mimi encoder，将码组数量配置为8 | 8组，codec码本与MiniMind含特殊符号的head分别读取原配置 | 12.5Hz、24kHz；不能把Mimi数据直接喂给CosyVoice或Qwen head |

CosyVoice3主要目标：

\[
L_{speech}=-\frac1{N_c}\sum_{n\in target}\log p(c_n\mid c_{<n},B(H^{commit}),V).
\]

在接口热身阶段增加归一化特征对齐 `Huber(norm(bridge_hidden), norm(native_text_condition))`；对齐在span/单调注意层面，不要求异源token一一对应。参考语音、提示token和padding损失为0；把参考语音与目标语音分开录制/选段，防止模型复制目标。

C2初始 `L_speech + 0.1 L_feature + 0.1 L_style`。`L_style`来自条件分类/读出监督，只对已标注样本计分。speaker相似度、CER/WER、MOS先作为评测；只有训练图确实输出可微波形时才将speaker或声学损失反传到相应可微网络。

若开启flow训练，使用其原生conditional flow matching：

\[
L_{CFM}=\mathbb E_{t,x_0,x_1}\Vert v_\theta(x_t,t,cond)-u_t(x_0,x_1)\Vert^2_{mask}.
\]

插值路径和mask采用发布recipe，不能自行把公式简化后声称复现原模型。HiFT继续使用原生谱/对抗训练配置；先独立稳定后联合。离散speech token输入切断flow损失到上游语义/语音LM的梯度，除非另行实现并验证连续松弛；首期不依赖该松弛。

### 4.8 联合训练与原能力保持

\[
L_{joint}=\sum_j\lambda_j(s)\bar L_j+\lambda_{keep} KL(\pi_{base}\Vert\pi_\theta)+\lambda_{moe}L_{balance}.
\]

`s`为阶段；`base`为本地冻结基线，在通用保持样本上提供logits，未使用远程API。保持KL只约束应保持的通用能力，不能在人物主观偏好样本上强迫回到通用助手人格。MoE解冻路由器时保留原负载均衡机制并监控专家使用；系数起点沿原配置，不让新增侧路把所有输入挤到少数专家。

每200–500步检查任务梯度范数与冲突，必要时降低某项loss系数或分阶段训练。不得只凭总loss下降判断人格更像或交谈更自然。

## 5. 冻结、缓存与梯度边界清单

1. **冻结参数不等于 `no_grad`。** 如果训练投影/Adapter的loss位于冻结Thinker之后，Thinker权重不更新，但必须保留对其输入的反向传播。把整个接收模块包进 `no_grad` 会切断桥训练。
2. 冻结感知编码器位于训练桥之前时，其输出可以离线缓存；缓存键包含模型revision、预处理、时间窗口、视角和有效通道。联合解冻编码器后不能继续使用旧缓存替代前向。
3. 推理运行时、量化内核、ONNX图和KV调度器不自动支持训练。训练路径使用支持autograd的实现，`use_cache=false`或专用可微训练缓存；部署路径另行验证数值。
4. 检索top-k、数据库读写、API请求、token采样、音频设备播放均是离散/外部边界。为其控制器配置监督学习或后续策略优化；不能画一条箭头就声称梯度跨过边界。
5. 状态TBPTT初始每32个100ms tick（3.2s）截断一次，前16tick作可选burn-in；跨更长记忆主要由显式事件库和延迟任务监督承担。扩到128tick前先核对激活显存和梯度稳定。
6. 冻结原配Talker不代表其行为永远不变：Thinker状态分布改变也可能影响声音。每阶段保留固定声学回归集；新增神经桥出现音质漂移时先缩小残差/恢复基线，不用更多人物数据掩盖接口问题。

## 6. 集群训练切分与内存计划

### 6.1 先做8卡试验，再扩大

S1–S3先按单节点8×H200训练，使用BF16、activation checkpointing、长度分桶。起始每GPU microbatch=1；文本上限4k，音频轨迹10–30s，视频最多4视角×2fps×8s，经有界重采样进入状态。有效批量先64个序列，靠梯度累积取得，不能把8卡全部显存视为可随意增大单样本。

冻结主干可在每GPU复制，但留给激活的空间不足时采用节点内FSDP/分模块放置。只训练旁路时，不必一开始跨6个节点同步0.2B参数；同时跑3个种子的独立实验更能验证机制。

S5主干全参训练首选**节点内8卡FSDP分片、节点间数据并行副本**，使参数all-gather主要留在NVLink域；跨节点同步梯度。先使用常规MoE实现，检查负载再引入EP，避免无测量依据地同时叠加TP/EP/PP。

需要进一步扩大时：TP限制节点内，EP只有在实际专家负载/通信profiling证明确有收益时启用；长序列优先缩减感知token和checkpoint，再考虑context parallel。48卡训练池可采用6个8卡组；对不支持相应并行组合的定制模块，使用独立数据并行和明确的同步边界。

此处FSDP是参数分片的数据并行，各rank仍处理自己的microbatch；因此48卡、每卡microbatch=1、梯度累积2时，有效批量为96个序列，而不是6个序列。8卡pilot的有效批量64不能原样套到48卡；扩容可用96/192起步，并按有效目标token控制学习率和对照实验的样本暴露量。TP/EP启用后再按真实数据并行组数重算，禁止重复乘GPU数。

[PyTorch FSDP2文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)说明按参数分片及前后向all-gather机制；具体模型与混合冻结模块的兼容性要在8卡版本先验证。

### 6.2 显存估算口径

原配完整checkpoint约70.5GB文件，按BF16近似约35B量级驻留参数估算，不精确等同可训练参数计数。Adam混合精度全参训练静态状态常用16–18字节/参数的规划范围：

- 若开放约35B参数，参数/梯度/优化器总计约560–630GB，8卡均分约70–79GB/卡。
- 再加激活、all-gather峰值、MoE临时缓冲、感知网络工作区和CUDA/通信余量，才与141GB显存比较；不把70GB直接当作总训练显存。
- 0.2B新增参数约3.2–3.6GB训练状态；但经冻结Thinker反传时，激活可能比新增参数占用更大。
- 若使用纯BF16优化器、不同FP32主副本方案、8-bit optimizer或不同梯度精度，逐项重算，不能用同一个16–18字节公式代替实测。

为训练报告记录每卡 `allocated/reserved/peak`、吞吐、all-gather时间、读盘等待、有效目标token/s。检查点和特征分片先落节点NVMe，再异步汇总；故障恢复包含优化器、采样器、随机数、数据版本和任务loss权重。

实时服务不使用训练的FSDP逐层all-gather路径；推理部署使用单卡/节点内TP及独立副本。训练时吞吐最优不代表实时P95最优。

### 6.3 优化器起点

| 参数组 | 起始学习率 | 初始设置 |
|---|---:|---|
| 新MLP、resampler、状态/策略头 | 1e-4–3e-4 | AdamW，warmup 3%，clip norm 1.0，weight decay 0.01 |
| 新cross-attention与承诺桥 | 5e-5–1e-4 | 输出近零初始化；保留可测的残差门控 |
| 人物LoRA | 5e-5–1e-4 | rank先16/32，按人物独立存储；beta/学习率通过验证集选择 |
| 预训练Thinker/语音LM后部层 | 2e-6–1e-5 | 不与新模块共用一个学习率 |
| 感知末层与flow后期适配 | 1e-6–5e-6 | 小步开放，保留原任务回放 |

使用3个种子的小规模pilot后确定训练步数；不用“训练若干epoch”替代token/小时暴露量记录。数据重复率、人物采样权重和是否缓存特征必须进入实验配置。

## 7. 数据切分、验收和消融

### 7.1 切分规则

- 自建数据起步按主体会话组70/15/15切分，固定test；同一次录制的视角、音频通道、字幕和相邻窗必须同组。
- 人物复刻分“同人物新情境”和“全新人物包”两个测试；前者同人物但不重叠事件来源，后者测试共享机制的可迁移性。
- 声学测试分已见说话人新文本、未见说话人参考克隆、已见人物新情绪；参考和目标不能来自同一段音频。
- 长期记忆按时间向后预测；目标人物未来剧情/未来访谈不能出现在历史快照。记忆库只含测试时该人物应已知的内容。
- LoCoMo、LongMemEval只作外部测试，不把其答案用于训练或调阈值。公共数据原有test保持隔离。

### 7.2 验收矩阵

下列值是用于首轮实验决策的**起始阈值**，不是已达到的结果。先以C0与简单规则/检索对照测量基线，再锁定阈值；失败项保留，不通过改口径隐藏。

| 目标 | 检查方式 | 起始通过标准 |
|---|---|---|
| 因果时间与人物隔离 | 遍历事件可见时间、来源、persona/session/request | 100%确定性检查通过；任何跨人物实际记忆泄露均阻止发布该实验版本 |
| 感知连接有效 | 固定主干比较有/无模态、随机桥、已训练桥 | 多模态证据任务显著优于随机桥；通用回归相对退化不超过5%起点 |
| 全双工 | 分别统计该停/不该停/澄清/旁人讲话 | STOP召回与错误停止率同时报告；起始错误停止率<5%，需要停止召回>90%，区间按会话bootstrap |
| 交流分寸 | 盲评分享价值、重复、无意义插话；记录沉默机会 | 对固定填充话术/逐帧描述对照胜率>60%，并保持有效回应覆盖 |
| 结论承诺 | 支持、冲突、未知、主观、过期分组 | 未支持的肯定断言率<2%起点；有支持问题回答覆盖>80%；不以静默达标 |
| 人物一致性 | 换声线与换人物条件的交叉盲评 | 价值/经历依据和语言风格分别打分，人物识别显著高于机会水平；无素材经历编造单列 |
| 记忆效用 | 固定预算，比较recency-only/RAG/学习压缩 | 关键事件与修正关系保持提升；无关内容引用率和重复分享率不升高 |
| 目标声音 | 原生文本TTS与神经桥AB，CER/WER、speaker、自然度 | 中文CER相对基线增幅≤10%起点，独立音色/自然度盲评无明显退化；不只靠embedding相似度 |
| 失效退化 | 10–30%随机模态dropout＋持续断流/恢复 | 仍可使用可用模态；无陈旧状态伪装为新观测，正常模式性能也保持 |
| 资源与时延 | 单会话及逐步加并发，记录P50/P95/P99 | 满足架构文档的起始预算；训练/批任务不污染实时队列 |

### 7.3 必做消融与负对照

1. 原配C0 vs C1；C1去掉持续状态，仅扩大上下文，控制新增参数/输入预算。
2. 仅人物文本档案 vs 仅人物Adapter vs 混合PersonPack；不允许三者训练素材规模不同。
3. 直接Thinker hidden→Talker vs 承诺计划桥；同时测不支持结论泄露和过度等待。
4. 固定等待话术 vs 云端状态条件策略；同一内容对照延迟/离线/过期。
5. 只按时间遗忘 vs 重要性/人格影响压缩；存储预算相同。
6. 只识别对方情绪 vs 增加自身情绪动力学；测情绪连续性及事实判断是否被错误改变。
7. CosyVoice3文本基线 vs 文本＋神经条件 vs 纯神经条件；不得把文本基线结果当神经桥结果。
8. 单人物训练共享主干 vs 多人物共享训练；检查换包后残留的人物偏好。

研究报告同时保存失败样本、置信区间和计算成本。只有在数据、参数规模及实时预算受到控制时，才能判断新增机制带来的效果。

## 8. 首批可交付实验资产

未来实现阶段首先产出以下资产，再启动大规模训练：

- 模型与数据版本清单，含权重、tokenizer、处理器和许可来源。
- 500条冻结基线结果、100个全双工对照轨迹、3个人物的最小PersonPack。
- 上述六类格式的各50条人工审查样本，验证时间可见性、标签掩码和来源可追溯。
- 8卡小训练的梯度检查报告：确认目标loss确实到达连接层，冻结主干无权重更新；确认API/top-k/采样边界没有被误写成可微。
- 一份按模块分项的显存、数据吞吐和实时延迟报告。
- 固定测试集与消融配置。完成这些后，才把同一路线扩到48卡或更大规模。

这个顺序用于尽早暴露接口、数据与行为定义问题。增加集群规模是扩展已验证实验的手段，不能替代人物素材、持续交互标签和可证伪的研究假设。
