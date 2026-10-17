# 单卡 RTX PRO 6000：待训练模块、数据格式与优化目标

版本：v0.1；主要资料核验日期：2026-10-13；整理与补充复核日期：2026-10-14。配套架构：[03-rtx-pro6000-architecture.md](03-rtx-pro6000-architecture.md)。人物专用细节：[07-character-memory-emotion.md](07-character-memory-emotion.md)。本文给出可落地的起始实验，不声称以下规模和超参数已验证。

硬件假设为单张 RTX PRO 6000 Blackwell Workstation Edition 96 GB。主选 checkpoint 为 `openbmb/MiniCPM-o-4_5`，共享模型先冻结；本地新增状态宽度 1024、64 槽，人物条件 `p=256/e=32/r=64/a=16`。首期云端仅在运行时推理：**不训练云端、不通过 API 生成或标注训练集**。训练中的云端状态和回复为人工编制、已有可核验证据配对的模拟观测。

## 1. 可复现训练前置工作与梯度边界

### 1.1 不能直接把流式推理 SDK 当训练循环

官方 `prepare/streaming_prefill/streaming_generate` 提供推理接口，不能据此假定音视频到语音的训练路径都保留梯度。实施时首先完成以下接入验证，再启动正式训练：

1. 固定模型 revision、config、processor/tokenizer、音频 tokenizer 与 Code2Wav 版本；记录主干/Talker/新增模块参数清单，避免加载两套主干。
2. 定位普通 `forward`、特征注入、语义→Talker 条件、文本/音频 logits，检查 `no_grad/inference_mode/detach/numpy`、离散采样和跨线程边界。参考源码入口为 [modeling_minicpmo.py](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py)。
3. 建立**离线 teacher-forced training forward**：把已对齐的输入和目标按 1 秒块展开，以因果 mask 计算损失；保留 200 ms 状态更新的时间索引。训练 forward 与在线接口共享权重和 token 定义，但不调用带无梯度装饰的 generation 函数。
4. 单个小样本检查：损失有限；每个计划训练的连接层梯度非零；冻结参数无梯度/不更新；连接层后面的冻结主干对输入可求导。
5. 做 32–128 条样本的过拟合实验和梯度数值抽查。若 loss 不下降，先排查对齐、mask 和断梯度，不扩大数据集。
6. 用相同前缀比较离线 forward 与在线一步 logits/状态；训练时不能看到在线时尚未到达的帧、词或 API 返回。

冻结权重不等于冻结整个计算图。若 $z=P_\theta(E(x))$，$y=F_{\text{frozen}}(z)$，仍需计算 $\partial L/\partial z$ 更新 $P_\theta$。只允许把**可训练路径前方**且不更新的编码器 $E$ 用无梯度/缓存运行。桥后冻结主干或 Talker 不能整体 `no_grad`。冻结特征缓存必须记录 encoder revision、预处理配置、采样时刻与特征是否使用未来上下文。

离散 token 采样、硬 top-k 检索、文件写入、播放器与云端 API 均无普通反向传播。首期通过独立监督训练选择器；生成部分用真实目标 token 的 teacher forcing。后期如需要轨迹级奖励，明确用策略梯度或可微替代实验，不把评测分数直接写成可反传 loss。

## 2. 训练阶段、冻结集合与规模起点

| 阶段 | 更新参数 | 冻结/旁路 | 数据与初始规模 | 阶段通过条件 |
|---|---|---|---|---|
| T0 基线与数据检查 | 无 | 整套 pretrained 冻结，新结构旁路 | 50–100 个短会话、100 个边界/故障片段 | 原生语音/图像路径及分块输出复现；来源和时间标签可读 |
| T1 连接层可行性 | C1/C2/C3 投影与重采样、C4 条件残差、对应辅助读出 | 主干、原生 encoders/Talker、人物适配冻结；S1 用无参数聚合旁路 | 5万–20万短样本；视听对齐、讲话人、空间、语音条件 | 对齐任务明显超过随机投影；原有能力下降在预设容差内 |
| T2 新机制学习 | S1 状态、S2 评价/情绪、S3 策略/claim、S4 记忆及相关连接 | 原有模型仍冻结；人物先用少量占位或已整理条件 | 2万–10万条 16–32 秒轨迹；2000–10000 个长期记忆 episode | 语义打断/等待/记忆在反事实测试超过无状态与固定策略 |
| T3 人物专用适配 | PersonPack embedding/adapter、风格连接，必要时主干/Talker LoRA | 基础共享权重冻结，冻结稳定的感知连接 | 每人物 1000–5000 个经审查事件、5000–20000 个情境回应、1000–3000 个偏好对；不足时从更小集开始 | 角色未见情境偏好与经历忠实度有效提升，无明显串人物 |
| T4 分组联合 | 新模块＋选定原生投影/Talker；随后主干少量层或 LoRA | 编码器先保持冻结，逐组开放 | 混合公共保持任务与自建轨迹；先 8k token、microbatch 1 | 综合提升且非目标模块无明显回退 |
| T5 更大规模验证 | 按实验需要扩大解冻范围 | 9B 全量 Adam 不作为单卡默认配置 | 迁移集群或专门评测 CPU/offload 配置 | 以实测资源与泛化收益决定，不以“全部解冻”本身为成功 |

T1 仅验证已有表征能否通过新连接被使用，不声称此时已经获得人格或持续记忆。新增随机状态/策略网络不能在冻结状态下自动学会新能力。规模数字是数据准备预算，不是最低充分样本数；先做小规模学习曲线，检查数据多样性和标签一致性后再扩大。

初始训练计划：AdamW；新投影学习率 `1e-4`，状态/策略 `5e-5`，人物 LoRA `2e-5`，分组解冻主干 `2e-6～5e-6`；warmup 3%，cosine decay，weight decay 0.01，梯度裁剪 1.0。BF16、microbatch 1、累计 16 次起步；rank 16 LoRA。局部冻结特征训练可增加 batch，联合训练采用激活检查点。每种配置先测显存峰值及 100 步吞吐；超出预算先缩短片段/降低视频采样，不盲目更换更大 GPU。所有数值均为待调起点。

## 3. 数据集来源：哪些可以下载，哪些需要自建

以下链接是官方发布者入口。数据取得方式与具体使用条款随版本核验并写入 manifest；仓库代码许可不能自动代替音视频素材的许可。无需在本次总体设计阶段下载数据。

| 数据/入口 | 获得状态与内容 | 本档用途 | 局限及需自行补充 |
|---|---|---|---|
| [AISHELL-1 / OpenSLR 33](https://www.openslr.org/33/) | 可公开下载语音、转写和说话人资料 | 中文语音语义保持、噪声/重采样对照 | 朗读/较干净，不能训练自然打断与人物经历 |
| [AISHELL-4 / OpenSLR 111](https://www.openslr.org/111/) | 可下载，CC BY-SA 4.0；8 通道会议录音、转写和讲话人活动 | 阵列/说话人连接、重叠讲话、远场鲁棒性 | 没有目标人物情绪与云端等待标签；自建音源空间和策略标签 |
| [AMI 下载入口](https://groups.inf.ed.ac.uk/ami/download/) | 公开信号/转写入口，页面注明 CC BY 4.0；小尺寸视频、阵列和近讲音频 | 多视角关联、对话状态、发言与附和 | 页面不直接提供全部高分辨率视频；会议不等于私人聊天，需重标交流意图 |
| [Ego-Exo4D 官方文档](https://docs.ego-exo4d-data.org/) | 官方申请/同意条款后按入口获取，多视角活动数据 | 多视角身份/事件融合；可选扩展 | 不作为中文人格聊天来源；不将申请制写成无需手续的直接下载 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 官方仓库有音频和五折划分 | 环境声音连接与扰动验证 | 小规模、单声源倾向；无可泛化的紧急程度标签 |
| [AudioSet](https://research.google.com/audioset/download.html) | 官方提供标签 CSV 与部分预提取特征；音频依赖原视频获取 | 扩展声音类别，弱监督环境事件 | 不是完整原始音频压缩包；视频可能失效；128 维特征不能当 BEATs 输入特征替换 |
| [AISHELL-3 / OpenSLR 93](https://www.openslr.org/93/) | 可下载中文多说话人语音和转写，页面标注 Apache 2.0 | 内容/音色分离与 Talker 条件保持 | 主要中性语音，不提供自然情绪轨迹 |
| [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) | 有下载入口；需按发布者要求完成协议，仅研究用途 | 同内容、不同情绪与不同音色的对照 | 表演性类别标签，不能监督人格价值判断或长期情绪更新 |
| [M3ED / RUCM3ED](https://github.com/AIM3-RUC/RUCM3ED) | 官方有 `annotation.json`、关系标注、splitInfo，以及 README 中音频片段和特征的网盘入口 | 中文情绪语境、关系与说话表现的辅助监督 | 视觉发布的是特征，不是完整原始视频；影视表演有域差异，不能当目标人物内心状态真值 |
| [BGE-M3 配套 MLDR](https://huggingface.co/datasets/Shitao/MLDR) | 可获取检索数据 | 查询投影、候选排序初始训练 | 文档检索不等于个人事件记忆，仍需自建时间/关系/重复分享标签 |
| [COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) | 发布者数据集页面可下载中文指令数据 | 联合训练中的通用语言保持 | 按子来源筛选版权/质量；不是人物偏好真值 |
| [LoCoMo](https://github.com/snap-research/locomo)、[LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 官方提供长期记忆评测入口 | 预留评测，检查时间、更新、跨会话记忆 | 不用测试问题/答案训练本项目；另建人格、主动分享与音视频记忆集 |
| 本项目 PersonPack 数据 | **必须自建**，真人材料/角色正史依据＋人工结构化 | 初始经历、观点形成、稳定倾向、知识边界、语气 | 二手推断标注为推断；没有依据不补造“亲身经历” |
| 本项目时间连续交互数据 | **必须自建/重标**，真实或受控录制和可控故障回放 | 打断、等待、沉默、分享分寸、已播后纠正、说话人对象 | 公共 ASR/TTS/问答集没有这些联合标签 |
| 本项目云端模拟数据 | **必须自建**，人工确认答案/证据、API 状态时间表 | claim 评估、请求、等待、过期与纠偏 | 正确返回、错误返回、证据不足、超时都要出现；不用在线云端模型制造训练标签 |

真实数据与模拟数据分开标记。模拟混音可使用公开语音、环境声音和测得/生成的房间脉冲响应，但其空间真值仅对仿真条件成立；真实麦克风阵列测试必须保留。

## 4. 统一格式与时间因果契约

数据按 session 组织，JSONL/Parquet 存元数据，音视频和大张量存 sidecar 文件。文件路径为相对数据根目录，`asset_sha256` 和来源 manifest 单独保存。所有示例均为虚构字段示意，不是已采集数据。

### 4.1 会话 manifest

```json
{
  "schema_version": "0.1",
  "dataset_id": "brain_interaction_pilot",
  "session_id": "s001",
  "persona_id": "role_A",
  "persona_version": "1",
  "source_type": "self_recorded",
  "split": "train",
  "clock_unit": "ms",
  "audio": {
    "path": "s001/mic8.wav",
    "sample_rate": 48000,
    "channels": 8,
    "channel_order": [0, 1, 2, 3, 4, 5, 6, 7],
    "geometry_ref": "s001/array.json"
  },
  "video": [
    {"camera_id": "cam0", "path": "s001/cam0.mp4", "offset_ms": 0},
    {"camera_id": "cam1", "path": "s001/cam1.mp4", "offset_ms": 14}
  ],
  "reference_voice": "personpacks/role_A/voice/ref01.wav",
  "initial_memory_version": "m0_v1",
  "annotation_version": "ann_3",
  "license_record_id": "rights_001"
}
```

前处理保留 48 kHz 阵列原始文件，分别生成 AEC/波束后的 16 kHz 主干输入；不能把重采样后的单通道文件当阵列训练原始数据。音频应标注重采样延迟、AEC 对齐、声道位置和播放参考。

### 4.2 时间片训练记录

```json
{
  "session_id": "s001",
  "persona_id": "role_A",
  "persona_version": "1",
  "state_version": 31,
  "request_id": "req17",
  "decision_ms": 6200,
  "semantic_block": [5000, 6000],
  "observations": [
    {"id": "a6", "kind": "audio", "t_start_ms": 5000, "t_end_ms": 6000, "arrival_ms": 6080, "validity": "valid", "asset": "features/a6.safetensors"},
    {"id": "v6", "kind": "video", "t_start_ms": 5900, "t_end_ms": 5900, "arrival_ms": 6010, "validity": "valid", "asset": "frames/cam0_5900.jpg"}
  ],
  "observed_text": "等等，我刚才说错了。",
  "text_available_ms": 6080,
  "played": {"utterance_id": "u3", "played_until_ms": 6120, "text_char_end": 9},
  "cloud": {"status": "inflight", "elapsed_ms": 1400, "deadline_ms": 11000},
  "labels": {
    "fast_action": "pause",
    "intent": "listen",
    "cloud_action": "cancel",
    "target_stop_ms": 6400,
    "actor": "human_annotation",
    "label_confidence": 0.9
  }
}
```

`observed_text` 只有在当时在线可获得或来自明确文本输入时才能进入特征；人工完整转写通常只是训练目标/标注依据，不能在 200 ms 快速头训练时作为已知输入。`label_confidence` 是标注一致性权重，不是模型的事实置信度。

### 4.3 claim 与云端模拟记录

```json
{
  "session_id": "s001",
  "persona_id": "role_A",
  "persona_version": "1",
  "state_version": 24,
  "request_id": "req17",
  "created_ms": 4800,
  "deadline_ms": 11000,
  "claim": {
    "claim_id": "c8",
    "text": "该活动在周六举行",
    "kind": "fact",
    "evidence_ids": [],
    "support_label": "insufficient",
    "needs_confirmation": true
  },
  "simulated_return": {
    "arrival_ms": 8100,
    "text": "公告写的是周日。",
    "evidence_ids": ["notice_v2"],
    "evidence_time": "2026-10-10",
    "quality_label": "supported"
  },
  "dependency_changed_ms": 6200,
  "accept_return_label": false,
  "annotation_basis": ["fixture_notice", "task_changed"]
}
```

同一 claim 制作两组正交变化：保持证据不变而改变 API 可用性；保持 API 可用而改变证据充分性。防止模型学到“云端离线就可以肯定回答”。延迟采样采用固定 seed 的 0.3/1/3/10/30 秒及超时、部分返回、重复返回，不将这些数值描述为任何供应商真实 SLA。

状态标签只含本地能观察的信息。没有服务端回执时，`QUEUED/RUNNING` 只能指本地排队/请求在途；训练样本不得给模型虚构的服务端进度或精确剩余时间。

### 4.4 事件记忆与压缩记录

```json
{
  "session_id": "s001",
  "persona_id": "role_A",
  "persona_version": "1",
  "state_version": 40,
  "request_id": null,
  "event_id": "ev11",
  "event_time_ms": 8400,
  "available_ms": 8650,
  "validity": "valid",
  "source_ids": ["a8", "u4"],
  "event_text": "交谈者纠正了活动日期，双方约定下次确认后再安排。",
  "participants": ["role_A", "user_01"],
  "interpretation": "对方在意安排准确，当前约定尚未完成。",
  "interpretation_status": "annotator_inference",
  "importance": 0.8,
  "identity_relevance": 0.2,
  "unresolved": true,
  "labels": {
    "write_action": "keep_evidence",
    "required_slots": ["corrected_date", "pending_confirmation"],
    "compression_budget_tokens": 80,
    "supersedes": ["ev10"]
  },
  "already_shared_with": []
}
```

`importance` 是标注信号而不是确定的心理事实；至少两名标注者抽样复核。后续 memory QA/行为 probe 单独存储，不能并入当时状态输入。

### 4.5 人格情绪与语音条件记录

```json
{
  "session_id": "s002",
  "persona_id": "role_A",
  "persona_version": "1",
  "state_version": 12,
  "request_id": null,
  "decision_ms": 7200,
  "source_event_ids": ["m0_17"],
  "situation": "朋友提出一个收益高但会违背已有承诺的机会。",
  "preferred_response": "先把已经答应的事情完成，再讨论新机会。",
  "rejected_response": "收益更高就直接放弃之前的约定。",
  "preference_basis": "source_supported",
  "emotion_readout": {"valence": -0.1, "arousal": 0.3, "dominance": 0.4},
  "emotion_observed_mask": [true, true, false],
  "event_present": true,
  "audio_target": "s002/u5.wav",
  "audio_codec_ref": "s002/u5_s3.npy",
  "word_alignment_ref": "s002/u5_words.json",
  "reference_voice": "personpacks/role_A/voice/ref02.wav"
}
```

32 维内部情绪无完整人工真值；上例三维辅助读出只在有相应标签时监督。没有依据的支配感、价值维度或人物内心状态设 mask=false，不将自动猜测当强标签。训练例只需行为、证据和简短标注依据，无须强求人物完整内心独白。

## 5. 每个待训练部分的数据与损失映射

统一记号：

- 有效目标集合 $\Omega_k$ 排除 padding、输入 token、参考音频/人物前缀和未知标签。
- $\operatorname{CE}_k=-\frac{1}{\max(1,|\Omega_k|)}\sum_{i\in\Omega_k}\log p_\theta(y_i|x_{\le t_i})$。文本、音频、动作分别归一化，不把高频音频 token 数当更高任务权重。
- 有标注置信权重 $w_i$ 时分母改为有效权重和；没有该任务标签的 batch 不计算该任务损失，也不把缺标签视为负类。

### 5.1 C1：感知/空间/讲话人连接

数据：AISHELL-1/4、AMI、ESC-50，外加标定阵列自录。每例含过去窗口音频、说话人/重叠活动、空间线索和明确的有效性标签；对“方位”只有具备真实标定或仿真真值时监督。

目标：

\[
L_{C1}=L_{\text{text}}+0.2L_{\text{speaker-pair}}+0.2L_{\text{sound}}+0.1L_{\text{DOA}}
\]

- `text`：teacher-forced 转写或短语义任务 CE，回传新投影，原生 encoder 冻结。
- `speaker-pair`：同一说话人窗口为正、其他人/混音为负的监督对比损失；验证时按未见说话人计算，不让说话人姓名替代声学身份。
- `sound`：事件多标签 BCE，以类别和有效窗口归一化；对未标注类别用 unknown mask，避免弱标签造成大量假负样本。
- `DOA`：圆周角误差 $1-\cos(\hat\phi-\phi)$；无有效方位时 mask。重叠音源首期用多候选概率而非强行唯一方向。

speaker/DOA 等辅助分类器是连接可用性的读出，不等于全项目必须永久保留这些显式分类输出。未通过跨设备验证前，空间特征质量低时降低门控权重。

### 5.2 C2：多视角融合与可用性连接

数据：AMI、Ego-Exo4D 可用子集，以及两相机同场景自录。字段：camera_id、同步误差、帧、标定可用性、跨视角共同实体/事件、某视角遮挡标签、没有新事件时的静默标签。

\[
L_{C2}=L_{\text{event/answer}}+0.2L_{\text{view-match}}+0.1L_{\text{availability}}
\]

`view-match` 为已标注同一事件/实体的跨视角 InfoNCE；不能把同时拍摄但观察完全不同区域的图像自动当全部正对。`availability` 是有效/遮挡/失联读出 CE。视觉重建可作小规模消融，但不把逐像素重建当对话相关性的主要目标。

随机删视角、时间错位和陈旧帧用真实 metadata 标注；缺失视角不要求模型重建不可见事实，而要求对剩余证据保持正确。

### 5.3 S1＋C3：持续状态及状态到主干连接

数据：按时间排序的 16–32 秒片段，后接 1–4 个只读取当前状态的 probe。标签包括当前话题、活动讲话人、指代、任务是否未完成、已播句子边界、claim 状态；长时通过相邻片段保持前向状态，反传在片段边界截断。

\[
L_{S}=L_{\text{next-output}}+0.3L_{\text{state-probe}}+0.2L_{\text{event-boundary}}
\]

- `next-output`：下一块的说/听控制与输出内容监督。
- `state-probe`：已标注实体/话题/任务状态 CE、数字时间 Huber；可比较在相同记忆预算下是否比最近文本更有效。
- `event-boundary`：事件结束/新事件开始 BCE，用人工切分和可验证变化点标注。

未来事件可作预测目标，但不得作为当前输入。不同情节、人物和版本的状态不能连续携带。训练片段前的 warm-up 段只更新状态、不计 loss，用于减少从全零开始造成的分布差异。

初期不要同时优化“槽位必须绝对不变”和“事件变化应立即更新”；状态平滑仅作用于无事件的稳定片段，并将其作为可选项验证。

### 5.4 S3：交互策略、云端动作与时机

标签由两人独立观看/聆听轨迹标注，再仲裁，保留可接受动作集合。正常“嗯”既可能是附和也可能是犹豫；模型输入必须包含前后可获得情境，不能只按关键词贴标签。

\[
L_{\text{policy}}=L_{\text{fast}}+L_{\text{intent}}+L_{\text{cloud-action}}+0.2L_{\text{timing}}
\]

前三项为分头 CE；若多个动作合理，使用标注者分布作为 soft label。时机损失采用 Huber：$\operatorname{Huber}((\hat t-t^*)/1000)$，只在实际需要发声/停止的记录计算。$t^*$ 不应早于足够语义证据可获得的时刻，避免用未来完整转写训练不可能的即时反应。Huber 的梯度对大误差为常数，损失本身并非有界。

每种轨迹至少构造反事实：同一句插话→附和/纠正/旁人；相同场景→忙碌/愿意交流；同一记忆→讲过/没讲过；相同云端等待→有值得澄清的问题/无需重复说话。保留自然的类别分布作为验证集；训练可重采样少见打断，但不能只训练需要开口的样本。

初期先监督学习，不启用长时 RL。后期若引入闭环模拟，可用动作级策略梯度优化任务完成、证据支持、适当沉默及打断代价；需要探索轨迹和记录行为概率，不能直接对人类静态记录套用未经校正的 on-policy PPO。

### 5.5 claim 评估、云端返回接受与纠错

为每个候选结论人工标注 `supported/refuted/insufficient`，并区分事实/偏好/回忆/建议。需要等待的标签来自当前可得证据，不来自 API 状态。训练数据必须有“答案碰巧正确但依据不足”和“云端自信却错误”的样本。

\[
L_{\text{claim}}=L_{\text{support}}+0.5L_{\text{Brier}}+L_{\text{accept}}+L_{\text{correction}}
\]

- `support`：三分类 CE，按 claim 归一化。
- `Brier`：$\frac1N\sum_i(\hat p_i-y_i)^2$，其中 $y_i$ 是“当前证据不足以肯定表达”的人工标签；此概率与客观真值概率分开。
- `accept`：接受/拒绝/需重新核验 CE。metadata 规则如错 persona/version 直接拒绝，同时记录给模型，不能靠学习去容忍身份串扰。
- `correction`：已播放错误内容后的正确修订文本 CE，尚未播放的片段则监督撤回，不混为相同动作。

保留独立校准集，报告 reliability diagram、Brier/ECE 及风险—回答覆盖率曲线。阈值根据验证集的目标风险选择，不能将 0.8/0.9 直接叫“80%/90% 正确”。API 不参与任何损失反传。

### 5.6 S4：记忆读、写和压缩

训练三类记录：query 与相关/无关事件；事件与写入/保留预算；原始事件与压缩版本及延迟 probe。每个记忆记录有 `persona_id/version、occurred_at、available_at、source_ids、evidence_state、already_shared_with、supersedes`。

\[
L_{\text{read}}=-\log\frac{\exp(s(q,m^+)/\tau)}{\sum_{m\in C}\exp(s(q,m)/\tau)}
\]

候选集 $C$ 含时间相近但人物错误、人物相同但已过期、语义相似但已讲过的困难负例。查询投影和 scorer 可反传；ANN top-k 是离散候选生成，不会把最终回答 loss 自动传回整个索引。

\[
L_{\text{memory}}=L_{\text{read}}+L_{\text{write}}+L_{\text{summary}}+0.3L_{\text{required-slots}}+\lambda_bL_{\text{budget}}
\]

- `write`：`discard/keep_evidence/summarize/merge/link_correction` CE；未解事项、经历意义、事实修正应有足够样本。
- `summary`：人工/经人工确认摘要的 teacher-forced CE，训练状态到主干连接或压缩 adapter。
- `required-slots`：压缩后能否正确回答保留字段的监督；使用 teacher-forced 软路径训练时可反传，硬生成摘要的评测则仅作外部指标。
- `budget`：若选择器给出保留概率 $p_i$，用 $\max(0,\sum_i p_i c_i-B)$ 约束期望字节/token 成本；最终落盘硬选择使用预算算法，不声称端到端可微。

保持相同存储预算比较按时间遗忘、RAG 检索、状态＋学习压缩，测延迟任务和主动使用记忆的效果；摘要 ROUGE 只作辅助。错误摘要不得覆盖原证据，压缩后版本保留来源和修订链。

### 5.7 S2 与人物适配

数据：有来源的个人经历、不同情境下的判断选择、共同行为历史、情绪序列和偏好对，细节以人物专项为准。公共 ESD 只帮助声音情绪对照，不能充当目标人物价值观标签。

\[
L_{\text{person}}=L_{\text{response}}+0.3L_{\text{preference}}+0.3L_{\text{memory-support}}+0.2L_{\text{emotion}}
\]

`response` 是人物情境回应 CE。`preference` 起步用成对评分 logistic loss；在 T3 语义模型 adapter 已稳定后可以用 DPO：

\[
L_{\mathrm{DPO}}=-\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y^+|x,p)}{\pi_{ref}(y^+|x,p)}-\log\frac{\pi_\theta(y^-|x,p)}{\pi_{ref}(y^-|x,p)}\right]\right)
\]

初始 $\beta=0.1$，参考策略冻结；按回答计平均，屏蔽输入/人物前缀。偏好依据必须是人物材料和情境，不是无条件奖励讨好用户。[DPO 原论文](https://arxiv.org/abs/2305.18290)

`memory-support` 对所引用经历是否有正确人物/版本/来源作监督。`emotion` 使用有标签维度的 Huber/分类 CE；32 维隐状态通过后续行为和语音任务学习，其三维读出带 mask。情绪转移额外比较同事件不同人格、同人格不同关系、事件后恢复轨迹。只有无新事件片段施加小权重平滑/恢复项，紧急事件不强行平滑。

多人物共享训练按人物均衡采样；人物专用参数独立存储。增加交叉干预：只换音色时价值选择应保留，只换人格时经历与价值变化但不自动换声音，换人物后不得检索前一人物后天记忆。

### 5.8 C4、Talker 条件和承诺边界

输入数据包括目标文本、词/字时间对齐、S3 token、参考音色、情绪/风格标签、当前已允许的 claim 集合和播放边界。训练监督只对目标说话区间计 loss，参考音频不当成需要复述的目标。

MiniCPM-o 本档使用 S3 单路音频词表，不能误套 MiniMind-O 的八码本损失：

\[
L_{\text{speech}}=-\frac1{|\Omega_a|}\sum_{j\in\Omega_a}\log p(c_j|c_{<j},h^{commit},v,e)
\]

桥先冻结已有主通道，只训练人物/情绪残差；以正确文本和对应隐状态 teacher force，之后 T4 可解冻 Talker。语音解码器先冻结。声音身份与情绪听感用独立评测器和人工盲听评测；若只有离散采样后的音频，不把说话人相似度或 MOS 直接加到可微损失。

承诺门控训练：$L_{\text{commit}}=\operatorname{BCE}(\hat b,b)+L_{\text{allowed-text}}$。$b$ 来自支持状态和允许表达方式：未知事实不能肯定表达，但可以说“我需要确认日期”；这段说明本身是可表达内容。训练用目标 mask 控制送入 Talker 的片段；推理硬 gate 不可微。另设外部指标统计未允许 claim 是否进入实际播放内容，并测试把隐藏候选信息通过风格残差泄露出来的情况。

若后续解冻流式声码器，另设 wave/mel/flow-matching 目标并检查其真实计算图；首期不为追求“所有模块训练”引入未验证的波形梯度路径。

## 6. 联合训练的初始权重与批次组织

建议先单任务 warm-up，再按任务批次混合；每个 batch 只算有标签的损失。T2/T4 起始宏权重可设为：文本 1.0、音频 0.5、状态 0.3、策略 1.0、claim 1.0、记忆 0.5、人格情绪 0.5、通用保持 0.2。第 5 节子损失先各自归一化再进入宏权重，避免重复乘权重而不自知。

训练日志分别记录损失、梯度范数和主要任务指标；一项损失下降不能掩盖另一项退化。若需使用梯度归一化/冲突处理，先与固定权重对照，不一开始同时引入多个动态权重方法。

初始 batch 混合比例：30% 时间连续交互、20% 人物与记忆、15% 结论/云端模拟、15% 语音、10% 视觉/环境声、10% 通用语言保持。联合解冻后根据回退调整保持样本比例。语言和音频任务不必每次同时出现；缺某模态时使用对应缺失标记与 mask，模型不需要虚构样本补齐全部模态。

T2 截断反传长度先 16 秒，稳定后 32 秒。200 ms 状态更新最多 80/160 步，语义主干仍每约 1 秒计算一块。人物稳定向量在片段内共享；动态状态在 episode 边界重置，跨 session 只由显式记忆恢复而不偷偷携带测试答案。

## 7. 必须自建的最小实验集与标注流程

建议第一轮先完成一个小而覆盖完整的试验集：

| 自建部分 | 起始预算 | 关键覆盖 |
|---|---:|---|
| 人物对照 | 3–5 个可充分取证的人物包，含至少一个对照人物 | 真实经历、未见情境判断、音色/人格可交换；动漫和真人素材分版本 |
| 交互录制 | 20–50 小时，2–4 人、多房间、不同阵列摆位 | 附和、纠正、旁人讲话、重要声音、沉默、分享过多、对方忙碌 |
| 精标决策片段 | 3000–10000 个 5–30 秒片段 | 每片段带时间可得信息、可接受动作、停止/继续目标与已播边界 |
| 云端模拟 | 3000–10000 条基础情境，每条若干状态/延迟变体 | 正确/错误/无依据返回、过期、取消、断网、重复返回；语义事实与可用状态正交 |
| 长期记忆 | 500–2000 条多会话 episode | 事件跨日、事实更新、分享去重、重要旧事、未完承诺、不同关系 |
| 表达条件 | 每目标人物可用的干净语音与情境对应素材 | 先参考音色零样本评测；不足再采集/整理训练集，音色资料不足不能承诺高相似 |

先双人标注 200 条校准概念；统计 Cohen kappa 或可接受动作的一致率。分歧样本保留分布或交第三人仲裁，不将主观判断伪装成唯一事实。紧急程度、分享价值和人物倾向分别标注。人物情绪不能从声音单一推断，材料只支持声音表现时标成“表现标签”。

数据增强仅改允许变化：加噪、混响、音量、设备丢帧、网络延迟可以保留语义；移位、换人、换事件需要同步修改标签。通过模板产生受控场景是可行的，但报告模板覆盖率，并将真实未见对话作为主要验收集。

## 8. 划分、评测与停止条件

1. **感知数据：**按会议/房间/说话人划分；同一原始视频的邻近窗口只能出现在一个 split。不得把同一声音混出多版本后分到训练和测试。
2. **人物数据：**同时做已见人物未见事件、已见事件未见提问、未见人物包三种评测。素材中的同一事件及其改写/偏好对绑定划分；另保留角色剧情时间点外的信息，测试是否越界。
3. **长期记忆：**按 episode 和时间划分；测试 episode 的未来记忆不预建入初始库。用户/人物命名空间严格隔离。
4. **云端模拟：**按事实/证据源和场景模板划分，测试有未见延迟组合和失败模式；可用状态不作为证据真值的捷径。
5. **校准：**validation 选择阈值，单独 test 报结果；三个随机 seed 和置信区间用于关键对照，不只挑最好一次。

必须并列报告的指标：

| 目标 | 指标 | 防止的错误结论 |
|---|---|---|
| 语义全双工 | 误打断/漏打断、附和正确继续率、证据可得→决策→实际静音各阶段时延 | 不能仅报 VAD 检测延迟证明语义交互 |
| 交流价值 | 每分钟无关发言数、重复分享率、适当沉默、用户盲评、应回应却沉默比例 | 不能靠一直沉默降低错误率 |
| 云端协作 | 无支持肯定输出率、Brier/ECE、等待时长、无用填充、过期返回接受率 | 不能靠所有问题等云端获得高准确率 |
| 人物 | 素材忠实、偏好排序、未见情境一致性、经历串人物率、声音与人格分离盲评 | 不能靠口头禅或音色误认为人物复刻成功 |
| 记忆 | 固定预算下延迟任务正确率、关键字段保留、错误传播、主动相关使用率 | 不能靠给出更长上下文宣称压缩机制更好 |
| 语音 | CER、说话人相似度、情绪听感、自然度盲评、流式 RTF、实际首声时间 | 单一自动分数不足以代表听感 |
| 退化 | 缺视角/断麦/检索失败/云端失效下剩余任务能力和恢复时间 | 不要求核心/GPU故障靠模型自行恢复 |
| 资源 | 峰值显存、训练步耗时、连续30分钟队列长度、p50/p95延迟 | 能加载权重不等于能实时运行 |

关键消融保持主干、数据和预算相同：无持续状态／有状态；原始隐状态连接／承诺连接；人物文件／人物 adapter／二者组合；时间压缩／重要性压缩；固定等待话术／状态条件策略；单视角／多视角；冻结／分组解冻。

T1 若只能在训练集对齐且跨设备失败，应暂停扩规模并修复输入表征；T2 若正确率提高但交流频率令人厌烦，应重平衡沉默/分享数据；T3 若人物切换造成经历串扰，先修复命名空间和参数分离；T4 若通用能力显著回退，回到较小解冻集合。以上是实验推进条件，不是项目失败判定。
