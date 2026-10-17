# 单卡 RTX PRO 6000：初始模型架构设计

版本：v0.1；资料核验日期：2026-10-13；整理与复核日期：2026-10-14。本文是可修改的探索性总体设计，不代表已实现、已训练或已经测得性能。`已核验`指公开配置、实现或模型卡中可确认的事实；`设计起点`指本项目拟采用的超参数；`研究项`指需要数据和实验验证的新增机制。

配套阅读：[本档训练与数据设计](04-rtx-pro6000-training.md)、[人物、记忆与情绪专项](07-character-memory-emotion.md)。

## 1. 硬件边界与本档研究目标

本档硬件明确为 **NVIDIA RTX PRO 6000 Blackwell Workstation Edition，96 GB GDDR7 ECC、1792 GB/s、600 W、PCIe 5.0 x16**。它不是 RTX 6000 Ada 48 GB，也不是同名 Server Edition 或 Max-Q 版本。规格来自 [NVIDIA 官方数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)。主机设计起点为 192–256 GB RAM、至少 4 TB NVMe，训练数据另存；这些是本项目容量建议，不是模型运行硬性规格。

本档利用 96 GB 的主要方式是：保留 BF16 的现成全双工多模态模型，增加可训练的持续状态、人格、记忆和决策结构，扩大训练时间窗口。显存增大不等于交流延迟按比例降低。先做单用户、单人物实例，规划两路摄像头和 4–8 通道麦克风；四视角作为扩展实验，不一开始承诺多用户服务。

相对 32 GB 档的实际收益：

| 项目 | 本档可安排的实验空间 | 仍然存在的限制 |
|---|---|---|
| 权重精度 | 约 9B omni 的 BF16 常驻，避免首轮研究同时混入低比特量化误差 | BF16 是否达到实时仍须端到端测量 |
| 新结构训练 | 新增约 0.1–0.25B 参数的状态、连接、策略模块；主干 LoRA | 冻结主干仍可能需要激活内存，不能只按权重大小预算 |
| 时间上下文 | 8k 起步，可比较 16k token；训练 16–32 秒片段再截断反传 | 长时个人历史仍存外部记忆，不无限延长 KV |
| 多模态并行 | 内置视觉/音频、环境声、说话人和检索模块可共存 | 多摄像头主要增加计算与输入 token，不只是显存 |
| 后续解冻 | 支持 Talker、投影层、状态核心等分组解冻 | 9B 全参数标准混合精度 Adam 的权重、梯度与优化器状态合计约 144–162 GB，尚未含激活；不能承诺全量 96 GB 单卡训练 |

## 2. 主选路线与可替代路线

### 2.1 主选：复用 MiniCPM-o 4.5 的已训练流式基础

主 checkpoint：[`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)。首期固定模型 revision、配置文件、tokenizer、processor、依赖版本和权重 SHA256；当前设计先指定模型名，不把变化中的 `main` 当可复现实验版本。

选择理由是已有时间分块的视听输入、语言主干及语音生成连接，能把研究集中在人格连续性、记忆选择、证据等待和交流分寸。官方存在 `as_duplex()`、`prepare()`、`streaming_prefill()`、`streaming_generate()` 路径；原生实例按约 1 秒块更新。不能把普通 `chat(stream=True)` 当作该路径，也不能仅修改块长就声称获得已训练的 200 ms 语义能力。[模型卡与接口示例](https://huggingface.co/openbmb/MiniCPM-o-4_5)、[流式实现](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py)

公开工作没有直接提供本项目所需的人物经历巩固、阵列空间融合、结论级云端等待及长时人格评价机制，以下均需新建或改造。现成 checkpoint 只能作为起始能力和对照组。

### 2.2 备选与触发条件

| 备选 | 何时使用 | 必须接受的代价 |
|---|---|---|
| 同模型官方 `openbmb/MiniCPM-o-4_5-awq` 或 `openbmb/MiniCPM-o-4_5-gguf` | 仅当推理常驻内存、主机限制或吞吐测量确实要求量化 | 用作推理对照；需核验对应引擎支持全部流式能力。不能默认量化引擎能训练新增层或暴露同样隐状态 |
| `Qwen/Qwen3-8B`＋专用编码器＋`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 原生 omni 中间接口难改，或需要更清楚地做连接层实验 | 从头训练跨模态和隐状态到语音连接；独立模型的公开接口组合不能自动形成端到端梯度或全双工 |
| MiniMind-O 小规模结构 | 快速消融、验证状态/门控机制，或需在单卡做更多模块全量更新 | 能力上限和中文/复杂语义能力需独立评测；其 VAD 打断不是本项目语义策略的替代 |

备选来源：[Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[CosyVoice 官方仓库](https://github.com/QwenAudio/CosyVoice)、[MiniMind-O 仓库](https://github.com/jingyaogong/minimind-o)。这些备选不与主选模型同时常驻，本档预算不包含第二套 8B 主干。

## 3. 模块清单及冻结边界

| ID | 功能与选型 | 输入→输出 | 初期状态与限制 |
|---|---|---|---|
| P0 | 音视频时钟、AEC、波束形成、缓冲 | 带采样时钟的多通道 PCM/图像→单声道语义流＋保留的空间特征 | CPU 工程模块；不是可训练模型。AEC 使用实际播放参考，不使用尚未播放的生成音频 |
| P1 | MiniCPM-o 内置 SigLIP 视觉编码器和 resampler | 单视角图像→视觉特征→每图 64 个主干维度 token | 复用已训练权重并冻结；多视角身份/融合需新增 |
| P2 | MiniCPM-o 内置流式改造的 Whisper 编码器和投影 | 16 kHz 音频块→压缩音频 token | 冻结现成流式实现；不能用普通离线 Whisper 原样替代并声称仍因果 |
| P3 | `iic/speech_campplus_sv_zh-cn_16k-common`，revision `v1.0.0` 起步 | 16 kHz 语音片段→192 维说话人嵌入 | 冻结；身份嵌入不是现成实时分离器，重叠发言可输出 unknown/multi |
| P4 | BEATs Iter3+ AS2M，官方 fine-tuned cpt2 | 过去的声音窗口→声学特征与事件分数 | 冻结；原型以过去 2–4 秒窗口、1 秒更新，不宣称该模型本身是流式因果网络 |
| P5 | `BAAI/bge-m3` | 事件文本/查询→1024 维检索向量 | 冻结、CPU 或空闲时 GPU；不是人格或可验证记忆事实库 |
| C1 | 来源、空间、时间连接层 | 各感知特征→1024 维状态 token | 新建训练 |
| C2 | 多视角选择和融合 resampler | 2–4 路图像 token→保留来源的融合表示 | 新建训练；原始视觉 token 通道仍可作为对照 |
| S1 | 持续状态核心 (H_t) | 当前证据、上次状态、已播内容→新状态 | 新建训练；首版 64 槽×1024，6 层更新网络 |
| S2 | 人格/事件评价/情绪更新器 | PersonPack、事件、关系、历史→条件状态 | 新建训练，详见第 6 节及人物专项 |
| S3 | 交互策略与 claim 评估头 | 状态、候选结论、云端状态→行为/确认需求 | 新建训练；VAD 仅作输入信号 |
| S4 | 记忆写入、检索查询和压缩选择头 | 状态、候选事件→存取决策 | 新建训练，外部存储本身不需要神经训练 |
| B1 | MiniCPM-o 内置语义主干 | 音视频 token、状态、记忆→语言隐状态 | 冻结；不是额外再加载原版 Qwen3-8B |
| C3 | state-to-backbone/PersonPack 接口 | 1024 维状态及人物条件→4096 维条件 token/残差 | 新建训练；不回写已经缓存的历史 token |
| C4 | commitment bridge | 准备表达的片段及其隐状态＋人格情绪→Talker 条件 | 保留已有 4096→768 通道并加零初始化残差、承诺选择训练 |
| T1 | 内置 MiniCPMTTS＋S3/流式 Code2Wav | 已承诺的语义、参考音频、风格→24 kHz 声音 | 先冻结；后续只解冻条件适配/Talker。声音身份和内部人格分别控制 |
| X1 | 云端推理 API | 最少必要的证据与问题→结论、依据、不确定项 | 仅推理、异步；不进训练图、不生成首期训练集 |

P3 来源：[官方模型卡](https://www.modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common)、[3D-Speaker 官方加载配置](https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/bin/infer_sv_batch.py)。P4 来源：[BEATs 官方权重入口](https://github.com/microsoft/unilm/blob/master/beats/beats_README.md)、[模型配置与特征接口](https://github.com/microsoft/unilm/blob/master/beats/BEATs.py)。P5 来源：[BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)。公开权重存在不代表在本项目声学条件中无须验证。

## 4. 维度、连接与时间约定

### 4.1 已有维度与新增维度明确分开

下列已有维度来自 [MiniCPM-o 4.5 配置](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/config.json)：主干 `hidden_size=4096`、36 层、8 个 KV 头、`head_dim=128`；视觉宽度 1152、`query_num=64`；音频宽度 1024、时间池化 5；TTS 宽度 768、20 层、音频词表 6562。输入图像尺寸、slice 策略和流式行为以 processor 与运行时 duplex 配置为准，不能只读 config 中某个 `streaming=false` 就否定或启用流式功能。

**新增设计起点：**状态宽度 (d_s=1024)，64 个状态槽，8 注意力头；人格 (p=256)、动态情绪 (e=32)、关系 (r=64)、事件评价 (a=16)。其中情绪 3 维可解释读出只用于辅助监督，不能把整个 32 维状态强行解释为已知心理量表。所有维度可在消融后调整。

### 4.2 输入观测统一格式

所有神经特征附带轻量元数据。示例是数据契约，不是实现代码：

```json
{
  "schema_version": "0.1",
  "persona_id": "role_A",
  "persona_version": "1",
  "session_id": "s001",
  "state_version": 37,
  "request_id": null,
  "observation_id": "mic0-4200",
  "kind": "speaker_audio",
  "source_id": "beam_0",
  "t_start_ms": 3200,
  "t_end_ms": 4200,
  "arrival_ms": 4260,
  "validity": "valid",
  "quality": 0.83,
  "speaker_id": "unknown",
  "self_generated": false,
  "feature_ref": "features/s001/mic0-4200.safetensors",
  "feature_shape": [10, 4096]
}
```

`validity` 取 `valid/missing/stale/failed/uncertain`；没有声音时仍是有效观测，设备失联则是 missing/failed。`quality` 为经校准或明确来源的质量估计，不伪装成语义正确率。每个输入只能使用 `arrival_ms<=决策时刻` 的观测；采集时间与可获得时间必须同时保留。

### 4.3 连接层具体方案

1. **感知→持续状态 C1：**每模态采用 `LayerNorm→Linear(d_in,1024)→GELU→Linear(1024,1024)`，再加入模态、来源、时间差、validity 嵌入。无效模态使用 missing token 和注意力 mask，不用全零假装观测为零。
2. **多视角 C2：**每路沿用冻结视觉 encoder/resampler；首版两路各 64 token，分别加摄像头 ID、标定是否可用。新增 2 层、16 查询的跨注意力模块输出 16×1024 状态表示。主干开始仍接收原始两路 token；待融合验证后比较压缩成总计 64 主干 token。camera ID 不能代替几何标定；有外参时另编码相机位姿和置信度，缺失时不编造 3D 坐标。
3. **阵列与讲话人 C1：**波束形成语音进入原生音频路径；保留 GCC-PHAT/IPD、方位估计、SNR、回声残留和说话人向量，经各自 MLP 合成最多 4 个音源 token。方位估计和身份置信度分开。CAM++ 对过去的已到达语音提取嵌入，首版累积约 1–2 秒干净语音才更新身份，不承诺 200 ms 新说话人识别。
4. **持续状态 S1：**每次以 64 个前状态槽作为 query，对新观测和检索记忆做 cross-attention，再经状态槽 self-attention/FFN；门控插值为 $H_t=(1-g_t)\odot H_{t-1}+g_t\odot\tilde H_t$。槽位分组作软先验：场景 16、话题/目标 16、关系/人物 8、待确认结论 8、当前表达 8、记忆读写 8；先用辅助标签稳定用途，不保证槽位天然可解释。
5. **状态→语义主干 C3：**首版从 (H_t) 重采样 16 个状态 token，MLP `1024→2048→4096`，只在下一个 1 秒块的感知区追加。给新增路径学习门控并初始化接近零，避免刚接入随机状态扰乱已训练模型。实际代码需要扩展 chunk 的 embedding 拼接与位置/mask，不是公共 API 直接可传任意隐状态。
6. **后续 C3 实验：**在主干每 6 层插入一个瓶颈为 1024 的 gated cross-attention 读取独立 (H_t)。每次解码读当前 state version；旧主干 KV 不变，新状态的 K/V 独立缓存并在版本变化时更新。比较这一方案与状态 token 追加的延迟、长期漂移和训练难度，首轮不同时启用两套路径。
7. **记忆→状态：**BGE-M3 候选检索 top-32，人物/关系/时间过滤后由可训练 scorer 选 top-4～8；事件文本总预算 1024～2048 token。向量经 `1024→1024` 接入状态，文本带证据 ID 接入主干。向量用来选择，具体经历依据仍是可追溯文字/多媒体片段。
8. **人格→状态/主干/Talker：**将 `[p,e,r,a]` 投影到 1024；稳定人物信息用 8 个条件 token＋人物 LoRA，动态情绪在状态和语音条件中更新。LoRA 起步 rank 16，只针对经过实验选出的 attention/MLP 模块；切换人物只换 PersonPack，不改共享感知权重。
9. **commitment bridge C4：**保留原有语义→Talker 投影，增加 `state/person→768` 的零初始化残差。源码中的条件由 `tts.emb_text(token_ids)` 与 `tts.projector_semantic(hidden_states)` 相加构造，投影使用原配 4096→768 MLP；归一化规则沿用锁定版本。新增残差加在该 768 维条件处，不能直接替换成不同模型的音频 embedding。输入必须来自当前获准表达的短片段；用文本片段、claim ID 和 evidence status 显式绑定。不要让整个候选推理隐状态未经选择直接成为语音条件。

语音组件成套锁定：原配文本 tokenizer 及特殊 token、S3 音频 tokenizer、MiniCPMTTS 权重、流式 Code2Wav/flow decoder。音频 tokenizer 输入为 16 kHz，模型示例的语音输出为 24 kHz；目标 S3 token 按原配 tokenizer 提取，不能借用 Mimi 的八码本或 CosyVoice 其他版本的 token。参考音色走官方 `prepare(ref_audio, prompt_wav_path)` 对应条件路径；人物情绪残差属于本项目新增训练接口，不能误称为现成的任意32维情绪控制 API。[原始条件构造代码](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py)

第 9 项不是数学上的防泄露证明：如果残差或主干状态隐含未确认信息，仍可能被说出。因此验证集必须测“已有正确猜测、尚未获得支持时是否提前肯定表达”，同时记录过度拒答和等待。

## 5. 全双工、快慢回路与 KV 生命周期

### 5.1 初始时钟

| 路径 | 设计周期 | 可以使用的信息与职责 |
|---|---:|---|
| 采集/AEC/设备故障 | 10–20 ms 音频帧 | PCM、实际播放参考、采集时钟；不做复杂语义判断 |
| 快速状态/交互头 | 200 ms | 已到达声学信号、当前播放位置、最新已完成语义状态；可请求暂停、保持倾听或继续 |
| 原生多模态语义块 | 约 1 s | 完成的音频块和抽样图像；更新语义、观点、说话意图 |
| 视频 | 每摄像头 1 FPS 起步 | 突发变化可提高抽样，但不能据此宣称主干语义更新已变成 10 Hz |
| 环境声 BEATs | 过去 2–4 s 窗口，每 1 s 更新 | 只含过去已到达样本；另测短窗事件漏检 |
| 记忆/云端 | 事件驱动 | 异步返回，不阻塞输入采集 |

1 秒时间块是沿用已有训练机制的起点；原论文对更短块长也报告了能力取舍，因此 200 ms 的快速头不是“已获得 200 ms 完整语义理解”。需要更快语义判断时，另行训练短块声学/语言策略或部分解冻，不把 MLP 投影视作因果能力来源。[MiniCPM-o 4.5 技术报告](https://arxiv.org/abs/2604.27393)

快速头输出 `continue/pause/listen/ack/end`；慢速意图头输出 `share/discuss/ask/answer/support/silence`，云端头输出 `none/request/wait/clarify/defer/cancel`。三个头分离，便于监督和防止动作类别混淆。听到“嗯，继续”的处理不能只依赖 VAD；声学信号先触发候选，语义可得后再确认动作。

### 5.2 缓存与已播放边界

- 每个会话有独立主干 KV、Talker cache、状态 (H_t) 和外部事件日志；人物版本变化使旧请求失效。
- 新观测、云端结果和记忆以新的时间块加入，不篡改旧 KV。跨注意力方案只替换独立状态 K/V，不重写历史自注意力缓存。
- 主干 KV 初始上限 8k token，达到高水位后按**完整时间块**截断/重建，保留人物前缀、压缩工作记忆、最近连续块。不得随意删 token 而不处理位置编码和对齐。流式实现中的窗口工具可作基线，但加入新 token 后仍需验证。
- 文本计划、已合成音频、进入播放器队列、实际播放完成四个位置分别记录。初始设计把未播放积压限制在约 0.5–1.0 秒的目标范围，具体由 codec 最小 chunk 及测量决定。
- 模型决定停止后，丢弃未播放数据，播放短淡出以减少爆音；将 `played_until_ms` 和对应文本跨度回灌状态。已说出口的错误触发后续纠正，不能视为取消成功后已消失。
- 语音调度使用短任务；后台记忆压缩/批量 embedding 避免占据长 GPU kernel。单 GPU 上并发 Python 线程不等于硬件具有可预测优先级。

## 6. PersonPack、情绪及后天记忆

PersonPack 至少包括 `persona_id/version`、有来源的初始经历 (M_0)、人格 embedding/adapter (P)、声音与风格条件 (V)、独立后天 memory namespace、动态状态初始化。真人和动漫角色都保留“人物知道的事实”和“整理者知道但人物尚不知道的事实”边界；动漫还绑定剧情版本与时间点。

本档中的人物条件起始值：$p\in\mathbb R^{256}$、$e_t\in\mathbb R^{32}$、关系 $r_t\in\mathbb R^{64}$、评价 $a_t\in\mathbb R^{16}$。人物适配参数属于人物包；共享主干后期联合训练使用多人物混合，避免一个人物渗入所有身份。稳定人格不按每帧更新，情绪和关系也不等于同一个向量。

情绪更新器接受事件及相关经历，学习“意义→反应→恢复”转移；对方的情绪估计另存，不能直接复制成模型情绪。情绪影响注意、分享欲望、语速和韵律；证据支持度单独维护。与素材一致的价值判断、相关经历是否值得分享、是否已讲过，比持续模仿口头禅更重要。

初始记忆分层：

| 层 | 设计容量/形式 | 写入与生命周期 |
|---|---|---|
| 原始缓冲 | CPU 环形缓冲，音频 30–60 秒，抽样视频按内存预算 | 普通感知滚动覆盖；重要片段保存为证据引用 |
| 工作状态 | $64\times1024$ 隐状态＋显式待办/claim 列表 | 200 ms 更新状态，语义以 1 秒块输入；场景切换可摘要 |
| 情景记忆 | JSON/Parquet 事件记录＋原始来源引用 | 事件边界、关系变化、承诺、重要纠正时写入 |
| 语义/关系记忆 | 可追溯事实、偏好、经历解释和版本边 | 多事件支持后凝练，冲突创建新版本 |
| 人物参数巩固 | 小型 adapter，离线低频 | 首期关闭自动在线权重更新；先评估持久存储与策略 |

压缩选择头输出 `discard/keep_evidence/summarize/merge/link_correction` 和预算；压缩文本由本地语义模型在空闲时生成，保存来源 ID、保留字段和丢弃字段。每次合并保留因果、人物意义和未完成事项。未通过验证的云端结果只能进候选区，不能变成“我曾经经历过”的自传记忆。

具体人物素材处理、损失和角色替换实验见 [人物专项](07-character-memory-emotion.md)。

## 7. 云端协作与结论控制

云端仅按需调用推理 API；不训练云端权重、不通过 API 生成首期训练集，也不让云端文本直接进入播放器。

每个候选 claim 记录 `claim_id/text/stance/evidence_ids/needs_confirmation/status`。`stance` 区分事实、假设、人物偏好、回忆和建议；人物偏好通常无需由云端“裁决”，其中依赖事实的前提可单独确认。

请求字段包括 `request_id/persona_id/persona_version/session_id/state_version/claim_ids/context_hash/created_ms/deadline_ms`。上下文按证据需要发送，不默认上传全部视听流或人物经历。云端可获得状态由真实传输层提供 `available/inflight/partial/completed/timeout/failed/cancelled`；可附近期延迟分布，不能让模型凭空声称“已经快好了”。

若使用 `QUEUED/RUNNING` 状态名，未取得供应商明确回执时只能解释为本地排队和本地请求在途，不能声称获知远端排队位置、推理进度或确定完成时间。延迟预测也是带不确定性的本地估计。

返回需保留 `supporting_evidence/limitations/source_time`，并重新核验：人物版本相同、目标 claim 未被新输入替代、证据没有过期、语义前提仍成立。仅 `state_version` 不同不一定意味着结果无效，要按依赖项判断；人物或目标任务已变则拒绝旧结果。

本地策略在等待时可沉默、说明正在确认、询问能减少不确定性的问题或讨论已获支持的部分。没有价值的新内容时不要重复填充。请求失败不会把 `needs_confirmation` 改成 false；超时后可以保留不确定表达或延后回答。

对用户打断要分别取消本地生成、清理未播放队列和发送 API cancel（若服务支持）。即使 API 无法真正停止远端计算，本地仍须忽略已经失效的返回。已经播放的结论错误则走明确纠正流程。

## 8. 内存规划与训练边界

以下均为**容量估算，不是设备实测**。GiB 与厂商 GB 不混为同一精度；表中保留宽裕区间，实施时以 `allocated/reserved/peak` 及系统可用显存为准。

| 常驻/运行项 | 规划显存 GiB | 说明 |
|---|---:|---|
| 主 omni BF16 权重、TTS 附件 | 18–23 | 9B×2 byte 约 16.8 GiB 是理论下界，给附属权重/实现留余量 |
| CAM++、BEATs、BGE 与前端 | 2–4 | BGE 可移 CPU；不能重复计入内置视觉/音频编码器 |
| 新状态、连接、人物参数 | 0.5–1.5 | 初期约 0.1–0.25B 新参数，实际按参数清单计数 |
| 主干 KV：8k / 16k | 1.13 / 2.25 | BF16、batch=1；$2\times36\times8\times128\times T\times2$ byte；不含 Talker |
| Talker cache、encoder 激活、音频工作区 | 5–12 | 实现/分块/图像数高度相关 |
| 框架、图捕获、碎片等 | 4–8 | 编译和后端可能改变 |
| 推理合计 | 约 31–52 | 剩余空间用于峰值，不能据此推出延迟 |

训练采用与推理不同进程/时段，单卡首期不同时实时服务和训练。训练新模块 0.1–0.25B 时，混合精度 Adam 的权重/梯度/状态按每训练参数约 16–18 byte 规划，即 1.6–4.5 GB，另加新增模块和主干反传的激活。若 C3 位于冻结主干之前，仍需要通过主干计算输入梯度，不能把整个主干放进 `no_grad`。

建议先用冻结特征缓存训练局部连接和状态，联合步骤启用激活检查点、microbatch=1、梯度累计。8k token 与 16–32 秒窗口分别试，先测峰值再加量。全参数 9B 单卡训练如使用 CPU optimizer offload/8-bit optimizer 是另一个研究配置，吞吐、精度与主机内存都须验证；本档正式初始路径是 adapters＋分组解冻，不宣称可复现大集群全量训练。

## 9. 故障退化、小脑预留与验收

可选模块失效时，状态获得 failure token：视频丢失时继续语音/文字；说话人不确定时用 unknown；BEATs 不可用时不再生成环境事件结论；检索超时用现有工作记忆并标记长期记忆暂不可得；云端失效时保留证据要求。TTS 故障降为文字，音频输入失效后靠文字/视觉，文本也未接入则只能保持状态。核心模型或 GPU 故障属于运行系统恢复职责，不保证靠神经网络自愈。

小脑只预留 typed observation `sensor_state` 和 intention `action_proposal`，字段含 frame、unit、timestamp、validity、capability_id。当前无执行器、无动作训练、无真实控制接口。

初期验收门槛是设计目标，需在固定场景/输入负载下测量后修订：

| 验证项 | 方法与初始目标 |
|---|---|
| 基线完整性 | 复现原生单视角/语音双工；禁用全部新结构时结果应与锁定 baseline 一致 |
| 实时可持续性 | 两摄像头 1 FPS、8 通道采集、单用户连续 30 分钟；队列不随时间无限增长，报告每模块 p50/p95 和峰值显存 |
| 打断与附和 | 至少 500 个“继续/纠正/旁人/环境事件”配对场景；分别统计误打断、漏打断、决策到实际静音延迟，暂停执行目标 p95<250 ms（不含等到语义可得的时间） |
| 云端确认 | 人工验证事实集；无支持的肯定输出与过度等待成对报告，过期/错人物返回必须不生效 |
| 人物连续性 | 分离音色与人格盲评；未见情境偏好、经历可追溯性、重复分享、适当沉默都计分 |
| 记忆 | 1 天/1 周模拟间隔，固定 token/storage 预算；比较记忆选择、错误更新和关键事实保留 |
| 故障 | 每个可选输入断流/延迟/恢复，观察是否出现把缺失当事实、陈旧帧当当前帧的问题 |

首个决策点不是提高参数量，而是验证：持续状态和承诺连接是否在相同主干下减少无效交流与错误断言，同时维持人物表达和实时交互。若只有话术风格变化，视为人格机制尚未验证。
