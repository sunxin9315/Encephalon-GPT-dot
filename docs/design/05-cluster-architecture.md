# 大型服务器集群：初始模型架构设计

> 版本：0.1；整理日期：2026-10-14；资料核验日期：2026-10-13至2026-10-14。本文是探索性项目的可执行起始假设，不是已经验证的性能承诺。训练设计见 [06-cluster-training.md](06-cluster-training.md)，人物专项见 [07-character-memory-emotion.md](07-character-memory-emotion.md)。所有毫秒、数据规模、显存预算和新增网络宽度，除注明厂商规格外，都是待实测调整的规划值。核验限于官方模型卡、源码/配置、论文和下载说明，未实际下载权重或数据，也未运行模型。

## 1. 首期决策与研究边界

采用 **Qwen/Qwen3-Omni-30B-A3B-Instruct 原配感知编码器、Thinker、Talker、Code2Wav 作为冻结强基线**。先保证原配路径可复现，再增加持续状态、人物经历与情绪、记忆读写、结论承诺和全双工决策。这样可以区分“新增机制没有效果”与“异源模块尚未对齐”。

三个实验版本按顺序推进，不同时从零拼装：

| 版本 | 路径 | 用途与进入条件 |
|---|---|---|
| C0：冻结原配基线 | 原配 Qwen3-Omni，外加可审计的记忆输入与日志 | 测量未修改模型的理解、语音和吞吐；内置音色不代表完成目标人物复刻 |
| C1：主要研究架构 | C0＋持续状态侧路＋状态读取 Adapter＋人物、情绪、记忆和交互头＋承诺桥 | 首个主要训练版本；原配声学链先冻结，验证持续感知和人格行为 |
| C2：指定人物语音版本 | C1 的 Thinker＋神经承诺桥＋经过桥接适配的 CosyVoice3 语音网络 | 先通过原生 CosyVoice3 文本—音频基线，再训练隐状态连接；达到音质门槛后作为人物交互主声学路径 |

`Qwen/Qwen3-32B`＋独立编码器＋C2 声学端是**后续重组分支 C3**，用于检验更强稠密语义主干与可控模块解耦。它不是首日可直接等价替换的模型。`jingyaogong/minimind-3o` 用于低成本重现实验与结构消融，不让其小 Talker 限制集群最终声学上限。

第一期远程第三方云端严格只提供推理 API：不训练、不反向传播、不调用 API 生成训练集。自有集群主干的训练不改变此边界。云端结论带证据与时效性进入状态，不能自动升级成事实；网络不可用也不能使原本证据不足的结论获得发言许可。

本项目的“端到端”包括实时交互由同一内部状态协调、存在可联合训练的连续神经路径。外部 API、磁盘记忆检索和实际音频播放不属于完全可微计算图。

## 2. 硬件参考配置与资源池

用户未指定集群型号，本文采用可替换的 **8 节点×8 H200 SXM** 参考配置：

| 项目 | 参考值与来源 | 设计含义 |
|---|---|---|
| GPU | 64×H200，每卡 141 GB HBM3e、4.8 TB/s 显存带宽 | 厂商十进制容量合计 9,024 GB，不等于一个连续地址空间 |
| 节点内连接 | 8 GPU HGX/DGX 型节点，NVLink/NVSwitch；每 GPU 标称 900 GB/s NVLink | 实时模型并行优先限制在一个节点内 |
| 节点间 | 每节点多路最高 400 Gb/s ConnectX-7，配对应交换网络 | 400 Gb/s 约 50 GB/s 线速，不能写成 400 GB/s；实际聚合带宽取决于 NIC 数、布线与拥塞 |
| CPU/RAM | 按 DGX H200 参考节点配置 2 TB 系统内存 | 缓存预处理数据、索引和 CPU 任务；无需把全部原始视频放入内存 |
| 存储 | 节点 NVMe 作活动缓存，共享对象存储作原始素材和版本化检查点 | 起步规划 100–300 TB 可用空间，容量随素材码率实算，不是模型硬性需求 |

规格依据：[H200 产品页](https://www.nvidia.com/en-us/data-center/h200/)、[DGX H200 数据表](https://resources.nvidia.com/en-us-dgx-platform/dgx-h200-datasheet)、[HGX 节点连接说明](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)。这里不把 SXM 节点的全互联能力套用到普通 H200 NVL PCIe 主机。

参考资源分配：

- **节点 0：实时服务**。先用 2–4 卡承载一份 C1/C2 服务；剩余卡用于独立副本或备用。感知、快策略、Talker 与音频队列同节点；先单会话测试，再增加会话数。
- **节点 1：批量预处理与自有深推理实验**。可运行冻结 `Qwen/Qwen3-32B` 作为额外异步推理对照；它与远程 API 共用返回协议，但分别记录来源。首期不要求启用这条路径。
- **节点 2–7：48 卡训练池**。连接、人物和持续状态实验优先每次只使用 8–16 卡，多组消融并行；真正联合解冻时再扩大。

推理与训练使用独立服务进程、GPU 配额与任务队列；训练中断不应阻塞交谈。模型更新通过版本化部署切换，不能把训练正在改写的权重直接暴露给会话。

缩小到 1×8 H200 时，保留模型和数据接口，串行安排训练/推理；不能同时声称提供上述所有资源池。缩小到 2–4 节点时先减少副本和并行实验。扩大集群优先增加数据覆盖、并发副本和长轨迹实验；只有已有模块容量受限的证据后，再增加主干规模。

## 3. 可直接冻结的模块与必须自训的部分

### 3.1 模型选型及冻结边界

| 职责 | 主选、备选与获取入口 | 首期冻结内容 | 实际接口与限制 |
|---|---|---|---|
| 音视频、语音/环境声理解 | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) 原配 audio/vision encoder | 编码器与原配投影/merger 一并冻结 | 保留原处理器、位置编码、视觉 DeepStack 路径；支持音视频理解不等于已支持阵列空间定位或自主语义打断 |
| 通用语义主干 | 同一 checkpoint 的 Thinker | 词嵌入、MoE 层、输出头冻结 | 新状态通过新增 Adapter 接入；不要把 `30B-A3B` 中的激活量当总权重显存 |
| 原配语音基线 | 同一 checkpoint 的 Talker＋Code2Wav | 全冻结，保留原 Thinker–Talker 接口 | 可直接验证原配输出；配置中的内置 speaker 为 chelsie/ethan/aiden，不能声称任意目标音色直接可换 |
| 指定音色与语气 | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | 初始冻结 speech tokenizer、语音 LM、flow、HiFT；C2 逐步开放声学适配参数 | 原生接口吃文本、参考语音和指令；神经桥必须另训，不能直接把 Qwen 隐状态塞进文本 API |
| 说话人表征 | CosyVoice3 包内 `campplus.onnx` 供输出声音条件；输入识别备选 [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | 编码器冻结 | 192 维向量；窗口级说话人嵌入不是重叠语音分离、不是持久人物身份识别；会话聚类/关联另做 |
| 记忆基础检索 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 文本编码器冻结 | 使用 1,024 维 dense 向量作初始索引；事件归并、检索决策、压缩和来源校验自训 |
| 独立视觉分支，C3 才启用 | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | 只取 vision tower，冻结 | 输出宽度 1,152；无原生跨视角时序状态，不可把不同视角的 patch 当无来源同一图像 |
| 独立语音分支，C3/消融 | [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | 编码器冻结 | 输出宽度 512、16 kHz；非自回归片段识别，短窗重算不自动变成严格因果流式模型 |
| 小规模机制对照 | [jingyaogong/minimind-3o](https://huggingface.co/jingyaogong/minimind-3o) 与原配 Mimi | 按其原配加载，保持声学码本一致 | 保留 Thinker–Talker 桥用于消融；其 VAD 打断和近似双工不是本项目要训练的策略 |

许可证核验：Qwen3-Omni、Qwen3-32B、CosyVoice3 模型卡标明 Apache-2.0；SigLIP2 模型卡标明 Apache-2.0；BGE-M3 标明 MIT；SpeechBrain 此 checkpoint 标明 Apache-2.0。SenseVoiceSmall 模型卡使用 `model-license`，不得用代码仓库许可证替代权重条款；因此不是集群主线的许可依赖。MiniMind 与外部 Mimi/CAM++ 分别遵守各自模型卡，不能由一个仓库许可证统一覆盖。实施前保存选定 revision、配置和对应许可文本；本文未下载权重。

关键原生维度核验来自 [Qwen3-Omni config.json](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json)：Thinker 宽 2,048、48 层；视觉内部宽 1,152、输出宽 2,048；音频内部宽 1,280、输出宽 2,048；Talker 宽 1,024、20 层、16 个码组，`accept_hidden_layer=24`。实现必须按其配置字段定位原有桥，不把“24”擅自解释为跨库统一的层编号。

CosyVoice3 的 [官方训练配置](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/raw/main/cosyvoice3.yaml) 使用语音 LM 输入宽 896、speaker 192、25 Hz speech token、6,561 speech-token 词表、24 kHz 输出；其 token/波形体系与 Qwen3-Omni、Mimi 互不兼容。更换 Talker 时同时更换对应 tokenizer、flow/codec 和缓存协议。

### 3.2 必须自行训练的核心

| 模块 | 起始结构 | 为什么不能直接采用现成模型 |
|---|---|---|
| 跨视角与空间音频连接器 | 特征投影＋注意力重采样＋来源/时间/几何编码 | 现成 omni 通常没有本设备的阵列标定、跨摄像头语义一致性训练 |
| 持续状态 (H_t) | 6 个递归 Transformer block；256×1,024 状态槽 | 普通聊天 KV 不等价于连续感知、待办事项和自我输出的状态 |
| 人物事件评价与情绪动力学 | 人格 256、情绪 32、关系 64、评价 16；条件 MLP＋GRU/门控衰减 | 情绪识别权重不等价于目标人物“为什么这样反应” |
| 记忆写入、归并、检索、压缩控制 | 事件边界头＋检索 query 头＋等级选择器＋摘要/关系更新头 | RAG 编码器只解决相似度的一部分，未学习人格重要性与有限预算 |
| 交流动机与时机 | 多头分类＋时间分布预测 | 沉默、分享经历、回应、暂停、继续取决于人物与历史 |
| 结论证据与云端等待策略 | claim 分类/校准头＋request/value 头＋交互策略头 | 大模型自报信心不是校准器，云端状态也不是结论真值 |
| 承诺桥 | 显式可表达计划＋神经 resampler＋门控残差 | 原有隐状态桥未隔离待确认想法、可表达事实与已播放内容 |
| 人物专用 Adapter | 语义与策略 LoRA、声音风格条件 | 需要目标人物素材监督，同时保证换包不串人格 |

“目前无可直接采用”指没有核验到完整满足本项目接口和行为目标的即插即用权重，不表示世界范围不存在相近研究。

## 4. 张量、时序与连接层初始规格

### 4.1 采集、时间轴与连续感知

起始设备假设：4 路摄像头、6 通道麦克风阵列、文字输入。采集时间统一使用会话单调时钟，另附 UTC 便于历史排序。麦克风坐标、摄像头内外参可选；没有标定时明确 `geometry_valid=false`，不得把视角匹配当成可靠三维定位。

- 原始音频按 20 ms 入环形缓冲，保存所有通道；为语义编码另产生 16 kHz 单声道/波束形成流。使用实际播放信号作 AEC 参考，保留原始分支；VAD 只给声学线索。
- 声学快分支每 100 ms 更新能量、重叠、方向、speaker 候选；语义音频每 200 ms 提交最近 1–2 s 滑窗，仅发布新时间段的表征。滑窗编码器可在窗口内双向注意，但窗口结束时间必须小于等于当前可见时刻。
- 视觉起步每视角 2 fps、动态处理器尺寸限制为约 384–448 像素档；突发事件可以临时提高频率。保持原 Qwen 处理器的尺寸与 `grid_thw` 约束，不能直接用 SigLIP patch 公式解释 Qwen 视频 token 数。
- 感知 token 对应 `t_start/t_end/available_at`。双向编码使用到的完整窗口范围必须记录；禁止把 2 s 窗口末尾信息标成在窗口开头已经可用。

初期先重算有界窗口以保证行为可审计，延迟接受后再研究因果编码器蒸馏/改造。发布模型宣称的流式输出不证明任意内部路径支持不断插入新感知，因此不得直接跨窗口复用未经验证的 encoder KV。

### 4.2 感知到持续状态

主线从原配编码器最终投影后取 `F_m:[B,N_m,2048]`，新建旁路：

\[
Z_m=R_m\bigl(W_{m2}\,\mathrm{GELU}(W_{m1}\mathrm{LN}(F_m))\bigr)+E_m+E_{view}+E_{time}+E_{quality}
\]

`2048→1024→1024` MLP；每种模态使用 2 层 cross-attention resampler，8 头，内部宽 1,024。音频每个新时间桶最多 16 token，视觉每视角每帧 16 token；状态更新时接收的感知总量先限 128 token，超额按新颖性和重要性选择。来源嵌入不将同一事件多视角重复计算为多份证据。

空间音频分支输入每对麦克风的 IPD/GCC 特征及几何信息，用 2 层时频卷积＋MLP 投影成每 100 ms 4×1,024 token；speaker 192→256→1,024，附 speaker 置信度与 `unknown/overlap` 掩码。该分支自训；不会把 speaker embedding 当作声源分离器。

任何 `valid=false` 的内容从注意力中屏蔽，同时提供单独的“失效/过期”状态 token；静音用有效的空事件表示。连接器的质量门控范围 [0,1]，禁止靠 NaN 或全零向量表达不同故障类型。

### 4.3 持续状态与人物条件

\[
H_t=U_\theta(H_{t-1},Z_{\le t},M_t,P,e_{t-1},r_t,a_t,\Delta t)
\]

`H:[B,256,1024]`，6 层，16 个 attention head、FFN 宽 4,096；前 32 槽分配给话题/目标，32 槽给人物关系，32 槽给未完成与待确认结论，32 槽给当前/已播放表达，其余 128 槽自由学习。槽位分配是可消融先验，不强迫每个槽有固定自然语言含义。

人物条件统一为 `p:[B,256]`、`e:[B,32]`、`r:[B,64]`、`a:[B,16]`；投影成 8 个 1,024 维条件 token，再通过 FiLM/门控残差调制状态更新。32 维情绪仅用一个读出头预测 3 维可解释情绪坐标，未标注维度不编造真值。初始每 100 ms 更新状态，语义信息未到达时只更新可观测的快线索和时间衰减。

### 4.4 状态到原配 Thinker：缓存兼容

保持原模型提示、感知前缀、RoPE 和视觉 DeepStack 不变。C1 在 Thinker 的 8 个选定层插入新的 gated cross-attention Adapter，起点为每 6 层一处，输出投影近零初始化：

\[
x'_{\ell}=x_{\ell}+g_{\ell}\,W_o\,\mathrm{Attn}(W_q\mathrm{LN}(x_{\ell}),W_kH_t,W_vH_t)
\]

`Q:2048→512`、`K/V:1024→512`、`O:512→2048`；8 头。一次生成 step 只读取一个状态版本，更新下一 step 才可见；同一 step 不混用两次状态更新。

旧文本 self-attention KV 保留当时历史，不修改已缓存 token 的内容；状态 cross-attention 的 K/V 独立按 `state_version` 失效重建。无需每次从头 prefill 全部历史，也不声称旧 KV 自动包含了新视觉。说话时新的感知走状态旁路，不能只在 `start_pos==0` 时注入一次。

跨注意力约 25M 参数，持续状态网络约 100M；加连接器、人物和决策头后新增共享参数先预算 0.15–0.25B。数值由标准线性层数量估算，最终以实际参数表为准。

### 4.5 交互策略、结论承诺和语音桥

从池化状态＋当前语义隐状态预测：

- `interaction_action`：LISTEN、BACKCHANNEL、SPEAK、CONTINUE、PAUSE、STOP、CLARIFY、WAIT_SILENT、WAIT_BRIEF、CORRECT。
- `intent`：分享经历、讨论观点、共情、询问、解释、提醒、无主动表达。
- 每个 `claim_id` 的 SUPPORT / CONTRADICT / INSUFFICIENT / SUBJECTIVE 状态，是否需要外部确认、时效和已有来源；主观偏好不强行当事实核验。
- 请求状态为 IDLE、QUEUED、RUNNING、PARTIAL、DONE、TIMEOUT、UNAVAILABLE、CANCELLED，来自执行器可观测事实。QUEUED默认表示本地队列，RUNNING默认表示请求在途；仅当供应商返回明确回执时，才附加其后端排队/运行状态。不能据此假称知道后端处理进度或确切返回时间。

承诺桥输入当前可表达计划 `C:[B,K,2048]`、人物/情绪/风格条件及播放进度。不能只给未经筛选的整体 hidden 加一个标量 gate：候选假设可能已混入表征，标量不能保证消除其语义。训练时为可表达片段建立显式 span/claim 对应，并预测应否发言；推理时在短语边界提交、允许已提交片段含带来源的不确定表达。

原配 Qwen Talker 基线保留其已训练的 Thinker 连接；新增状态只通过残差条件和输出提交时机控制。达到人格与策略门槛前不改原声码本。

C2 的可落实神经语音适配步骤：

1. 用 CosyVoice3 的原生文本＋参考音频获得声学质量基线；这一步是可用对照，不标记为最终隐状态端到端。
2. 从 Thinker 的对应输出 span 提取 `[B,L,2048]`，经 `2048→1024→896` MLP 与 2 层单调/因果 cross-attention 形成条件；训练初期与真实文本 embedding 对齐。
3. 在 CosyVoice3 语音 LM 的选定层加入读取该条件的 gated cross-attention，保留其原生 speech-token 自回归路径、参考 speech token 和 192 维 speaker 条件。其 tokenizer 是独立的，不能逐 token 假定与 Qwen 文字 token 一一对应；按字/音节时间 span 或单调注意对齐。
4. 先冻结语音 LM 训练桥，再开放语音 LM 的 LoRA/后部层；flow、HiFT 初期仍冻结。训练标签由公开的 `speech_tokenizer_v3.onnx` 编码真实训练音频得到，不通过语音采样反传。
5. 用“文本条件保留”“文本条件随机丢弃”“纯神经条件”三级实验测量退化；纯神经条件未过可懂度和人物音色门槛时，不宣称 C2 已完成。

核验 [Qwen3-Omni 权重目录](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/tree/main) 时未发现可直接为任意真实音频生成其原配目标 codes 的独立编码器与完整训练入口，因此本期不把“原配 Talker 换任意人物并全量训练”作为已具备前提。CosyVoice3 的 [公开文件目录](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/tree/main) 有 speech tokenizer、flow、HiFT 和语音 LM，是选择 C2 的实际原因。若以后原配训练资源补齐，可再将原配 Talker 适配作为对照。

### 4.6 记忆、云端返回与感知表征的连接

检索控制器从 `[pool(H_t),p,e,r,a]` 经 `MLP→1024` 产生查询向量，并与冻结 BGE-M3 的候选向量对比；候选重排器输入查询、候选向量、事件年龄、来源与分享历史，经两层 MLP 输出分数。相同宽度不代表相同语义空间，因此该查询投影必须用检索正负例训练。

长期记忆读取分两路：BGE dense 向量 `1024→1024` 投影成事件索引 token；入选事件的文字用原 Thinker tokenizer/embedding 得到 `[B,N,2048]`，再经 `2048→1024` 形成保留顺序的文字内容 bank。精确日期、否定、承诺和引用原句不只压成一个 dense 向量。内容 bank 起始总上限1,024 token，带 `memory_id/source_id/validity`；Thinker 的状态读取 Adapter 对 `[H_t; memory_bank]` 做 cross-attention，内容 bank 的 K/V 按 `memory_version` 独立缓存。H_t 维护事件意义和使用动机，bank 提供可核对细节。

云端返回采用同样的文字 bank 接口，起始上限512 token，另以结构化 token 保存 `request_id/claim_ids/context_version/status_scope`。长返回先选择有来源的相关片段，保留完整返回引用供后续读取；不把供应商隐藏推理当必需输入。新结果的状态回执可以立即影响等待策略，具体结论只有在本地检查适用上下文和证据后才进入可表达计划。训练这两类投影和读取 Adapter 时，可微梯度到达 bank 投影与本地接收网络，不能穿过离散检索或外部API。

## 5. 人物包、长期记忆与云端接口

`PersonPack` 的稳定部分包含 `persona_id/version`、有来源的初始经历 `M0`、人格 embedding 与 Adapter `P`、声音/风格 `V`。每个实例独立持有后天记忆、关系和动态状态；共享主干不保存唯一人物的全部经历。人物切换必须同时切换 Adapter、记忆命名空间、声音条件与会话缓存，并取消旧人物未完成请求。

记忆起始配置：最近 30–60 s 感知环形缓冲、256 状态槽、4k–8k 当前语言工作窗口、按需检索的长期事件库。BGE-M3 dense 检索取 32 候选，重排最多选 8 条；其中需要精确日期/承诺的事实保留原字段，叙事段可压缩到每条 64–256 语言 token，但内容 bank 总计不超过1,024 token。超过总额时减少条数或采用更短压缩级别。总输入预算硬约束，不能无限把摘要追加进 prompt。原始媒体引用按重要性保留，时间越久不必然越不重要。

跨模块共用信封：

```json
{
  "schema_version": "0.1",
  "persona_id": "person_a",
  "persona_version": "1.0",
  "session_id": "s_001",
  "event_id": "ev_013",
  "request_id": null,
  "state_version": 42,
  "t_start_ms": 12000,
  "t_end_ms": 12200,
  "available_at_ms": 12230,
  "source": "audio.front",
  "validity": {"valid": true, "age_ms": 30, "quality": 0.91},
  "payload_ref": "tensor_or_media_reference",
  "provenance": ["recording_001"]
}
```

云端请求额外携带 `claim_ids/evidence_refs/deadline/context_version`；返回携带 `answer/evidence/uncertainty/status/status_scope/received_at`，其中 `status_scope` 区分本地执行器与供应商回执。返回后再次判断上下文是否仍适用。已过期结果可以入证据库但不自动插入当前语音，已说出的错误通过新一轮明确纠正处理。

自有重推理与第三方 API 都只交换经过筛选的任务文本、证据片段和必要媒体。不会把每帧的 256×1,024 隐状态通过广域网传给第三方，也不能假定对方有兼容隐空间。请求结束时间未知时使用真实执行状态和校准后的等待分布，不生成虚假的“马上查完”。

未来小脑预留 `sensor_event` 与 `action_intent` 类型，其中 action 带目标、坐标系、有效期、置信度和回执关联。首期输出动作接口始终关闭，只允许离线记录，不训练实际执行器控制。

## 6. 推理内存、时延和网络预算

Qwen3-Omni 完整发布文件约 70.5 GB；该磁盘统计包括多个模块，不能以“3B 激活参数×2字节”推算部署内存。以下按 BF16、单会话、8k Thinker 文本 KV 的规划口径估计：

| 项目 | 初始预算 | 说明 |
|---|---:|---|
| 原配 omni 权重 | 约 70–75 GB | 各模块实际加载精度与复制次数需统计 |
| 新模块＋人物 Adapter | 1–3 GB | 包含参数、状态及运行缓冲的余量 |
| 额外 CosyVoice3/BGE/speaker | 约 3–6 GB 权重与常驻缓冲 | C2 切走原 Talker 时可回收相应权重；导出文件大小不等于推理驻留 |
| Thinker self KV | 8k 约 0.75 GiB/会话 | 按 `2×48层×8192×4 KV头×128×2字节`，未含其他缓存 |
| 视频/音频激活、codec 工作区、CUDA/通信 | 15–35 GB 起步余量 | 对输入分辨率、窗口长度与批量极敏感 |
| 总体 | 约 90–125 GB/单会话配置 | 仅容量规划；不承诺单 H200 能满足实时延迟 |

首个低延迟部署优先比较“单 H200 主干＋单卡感知/声学”与“节点内 TP2 主干＋单卡感知/声学”。TP 并非越大越快，需实测 P95；禁止直接把一个会话的每个 token 铺到 64 卡上。使用 MoE 时活跃专家减少计算，并不减少全部专家权重驻留。

原始 6 通道 48 kHz/16-bit 音频约 0.576 MB/s；4 路压缩视频若每路 4 Mb/s，总计约 2 MB/s，实际由编码器决定。状态张量 256×1,024×BF16 约 0.5 MiB，10 Hz 全量跨服务传输约 5 MiB/s/会话；优先同进程或共享 GPU 内存，跨节点使用增量/选中槽。1000 会话会放大为 GB/s 量级，不能忽略。

时延验收预算先记录三类，不混成一个首包数：

- 快策略：每 100 ms tick 的计算 P95 <50 ms；紧急 STOP 从形成足够证据到音频停止 P95 <250 ms。
- 本地回应：用户可判断意图后到首个有意义语音 P95 <800 ms 作为起始目标；另记录用户句末到首包，避免靠提前填充话术达标。
- 语义插话：从插话中足以判断“纠正/继续”的时间点开始计时。音频语义窗口可能带来数百毫秒以上等待，不能用 VAD 延迟替代语义决策延迟。

目标未达时先缩短声学窗口、限制感知 token、拆分服务副本、优化声学队列，再考虑扩大 TP。保持 100–200 ms 可撤销播放缓冲作为起点，分别记录生成、提交和实际播放的时间戳。

## 7. 故障退化与阶段验收

| 故障 | 应保留能力 | 不能做的事 |
|---|---|---|
| 一路摄像头缺失 | 其余视角＋语音文字交流 | 使用陈旧帧假装仍在看 |
| 阵列部分通道失效 | 可用通道/单声道理解 | 沿用失效几何给出高置信方向 |
| speaker 不确定或重叠 | 继续听，必要时澄清对象 | 把混合音色当已确认身份 |
| 记忆检索超时 | 当前工作状态、基本人格 | 编造检索不到的亲身经历 |
| 云端超时 | 本地可支持内容，合理等待/澄清/延后 | 把“拿不到确认”改成“已确认” |
| Talker 故障 | 文本输出或已验证的备用声学路径 | 播放未经提交的旧缓存 |
| 整个实时节点故障 | 通过独立副本恢复已持久化状态 | 宣称单个神经网络自行解决硬件故障 |

进入 C1 联合实验的门槛：C0 测试集和日志可复现；连接器的缺失模态掩码、时间可见性、人物隔离通过确定性检查。进入 C2 的门槛：指定人物音色的文本声学基线合格、speech-token 编码可重现、神经桥比随机桥和纯文本对照具有可测效果。进入 C3 的门槛：C1/C2 的新增机制在至少 3 个随机种子及独立会话上稳定获益，且明确瓶颈来自主干而非数据或接口。

本架构优先验证持续状态、经历驱动的人格与有分寸的交流；能回答“看到了什么”只作为感知验收，不能作为完整人物交互的成功标准。

## 8. 主要来源与版本固定清单

- [Qwen3-Omni 官方仓库](https://github.com/QwenLM/Qwen3-Omni)、[技术报告](https://arxiv.org/abs/2509.17765)：原配多模态与语音结构。官方实时演示不代替本项目自主打断与待确认结论测试。
- [Qwen3-32B 模型卡](https://huggingface.co/Qwen/Qwen3-32B)、[配置](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)：C3 宽度 5,120、64 层；换用时连接层维度和 KV 预算必须重算。
- [CosyVoice3 官方仓库](https://github.com/QwenAudio/CosyVoice)：指定音色、流式音频与训练组件。论文/演示延迟不作为本项目实测。
- [SigLIP2 配置](https://huggingface.co/google/siglip2-so400m-patch14-384/raw/main/config.json)、[SenseVoiceSmall 配置](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/raw/main/config.yaml)、[SpeechBrain speaker 配置](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/raw/main/hyperparams.yaml)：独立模块的接口核验。
- [MiniMind-O 仓库](https://github.com/jingyaogong/minimind-o)：透明的小规模 Thinker–Talker 机制对照。

实施时登记模型 revision、权重 hash、tokenizer/processor、声码本版本、预处理配置、训练数据版本和 PersonPack 版本。本设计不要求永远使用上述型号；替换必须复验接口、许可证、缓存及数据标签兼容性。
