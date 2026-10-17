# 单卡 RTX 5090：初始模型架构设计

版本：v0.1；资料核验日期：2026-10-13；文档整理与一致性审查：2026-10-14。性质：探索项目的初期设计，不是已完成实现或实测性能报告。训练方案见 [02-rtx5090-training.md](02-rtx5090-training.md)，人物机制见 [07-character-memory-emotion.md](07-character-memory-emotion.md)。

## 1. 本档目标、硬件与边界

硬件限定为桌面 GeForce RTX 5090，32 GB GDDR7；以 [NVIDIA 官方产品页](https://marketplace.nvidia.com/en-us/consumer/graphics-cards/nvidia-geforce-rtx-5090/) 为准。建议主机内存 64–128 GB、NVMe 可用空间 2 TB 起；这些是工程规划，不是运行模型的最低要求。优先建立 Linux/CUDA 实验环境，Windows 主机可作为采集与展示端；部署前分别验证驱动、PyTorch 和注意力算子对该卡的支持。

初始负载：单用户会话、2 个同步摄像头、4–8 通道麦克风、中文为主。摄像头采集可为 15–30 fps，神经视觉语义分支首期只处理每视角 2 fps；阵列采集保留原多通道，语义分支输入 16 kHz，输出语音 24 kHz。多人同时讲话先做到辨别重叠和维持候选身份，不承诺完整分离所有讲话内容。

本地负责持续感知、人物和关系状态、记忆、表达意图、语音以及全双工决策。云端首期只通过异步 API 提供复杂分析；不参与训练，不调用云端制造训练数据。返回内容须带证据和时效，不能当作真值标签。动作控制只预留接口，不训练执行器。

本档分两种配置，避免把研究目标误写成现成能力：

- **B0 可复现基线：**保留 MiniMind-O 原配 Thinker–Talker，冻结已有参数，只加新接口和机制。用于验证张量、时钟、状态和实验流程，不把其小语言主干当作人物深度的最终能力上限。
- **B1 本档目标配置：**Qwen3-1.7B 作本地语义主干，沿用专用感知与音频 codec，将 MiniMind-O Talker 作为初始化。新的语义—语音连接必须训练；若仅训练连接层无法对齐，需解冻 Talker 的条件层或末端层。这是重组模块的研究实验，不是直接加载即可工作的 omni 模型。

优先完成 B0 的最小闭环，同时在文字任务上训练 B1 的持续状态与人物机制。B1 的语音桥接达标后才替换 B0。不要同时常驻两套主干进行日常推理。

~~~mermaid
flowchart LR
    O[多视角、阵列音频、文字] --> E[冻结感知编码器]
    E --> C[投影与时间重采样]
    C --> H[持续状态 H_t]
    P[PersonPack 与情绪] <--> H
    M[CPU 事件与关系记忆] <--> H
    H <--> L[B0 原 Thinker / B1 Qwen]
    H --> A[交互与结论头]
    L --> B[commitment bridge]
    A --> B
    B --> T[Talker + Mimi]
    T --> F[实际播放游标与内容]
    F --> H
    A --> G[云端请求]
    G --> H
~~~

## 2. 现有冻结模块清单

下列“流式”区分原生因果增量能力与在已收齐窗口上重复运行。配置项在下载后必须锁定 revision、文件 SHA256 和实际输出形状；本设计没有下载并运行权重。

| 模块 | 主选 checkpoint / 来源 | 初始冻结范围与接口 | 因果性、许可与适用限制 |
|---|---|---|---|
| B0 配对主干 | [jingyaogong/minimind-3o](https://huggingface.co/jingyaogong/minimind-3o)，训练格式 [jingyaogong/minimind-3o-pytorch](https://huggingface.co/jingyaogong/minimind-3o-pytorch) | 原 Thinker、Talker、原连接层全部冻结；隐空间按发布架构为 768，Thinker 8 层、Talker 4 层 | 模型卡 Apache-2.0；输入侧不能直接当作持续因果编码；现有 VAD 打断是对照基线 |
| B1 语义主干 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 冻结 embedding、28 层 Transformer、LM head；hidden=2048，16 query heads、8 KV heads、head_dim=128，见 [config](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json) | Apache-2.0；自回归文本因果，不是多模态 encoder 或已训练全双工策略；实时路径关闭长思考输出 |
| 语音语义 | [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | 冻结 encoder，输出帧特征维度 512；前端配置和长度由 [config.yaml](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/config.yaml) 读取 | 非自回归窗口模型；每次只输入已到达片段，但窗口内双向注意力不等于原生流式；权重是 FunASR 自定义模型协议，代码许可不可替代权重许可 |
| 视觉 | [google/siglip2-base-patch32-256](https://huggingface.co/google/siglip2-base-patch32-256) 的 vision tower | 只加载视觉塔，256×256 图像→64 个 768 维 patch tokens；不常驻无用的文本塔 | Apache-2.0；单帧编码，没有多视角几何和时间记忆；[配置](https://huggingface.co/google/siglip2-base-patch32-256/raw/main/config.json)给出 image_size 与 patch_size |
| 环境声音 | [Microsoft BEATs](https://github.com/microsoft/unilm/tree/master/beats) 的 BEATs_iter3+ (AS2M) 预训练模型 | 冻结特征网络，输出维度取 checkpoint cfg.encoder_embed_dim，发布 base 通常为 768；接自己的事件头 | 窗口双向 Transformer，非流式；官方 README 给出下载入口，仓库指向根目录 MIT；导入时存档权重相关许可 |
| 说话人 | [iic/speech_campplus_sv_zh-cn_16k-common](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)，[3D-Speaker 官方说明](https://github.com/modelscope/3D-Speaker) | 冻结 CAM++，16 kHz 短段→192 维向量；用于输入身份匹配与输出音色条件 | 窗口级 embedding，不是可直接冻结的在线分离器；工具仓库 Apache-2.0，模型文件许可另行存档 |
| 输出 codec | [kyutai/mimi](https://huggingface.co/kyutai/mimi) 或 MiniMind-O 配套同源版本 | 冻结编解码器；与 Talker 一致使用 8 个码本、12.5 Hz、24 kHz；不要混用不兼容码表 | 原生流式 codec；模型卡 CC-BY-4.0；编码器在离线目标预计算时运行，在线仅需解码与必要参考音频编码 |
| 记忆向量 | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 冻结，512 维；中文事件短文本→索引向量，CPU 优先 | MIT；离线/按需文本编码，非持续神经状态；英文 small 的 384 维不能套用到中文 checkpoint |
| 活动检测与阵列前端 | WebRTC VAD/AEC、GCC-PHAT 或 SRP-PHAT 的可替换实现 | VAD、声源方向、输入能量、播放回声参考形成快速特征 | 工程辅助，不把其阈值决策冒充“模型自主打断”；具体库在实现阶段锁版本 |

SenseVoice 权重许可核验入口：[官方 README 许可说明](https://github.com/QwenAudio/SenseVoice/blob/main/README.md)、[FunASR MODEL_LICENSE v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)。本设计采用其公开权重做研究，并保留模型出处；不能标注成 Apache-2.0。

备选策略：B1 认知容量不足时再对照同家族更大主干，先做文本评测和实际显存测量，不直接堆入 7B–9B omni。语音桥接未成功时允许以明确标记的文本输出完成认知实验；级联 TTS 可作为音质和体验对照，但不计为神经连接目标已实现。BEATs 首轮可关闭，用 SenseVoice 的有限事件能力跑通流程，但环境声音能力须记为降级。

## 3. 需要自行训练的模块

| 名称 | 起始结构与参数规划 | 不能由现成组件直接提供的能力 |
|---|---|---|
| 多模态 connector | 各路 LN→Linear→GELU→Linear，统一 d=512；共享或分路 2 层注意力 resampler | 带时间与来源的融合、固定 token 预算、多视角身份、质量和缺失条件 |
| 持续状态 H_t | 64 个 512 维槽，4 层更新块；每层 self-attention＋观察 cross-attention＋FFN(2048) | 跨说听阶段持续更新状态；约 15–20M 参数，实际以实现统计 |
| 快速声学状态 | 2 层 causal Conv1D＋GRU，d=128，输入能量/频谱/方向/VAD/播放状态 | 语义编码尚未更新时的即时活动判断；不负责凭空理解尚未识别的语义 |
| 人物与事件评价 | p=256、e=32、关系 r=64、评价 a=16；小型 MLP＋门控递归状态 | 个人经历影响价值判断、分享动机和情绪恢复；详见人物专项 |
| 状态→语义主干连接 | B1 在第 7/14/21/28 层后插入 bottleneck=512 的跨注意力残差，每层独立参数 | 生成期间读取新状态；保留原自注意力缓存，不更改旧 token 的历史 |
| 交互、证据和候选结论头 | 2 层 MLP；对动作、多项证据状态、是否需要云端确认分别预测 | 自主继续/停止、沉默、澄清、等待、确认到来后的调整 |
| commitment bridge | 候选表达槽＋证据门控＋因果 resampler，输出 Talker 所需 768 维条件 | 内部猜测与可表达计划分离；主干更换后的语义—声学时序对齐 |
| 记忆读写与压缩策略 | 读查询投影 512→512，写入/压缩头，事件摘要的 persona adapter | 按经历意义、未来用途和时间分配记忆预算；主动联想与避免重复分享 |
| 人物与声音适配器 | Qwen LoRA rank=8 起；Talker style 条件投影和必要的少量 LoRA | 人格选择与韵律连续性；更换 p/adapter/M0/V 与状态即可切人物 |

本档 p/e/r/a 和 d 均是建议的起始超参数，不是预训练模型固有规格。新增组件合计先限制在约 40–80M 参数；打印去重参数统计后调整。不得将未训练的随机新模块冻结，期待仅输入投影层自动学出全部机制。

## 4. 张量、时钟与连接规则

### 4.1 输入包与可见性

所有 Observation/State/Claim/CloudResult/OutputPacket 共享下列头字段：persona_id、persona_version、session_id、state_version、request_id（无请求时 null）、source_id、event_start_ms、event_end_ms、available_at_ms、validity、quality、revision。时间统一为单会话单调时钟，另存 UTC 起点。训练和在线消费者只能读取 available_at_ms≤当前时刻的数据。

validity 取 present/missing/stale/failed；静音是 present 且能量低，不能编码成 missing。位置和讲话人只能输出已知、候选或未知，不应伪造稳定 ID。

| 数据与建议更新率 | encoder 输出 | connector 输出上限 | 缓存策略 |
|---|---|---|---|
| 双视角各 2 fps | [B,2,64,768] | 各视角 8 tokens，共 [B,16,512] | 保留最近 2 秒 patch 特征及帧时间；无新帧不重复宣称新观察 |
| 语音已到达滚动窗口 0.8–2 秒，步长 320 ms 起 | [B,T_a,512] | [B,16,512] | 每次重算窗口；latest revision 替换同窗 provisional 语义，不能把重叠窗当新事件重复写记忆 |
| 环境声 2 秒窗口、步长 1 秒 | [B,T_e,D_e] | [B,4,512] | 单独背景声缓存，避免波束成形把环境事件全部滤掉 |
| CAM++ 1–2 秒有效语音，步长 0.5 秒起 | [B,192] | [B,1,512] | 按说话人候选缓存、质量加权；重叠片段降低身份确信 |
| 声学快速特征 20 ms 帧 | [B,T,features] | [B,1,512] 每 80 ms 更新 | 因果环形缓冲，不等待语义编码器 |
| 文字/云端/检索结果 | 已到达文字或 512 维检索候选 | 最多 16 个事件/记忆 tokens | 带事实来源和与当前状态的关联版本 |

“先用已到达窗口”只保证系统不偷看未来，不保证原非因果 encoder 在短窗上的效果。首期评测必须比较 0.8/1.6/3.2 秒窗口与完整句子的识别、情绪和打断质量。若依赖数百毫秒以内的语义判别，应改用真正因果编码器或训练 causal adapter/encoder；投影层不能消除等待输入窗口的成本。

### 4.2 多视角、阵列与人物条件

每个模态先做 LN 和两层投影，再加入 modality/source/view/speaker 编码、连续时间差投影和质量特征。resampler 的 query 为可训练 512 维 token，key/value 是本时刻可用特征；对缺失模态用 missing token 加 attention mask，不把全零假装为有效观察。

相机标定存在时，把内外参转换成 ray/pose 特征经 MLP 后相加；无标定时只声称跨视角语义融合，不宣称三维几何重建。阵列保留两条通路：AEC 后参考/波束语音送语义 encoder，原多通道的相位差与方向候选送空间 MLP，同时保留环境声参考通路。重叠语音的原始通道需留待后续分离实验。

人物条件 p256、情绪 e32、关系 r64、评价 a16 分别经独立投影进入 core，避免把音色向量误作人格。检索首先按 persona_id 做硬隔离，再对候选内容学习排序。

### 4.3 持续状态更新

H_t∈R^(B×64×512)。每 80 ms 读取新到达的观察，不要求每路均更新。更新块采用 pre-norm residual，自注意力维护槽间关系，cross-attention 读取带时间戳的观察与检索；门控融合旧状态：

    U_t = Core(H_(t-1), Z_available, P, E, CloudStatus)
    G_t = sigmoid(MLP([H_(t-1), U_t, Δt, validity]))
    H_t = (1-G_t)⊙H_(t-1) + G_t⊙U_t

槽位可分组为场景、当前话题、关系、待确认事项和当前输出，各组只是归纳偏置，仍需学习。长时人物经历写入独立存储，不把有限 H_t 当永久记忆。训练使用 8 秒截断反传窗口；在线更新不截断会话，但按事件做显式状态摘要与检验，防止长期漂移。

### 4.4 语义主干、KV cache 与表达

B1 每个指定层新增 Query(2048→512)、K/V(512→512)、Output(512→2048) 的 cross-attention，乘 tanh 标量门后加入原残差；门初始为小值或零。H_t 变更时只更新这些附加 K/V；原文本 KV cache 不原地篡改。新 token 读取新状态，旧文本仍代表当时已经处理的历史。

B0 插入位置按 8 层主干选第 2/4/6/8 层，输入输出维度改为 768。状态更新与生成任务分开调度；语义主干不必每个 80 ms 都完整运行。短反馈可由策略启动已有表达计划，不能依靠固定“嗯、我想想”无限填充。

候选结论使用最多 8 个 claim slots，字段含内容、证据来源、unsupported_probability、requires_cloud、scope、valid_until、是否已经说出。初期显式标注结论跨度；后期训练 claim query 读取语义 hidden 并预测这些字段。模型决策为主，运行时只执行已决定的播放/取消动作。

commitment bridge 接收被选中的表达内容、人格/情绪和说话风格，不能给 Talker 无差别暴露全部待确认假设。硬选择在推理时执行；训练采用标注选择或软门控，不能宣称 hard gate 之后仍处处可微。门控降低误说概率，不构成事实保证。

B0 保留原 8 码本延迟排布，先验证新状态不会破坏原 Talker。B1 使用因果注意力 resampler：每个 80 ms 音频帧 query 读取截至此刻已形成的表达 hidden（2048→768），再输入原 Talker 条件位置；参考码与 CAM++ 向量仍走原接口。B1 的 tokenizer、文本步与音频步分布不同，不能只修改 hidden_size。先进行原 Talker 条件分布对齐和 teacher forcing 音频损失；必要时解冻 embed_proj、前两层 Talker，并重新训练时序桥接。若冻结 Talker 不收敛，这属于可接受的实验否定结果。

音频实际播放以 80–160 ms 包为起始设置。记录 queued/generated/played 三个游标；决定停止时立即丢弃尚未播放包，已播放内容只能通过之后的语言纠正。音频设备缓存必须一同计入停止延迟。

## 5. 记忆、云端、人物切换

人物采用 PersonPack：persona_id/version、启动前经历 M0、p 与 persona adapter、speaker/style V、独立后天事件索引与运行状态。具体方案见 [人物专项](07-character-memory-emotion.md)。本档默认所有长时记忆在 CPU/NVMe；GPU 只放候选 top-8 内容的编码和有限工作状态。

CPU 上保存原事件→压缩事件→语义/关系三层版本，不原地覆盖互相矛盾的旧认识。检索由当前话题和人物内部关注共同发起；已向某人分享过的经历须有记录。写入/压缩异步运行，不阻塞实时语音。禁止把每帧视觉描述自动写成经历或自动播报。

云端状态来自真实请求器：unavailable/queued/running/succeeded/failed/timeout。若供应商不暴露后台进度，queued 只表示本地待派发，running 只表示请求已发出且在途，不能声称知道远端排队位置或完成比例。模型同时接收 elapsed_ms、经验延迟区间、证据是否返回、上下文版本；不知道准确剩余时间时不编造 ETA。返回用 request_id、persona_id 和 state_version 判断是否仍适用；过期结果进入可检索记录或丢弃，不直接插播。

切换人物需切 adapter、初始经历与后天索引、声音条件、情绪状态，清空不兼容 KV/音频队列，取消或隔离旧请求。历史是否共享必须显式配置，默认不共享。

未来动作接口只定义 ActionIntent{modality, target, parameters, state_version, expires_at} 与外部反馈 Observation；本期不生成物理执行指令或训练动作损失。

## 6. CPU/GPU 边界与故障退化

| CPU / 系统侧 | GPU / 神经侧 |
|---|---|
| 设备采集、时钟同步、AEC、环形缓存、VAD、方向估计、网络请求、SQLite/向量索引、音频播放游标 | 感知 encoder、连接层、持续状态、语义主干、人物/证据/交互头、Talker/Mimi |
| 分片数据预处理、素材出处、PersonPack 切换事务、超时与故障检测 | 训练时现有权重冻结；需要传回输入梯度的冻结主干仍保留 autograd |

单卡调度优先级：播放包生成与快速状态更新优先，随后语音语义、按需主干生成，最后视觉、环境声和记忆编码。限制长 prefill 和大批量视觉任务造成不可抢占的 GPU 阻塞；通过实际 Nsight/时间戳测量调整。CPU 卸载可以节省显存，但不能预先保证延迟。

| 故障 | 退化方式 | 必须避免 |
|---|---|---|
| 单摄像头掉线 | missing 标记，使用其余视角和已知过期时间 | 将缓存画面当实时状态 |
| 麦克风丢包/重叠严重 | 降低语义确信、保留文字交互、必要时澄清 | 将错误转写作为已确认事实 |
| 记忆检索超时 | 用工作记忆继续，减少个人经历断言 | 临时编造经历填补空白 |
| 云端不可用 | 继续低依赖话题，保留未确认状态 | 因不能核验而降低证据门槛 |
| Talker/codec 失败 | 停止失真播放，允许文本输出 | 让音频循环/陈旧队列继续播 |
| GPU OOM/核心进程崩溃 | 外部 watchdog 重启、恢复已提交事件，告知中断 | 承诺整张 GPU 失效仍保持神经服务连续 |

训练包含 modality dropout、延迟、陈旧数据、恢复以及质量变化；故障隔离需要工程机制和模型训练共同承担。

## 7. 规划显存与训练可行性

以下单位为 GiB 近似规划，batch=1 推理、文本缓存上限 4096 tokens、双视角低频编码、BF16 为主；不是实测。SenseVoice 若需 FP32 数值稳定，按 FP32 计入。

| 项目 | B1 推理预留 |
|---|---:|
| Qwen3-1.7B 权重 | 约 3.2–3.5 |
| 语音/视觉/环境声 encoder、CAM++、Talker、Mimi | 约 1.5–2.3 |
| 新模块和人物 adapter | 约 0.1–0.3 |
| Qwen 原始 KV（4096 tokens） | 约 0.875 |
| 持续状态、附加 cross-attention KV、Talker KV、输入特征 | 约 0.5–1.5 |
| 激活、算子 workspace、分配器与设备上下文 | 约 4–8 |
| 峰值余量 | 约 3–5 |
| 合计规划 | 约 14–22；先将稳态峰值目标设为 ≤24 |

Qwen KV 的估算式为 2(K,V)×28 层×8 KV heads×128×4096×2 bytes；不是按 query heads 估算。B0 权重规模显著更小，但实际速度仍取决于 kernel、重算窗口和音频调度。[MiniMind-O 参数统计](https://github.com/jingyaogong/minimind-o#-模型细节)可用于初始核算，不能用“0.1B”替代全部运行权重。

连接层训练：冻结 encoder 可预计算特征；冻结主干用于输出损失时仍需保存/重算传入连接层的梯度。B1 microbatch=1、长度 1024–2048、gradient checkpointing、梯度累积 16 起，新增 40–80M 参数的常规 Adam 状态规划约 0.6–1.4 GiB；总训练峰值先按 18–28 GiB 预留，并实测调整。

B0 原发布组合约 0.54B，加入本方案的 BEATs、持续状态与其他新模块后，完整训练规模预计约 0.65–0.75B（BGE 保持在独立检索分支且冻结，最终以参数账本为准）。该规模可用于“全部可微本地模块解冻”的小规模研究，但完整感知激活仍可能成为瓶颈。原 MiniMind-O 对外部 encoder 使用冻结与 no_grad 路径，实际全解冻必须改变参数注册、梯度上下文和优化器分组；切换其 all 模式不会自动做到。B1 约 2B 以上总参数若使用常规混合精度 Adam，参数/梯度/主副本/动量等约 16–18 bytes/可训练参数，尚未计激活就接近或超过 32 GB，因此**本档不承诺 B1 全模块同时完整 Adam 训练**。逐组解冻、8-bit optimizer、CPU offload 可研究，吞吐需要测量；量化低秩训练也不等于所有权重都训练。codec 量化属于离散边界，完整解冻时需采用其训练定义的估计路径，并不使 RAG/API/播放控制自动可微。

## 8. 首轮实验与验收门槛

所有阈值为建议实验门槛，需依据实测迭代；结果报告同时给分母、场景与置信区间。

1. B0 原配复现：小规模中英文语音/图像回归集，记录 CER、说话人相似度、自然度和 RTF；不要把英文 mini 训练当作中文模型复现。
2. 新连接层：在冻结主干上验证输入标签、特征变化确实影响输出；与固定 MLP 对照。失效标记、时钟泄漏和人物隔离检查必须通过。
3. 持续状态＋全双工：至少 1,000 个独立事件，覆盖附和/纠正/旁人讲话/环境突发。相对 VAD 停止基线，错误打断率目标降低 30%，有效打断召回不得下降超过 5 个百分点；停止延迟分别报告“首音→停止”和“语义足够可判别时刻→停止”，后者 p95 起始目标 ≤300 ms。
4. 证据＋云端等待：至少 500 个可核验案例与相同数量不需要云端案例。报告支持不足却肯定陈述、过度等待、无效填充、过期结果误用；不能通过一直沉默取得好指标。过期版本错误接纳和人物串话在确定性回放中目标为零。
5. 人格与记忆：接口 smoke test 可先用 2 个 PersonPack；身份可分离验证至少采用 3 个条件（如人物 A、人物 B、中性基线），使用未参与训练的事件和话题，交叉更换记忆、人格参数与声音条件。检查经历依据、价值取舍、重复分享及沉默是否合适。盲评另比较“只有人物 prompt”“prompt＋RAG”“本设计”。
6. B1 桥接：先在相同小集合与 B0/文本通路比较，再扩音频；若解冻连接层后仍无稳定语音，记录失败并调整 Talker，不掩盖为人格问题。
7. 长运行：单会话 30 分钟以上，无缓存无界增长，记录 GPU 峰值、丢帧、平均/p95/p99 包生成时间；音频 RTF<1 为基本条件，建议留余量至 p95 包生成耗时<包时长的 0.8 倍。不能把模型单次前向延迟代替完整交互延迟。

## 9. 实现前锁定事项

- 固定模型 revision、预处理、tokenizer、Mimi 码本版本与数据分片来源；未下载的 checkpoint 尺寸和许可在导入时复核。
- 优先验证 B1 更换语义主干后的语音桥接和短窗 SenseVoice 质量，这是本档最大的两个模型风险。
- 首轮不训练强三维理解、多人全分离或长期在线修改主干权重；用可追溯经历存储和小人物 adapter 验证人格连续性。
- 每项设计变更须记录前后实验配置，允许否定原假设；本文件中的模块名称和数值不应变成不可修改的产品承诺。
