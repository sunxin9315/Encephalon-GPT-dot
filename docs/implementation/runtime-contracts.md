# S0a 工程契约 v1

日期：2026-10-17。适用本批标准库实现，不表示神经模型已接入。公共语义以原设计 README §4 优先，人物事实以 07 优先；旧示例不直接作为运行数据。

## 信封与 JSON

Header 在 JSON 中展开到记录顶层，包含 schema_version、persona_id、persona_version、session_id、state_version、event_start_ms、event_end_ms、available_at_ms、source_ids、provenance、validity、revision、request_id。全部字段必须显式存在；默认值只方便 Python 构造，不使外部 JSON 的省略合法。

- schema_version = encephalon.runtime.v1，与模型或人物版本不同。
- 时间与版本是非负整数，bool 不作为 int 接受。
- event_start_ms ≤ event_end_ms ≤ available_at_ms；点事件起止相同。
- validity = {status, quality, age_ms, reason}，status 为 valid/missing/stale/corrupt。
- quality 是有限 [0,1] 数，不代表“事实正确概率”；非 valid 必须有 reason。
- age_ms 在可用时刻计量，恰等于 available_at_ms − event_end_ms。在线 age_at(now) 在数据可用后返回 now − event_end_ms。
- source_ids 至少一项且唯一；provenance 保存派生/原始来源链，不能只留下最后摘要。
- JSON 不接受未知键、重复键、NaN、Infinity、数值溢出或二进制/tensor 对象。大特征应放外部资源，当前不加载资源。
- 输入 JSON payload 复制并递归冻结，避免调用者随后改写输入。

canonical_records.jsonl 给出 observation、claim、async_request、async_result、output_packet、memory_event 六种完全可验证的示例。

旧 present→valid 可用 normalize_legacy_validity 明确转换。failed 含义不唯一，必须 supplied failed_as=missing/corrupt 和故障原因。其他旧字段别名未自动迁移，以防 event_end 被当作 available_at。

## 观察：可见性在修订选择之前

ObservationBuffer 绑定不可更改的 Scope(persona_id, persona_version, session_id)。每条观察用 observation_id 表示同一源事件/窗口；revision 修订须保持模态、来源、真实窗口不变，且新 revision 的可用时间不得倒退。

决策时先过滤 available_at_ms ≤ now，再选该事件的最大可见 revision。未来版本即使预先进入回放缓冲，也不会影响当前版本。重复 revision 的完全相同内容是幂等；冲突内容是错误。

consume 只返回未消费的版本。change 分 inserted/revised；valid_new_event 只在该 stable ID 第一次消费到 valid 数据时为 true。重复刷新和后续修订不重复刺激；修订仍作为 revised 交给未来的状态纠正逻辑。missing/corrupt/stale 不产生有效新事件；从缺失恢复后的首次有效输入可以产生一次。

不同 observation_id 是否实际上源于同一物理事件，必须由上游分段器保证；S0a 不假装已经实现语义事件聚类。没有从文字推断人物心理的规则模型。

默认最多 4096 个观察版本。prune_before 只删已消费版本之前的历史修订，保留当前已消费与未来/未消费版本；新事件无限增长仍会达到上限并报错，需下一阶段持久化/归档。

## 异步结果：适用性不是真实性

AsyncRequest 保存当前 state_version、question_version、request_version、claim_ids、premise_versions、valid_until_ms。注册时必须与当前状态一致；dispatch 只改变本地 queued→running，未调用任何 API。

接纳同时要求：

1. 人物/人物版本/会话、request_id、request_version 精确匹配。
2. 结果已可用，且 available_at_ms 不早于对应 dispatch；请求仍在 running。
3. 结果 state_version 是请求读取版本，question_version 未变化。
4. 返回结论为请求结论子集，且仍是活跃未过期结论。
5. 结果前提快照等于请求快照，当前每项前提版本仍一致。
6. 当前 state_version 不在请求版本之前；若已推进，必须存在明确依赖快照，不能靠空依赖证明不受影响。
7. 同时满足请求和结果有效期、valid 状态、原始请求来源链。

状态推进但依赖不变的结果可适用；问题/前提改变、过期、取消、重发旧版本或跨人物返回均拒收。接纳只设通信状态 succeeded；claim、M+、播放均不因此改变。返回仍需未来独立证据评估。

status_view 只输出本地状态、已发生 elapsed_ms、请求/状态版本和 claim_ids。不暴露模拟调度的未来返回时刻，不编造 ETA。

候选结论默认最多 8 个；满时必须通过 retire_claim 明确退役后再添加，不能无界增长。注册表最多 1024 个请求 ID；拒收/接纳日志最多 1024 项。当前保留在内存中，满时报错，不承诺无限期运行。安全部署需要独立认证、授权和磁盘归档，不由这些 correlation 字段替代。

## 播放：三个高水位和两阶段停止

每个 OutputPacket 绑定 Header scope、utterance_id、sequence_id。音频相对偏移连续，序号从 0 开始；SpeechSpan 的已对齐文字跨度不能跨包或重叠。

- generated_until_ms：已生成音频高水位。
- queued_until_ms：已提交设备队列的历史高水位。
- played_until_ms：设备实际确认的高水位。

正常顺序为 played ≤ queued ≤ generated。三个高水位不可外部写入。入队必须按序且只一次，默认尚未播队列≤320ms，账本≤4096包。每包记录最早可播起点=max(入队时刻, 前包最早结束)，因此设备空闲一段时间不能给刚入队的音频“补算播放时长”。这只是物理可行性下界，不代替真实设备回执。

acknowledge_played 必须给出确切 scope 和 utterance_id，不能回退、超出队列或物理上过早。只有完全播过的 SpeechSpan 才计入 said_text/said_claim_ids；claim_ids 应标在完整主张的完成跨度，不应把分包的每个前缀都标为完整主张，实际语音对齐尚需模型集成后的核验。部分片段仍保留 played 游标，不擅自声明整句已说。

request_stop 阻止新生成/入队，但不把已有设备缓存假装清空。设备最后仍可能播出少量音频，必须通过 confirm_stopped(final_played_until_ms, …) 确认实际最后游标与清空。确认后 active_queue_ms=0，未播段逻辑作废；生成/入队历史高水位保留，已播不可撤回。晚到旧回调拒收。每次创建下一utterance前，将已停止的前一账本快照放入有界会话历史；人物切换存档包含该历史和最后一条账本，不会只保留最后一句。

当前没有音频 backend；回放只模拟回执。不能据此声称真实中止延迟已达标，也不应将待确认的设备 flush 自动当作成功。

## M+ 与人物切换

SessionMemory 只管理会话内 M+，保存 source IDs、派生链、reported/observed/inferred/hypothetical/corrected 属性。来源记录不是事实担保；原始用户报告不会自动改成事实。

同一 event_id 不能生成多个 memory_id，修订需增加 revision、保留前来源，历史版本保留且按可用时间读取。默认 2048 个版本，到上限明确报错。M0、人物知识截止、世界线、向量检索、凝练、神经记忆控制和持久化恢复未实现。

begin_switch 先阻止新计划并取消旧请求，必要时发停止请求。commit_switch 只在最终播放清空已确认后执行：存档旧快照及全部可见 M+ 版本，清空缓存/claims/前提和请求表，切换 scope/voice 引用，状态和会话单调时钟从 0 重建。最多保留64个内存会话档案。

每次激活使用从未用过的 session_id，包含 A→B→A 返回场景；同会话 utterance_id 也不可复用（默认最多 4096 个，满时必须切换新会话），避免相同标识令陈旧回调复活。旧档案是只读复制，不自动注入新会话。真实 adapter、声音条件、情绪向量和完整 PersonPack 资产尚未接入，因此不声称完成全部人物包管理。

## 并发边界与待扩展项

SessionRuntime 的协调动作使用进程内 RLock。独立 buffer/registry/ledger 对象按单一调度器串行调用设计；调用者必须先推进运行时的共享时钟再消费数据/回调，snapshot 检测到组件已观察时刻领先会话时钟会拒绝，而非暴露未来播放，不是跨进程共享内存协议；若把真实设备回调接入，应通过一个事件队列交给拥有者，不直接从多个线程改内部对象。

尚未实现网络传输、边缘时钟同步、音频PCM、AEC、GPU调度、真实感知、任何神经层/损失/optimizer、素材许可判定、磁盘事务和缓存恢复。后续新增模块必须通过同一因果/身份/实际已播边界，不能为了演示绕过校验。
