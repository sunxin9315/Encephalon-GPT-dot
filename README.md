# Encephalon · RTX 5090 路线

当前交付：**S0a 第一批工程基础**，目标系统为 Ubuntu，目标显卡为 RTX 5090 32GB。硬件尚未就绪，本批在 CPU 环境完成验证。

这不是已经可对话的人物模型，也不是完整 S0：**没有下载权重/数据、没有模型 forward、没有训练、没有 GPU 实测、没有真实录音或播放、没有调用外部 API。** 原配 B0 复现必须另行完成后，才进入新增机制和 B1 声学桥试验。

## 1. 本批实际完成

- 标准库 Python 包，Python 3.10+，运行依赖为空。本次实际测试 Python 3.12.14；3.10 仅通过语法检查，尚未做多版本运行矩阵。
- 公共数据契约：Observation、Claim、AsyncRequest/Result、OutputPacket、M+ MemoryEvent；JSON/JSONL 严格验证。
- 会话单调毫秒时钟；先按 available_at_ms 过滤，再取可见的最新 revision。未来修订不会提前隐藏当前可见版本。
- valid/missing/stale/corrupt 标准有效性；正常静音仍是 valid。旧 present 只能显式迁移；failed 必须明确是 missing 还是 corrupt 并保留故障原因。
- 人物 ID、人物版本、会话、请求版本、问题版本和前提版本隔离。旧状态返回并非一律丢弃：有明确且未变的依赖可以继续适用；前提/问题已变则拒收。
- 生成、入队、实播三边界分开；未完整播出的文字跨度不计为“已说”。队列中断分为停止请求和设备最终游标/清空确认，不能拿请求时刻冒充真实静音时刻。
- 修订去重、首次有效事件门控，避免反复刷新同一事件导致重复记忆/刺激；M+ 保留原始来源、reported/inferred 等属性和修订历史。
- 人物切换事务：停止旧表达、取消旧请求、等待播放清空确认、存档旧已播和 M+ 版本、重建运行缓存；每次激活必须使用新 session_id，同会话 utterance_id 不可复用。
- RTX 5090 设计 profile、明确未就绪的模型锁文件模板、默认 CPU-only 的环境诊断。
- 4 个明确标注为 synthetic/mock 的确定性回放场景，38 个指令、44 项结果断言；不是人工审核训练轨迹。

这里的 state_version 是工程版本号，不是实现了神经持续状态 H_t。人物支持仅到**命名空间、声音配置引用和切换隔离**；完整 PersonPack 资产加载器、adapter/音色加载、M0 知识边界验证尚未集成。

## 2. Ubuntu 上立即可运行

克隆仓库或解压项目后进入本目录。只要已有 Python 3.10+，以下命令不需要 pip、不访问网络，也不需要显卡：

```bash
python3 --version
python3 -m unittest discover -s tests -v
python3 -m encephalon doctor
python3 -m encephalon validate fixtures/canonical_records.jsonl
python3 -m encephalon replay
python3 -m encephalon check-manifest
```

最后一条**应返回退出码 2**，因为模板仍含 UNRESOLVED；这是阻止误加载的预期结果，不是测试失败。检查只判断清单结构，不证明文件真的存在、hash 与字节一致、许可证允许用途或模型相互兼容。

当前回放摘要：

```json
{
  "passed": true,
  "scenario_count": 4,
  "steps": 38,
  "assertions": 44,
  "model_inference_performed": false,
  "physical_audio_played": false
}
```

回放完整输出包含每个检查点和最终状态。修改预期值或破坏因果/身份约束会失败并返回非零；没有固定“成功”输出。

doctor 仅记录当前执行机，不会把 profile 中的 RTX 5090/32GB 当作检测结果。默认不查 GPU；今后在目标机上，可明确选择只读检测：

```bash
python3 -m encephalon doctor --probe-gpu
```

该选项只执行固定参数的 nvidia-smi 查询，最长 5 秒，不安装驱动、不导入 PyTorch、不初始化模型。结果仍不能证明 CUDA、BF16、attention 算子或模型兼容。

### 可选：打包安装

直接源码运行是最简单的离线入口。Git 仓库不提交构建产物；本次离线 wheel 构建及安装的验证记录保留在 validation/。如果本地已有 setuptools≥68 和 wheel，可无网络构建并安装：

```bash
python3 -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir dist .
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --no-deps dist/encephalon-0.1.0-py3-none-any.whl
.venv/bin/encephalon replay
.venv/bin/encephalon doctor
```

如打包工具或 venv 系统组件未安装，不要为本批先装 GPU 依赖，源码命令仍可使用。wheel 自带默认回放与配置模板，安装后从其他工作目录也能运行。自定义输入通过命令参数显式指定：

```bash
encephalon replay /absolute/path/to/fixture.json
encephalon check-manifest /absolute/path/to/models.lock.json
```

## 3. 项目结构

```text
encephalon/             Python 实现、CLI、内置默认 JSON 资源
  contracts.py          公共头、有效性、身份、观察/结论/异步契约
  serialization.py      严格 JSON 边界与记录类型往返
  clock.py              在线单调时钟/手动离线回放时钟
  observations.py       因果可见性、最新可见修订、去重和背压
  async_requests.py     本地请求状态及返回适用性检查（无网络）
  playback.py           音频生成/入队/已播账本（无音频设备）
  memory.py             有来源的会话内 M+ 修订账本（非向量库）
  runtime.py            身份切换和状态协调（非神经 core）
  replay.py             显式合成的工程回放器
  diagnostics.py        CPU 诊断、可选 GPU 查询、模型模板检查
configs/                RTX5090 profile、模型锁文件模板
fixtures/               可阅读的回放和六种标准记录示例
tests/                 标准库单元、对抗回归、CLI/打包资源测试
docs/design/           上传的原始设计文档，逐字保留
docs/review/           上传的原始审阅交接说明，逐字保留
docs/implementation/   本批契约、实施路线、B0 官方入口核查
validation/            本批真实检查输出和覆盖/未验证说明
```

configs/ 与 fixtures/ 是可阅读的源配置。默认安装资源位于 encephalon/data/；测试要求这三份 JSON 副本字节一致，修改时需同步，不允许安装版悄悄使用旧配置。

## 4. 数据和运行约束

详见 [工程契约说明](docs/implementation/runtime-contracts.md)。几个重要口径：

1. schema_version 固定为 encephalon.runtime.v1；旧设计示例需显式转换，不能直接静默接收未知字段。所有来源 ID 和派生 provenance 都保留。
2. event_start/end_ms 和 available_at_ms 是同一会话时钟；UTC 和剧情顺序不允许混入。跨设备时钟映射尚未实现。
3. validity.age_ms 是数据真正可用时刻的年龄，等于 available_at_ms − event_end_ms；读取时年龄另行计算。
4. 播放游标是同一 utterance 内的音频偏移，不是观察时间；只根据实际设备回执推进。当前回放回执全部是模拟数据。
5. queued 只是本地待发，running 只是已发出在途；没有远端队列位置、百分比或模拟器未来完成时间输入。
6. accepted async result 只代表仍适用，不代表内容真实，绝不自动把 claim 改为 supported 或自动续播。
7. Python 对象层为可信进程内 API，不是多用户权限系统。身份字段和关键版本/账本游标只读，JSON 数据不可变；尚无网络认证、加密或持久化访问控制。
8. 缓冲和日志有上限，满时明确报错/背压，不默默丢失事实。当前没有无限期运行、自动归档或磁盘崩溃恢复保证。

## 5. 验证结果与边界

本批随包测试 **164/164 通过**；最终实际执行记录见 [验证报告](validation/README.md) 和同目录日志。标准库测试覆盖有效/非法输入、乱序修订、未来信息、过期/换人返回、未播/部分已播、队列下溢、身份事务和安装资源。

不得把这些测试数量或模拟毫秒值解释为 CER、RTF、真实停止延迟、GPU 吞吐、人格质量或 200 条人工审核轨迹。本文中 32GB、64×512、80ms、8 码本等是原设计 profile，均不是本次测得的性能。

## 6. 下一步：先 B0，再机制，再 B1

### S0b：Ubuntu + RTX 5090 原配复现

硬件和系统准备好后进行，当前没有代为安装或采购：

1. 保存目标机诊断，核对实际 Ubuntu/驱动/PyTorch/CUDA/音频环境；版本尚未锁定，不照抄上游 requirements 就视为支持 5090。
2. 完成 B0 本地资产解析与完整 PersonPack 验证器，锁定官方源码 commit、权重 revision、每文件 SHA256、许可、tokenizer/processor/Mimi 版本。先只选 B0 需要的资产，不同时下载整套 B1。
3. 审查自定义模型代码及权重反序列化路径，验证所有必要感知组件确实加载；检测 missing/unexpected keys、固定 seed、记录逐模块真实 dtype/参数/shape。不要用 strict=False 或 None 感知输出掩盖加载失败。
4. 原配 MiniMind-O Thinker–Talker 冻结推理，不接随机新 core、不用新训练结果冒称原配复现。中文测试不能拿英文 mini 数据替代。
5. 建立小规模合法、可人工复核的中英文语音/图像回归和约 200 条短轨迹，再测目标/码表、首包、有效语音首包、显存、RTF、队列、真实中止到静音及 30 分钟长运行。
6. 上述证据通过后再进入 S1；失败要定位和记录，不跳过基线。

[B0 官方入口核查](docs/implementation/b0-source-audit.md) 已区分 GitHub 代码、HF 推理包和原生训练权重。观察到的远端 revision 只是只读来源记录，**不是已下载并验证的锁文件**。模型模板刻意保持 UNRESOLVED。

### S1–S5：研究推进

- S1 冻结接口：connector/resampler/cross-attention；验证输出依赖输入、梯度到达连接、冻结权重不变。
- S2 持续机制：64×512 core、快速声学、claim/策略/记忆控制；按 VAD、无状态等对照做消融。
- S3a 人物：有来源 M0、至少三种人物条件、身份/声音/临时情绪交叉验证。
- S3b B1 声音桥：Qwen3-1.7B → 原 Talker 条件必须重新训练，不是改 hidden_size 即可加载；失败时保留 B0 或明确文字实验。
- S4 分组联合：先按实际显存和梯度覆盖逐组解冻，不承诺 32GB 全量 Adam 训练全部 B1。
- S5 长期积累：外部事件和审核后离线巩固，默认不在线逐条改共享主干。

端侧分布式延后到同机基线可复现之后。完整依据和风险见 [实施路线](docs/implementation/rtx5090-implementation-plan.md)。原设计中的“审阅任务不写代码”是历史任务说明，不影响用户本次逐步实现请求；原文仍原样保存。
