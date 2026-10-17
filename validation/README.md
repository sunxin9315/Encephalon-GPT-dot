# S0a 验证记录

日期：2026-10-17 UTC。执行环境为本次实现的 CPU 执行机，不是尚未准备好的 Ubuntu / RTX 5090 目标机。实际测试 Python 3.12.14；Python 3.10 仅通过语法检查，未执行其运行矩阵。具体 Python/OS/内存信息见 cpu-diagnostics.json；宿主物理资源可能不等于容器配额，报告已注明。

## 已执行

- `python3 -m unittest discover -s tests -v`：**164/164 通过**。完整日志 unittest.txt。包含已交付的34项独立对抗回归，不要再加算成198项。
- `python3 -m compileall -q encephalon tests`：通过。
- `python3 -m encephalon replay`：4个合成场景、38个指令、44个预期断言通过；独立重复运行输出逐字节相同。完整结果 replay.json，hash 在 summary.json。
- `python3 -m encephalon validate fixtures/canonical_records.jsonl`：6种记录通过；结果 records-validation.json。
- `python3 -m encephalon doctor`：成功，默认 GPU 状态 not_probed，无框架导入、模型 forward 或网络访问。
- `python3 -m encephalon check-manifest`：预期退出码2、87个未就绪原因；结果 manifest-check.json。模板仍含 UNRESOLVED，不可以据此加载模型。
- 原设计9份Markdown（8专题+索引）及原审阅说明1份，共10份，均与上传文档逐字节一致；原始SHA256与比较结果见 original-documents.json。
- 离线wheel构建、干净venv安装及离开源码目录后的默认CLI验证：见 package-audit.json。

## 重点回归

已覆盖未来修订隐藏当前版本、重复刷新刺激、缺失与静音混淆、未到达数据/返回提前可用、人物/版本/会话/问题/前提错配、过期和重发请求、返回成功被误当事实、播放空档误算新入队音频已播、部分文字提前记为完整已说、身份字段可写导致串人物、旧utterance回调、切换旧M+及已说历史保存、缓冲上限、日志背压和运行时/组件时钟不一致。

测试属于工程契约验证，生成的数据仅synthetic/mock。没有测试数据是实际人物生平或人工审核交互轨迹。

## 没有执行，也不构成的证据

未下载模型/数据，未执行上游自定义模型代码，未进行模型 forward、训练、真实GPU查询、CUDA或5090算子测试、实际录音/播放、外部推理API、完整模型长运行。

没有 CER/WER、RTF、显存峰值、真实首包/中止延迟、人格忠实或情绪质量指标。没有完成B0原配复现或S0全部工作。尚无完整PersonPack加载器、神经core、任何训练头、持久化存储、恢复或真实设备适配。

内存有界不等于长期可靠存储；当前满载时明确报错，需后续归档/持久化设计。四个回放场景不等于设计建议的约200条人工复核轨迹。

## Git 仓库发布说明

源码、配置、测试、原始设计及事实验证报告保留。构建产物 dist/、缓存及临时文件不提交；构建日志中的执行机路径已做明确脱敏，测试结果、命令和产物 SHA256 未改动。历史 wheel 审计记录描述已完成的构建验证，不表示仓库内附带二进制 wheel。发布前再次运行 164 项测试及两次确定性回放，结果见 publication-validation.json。
