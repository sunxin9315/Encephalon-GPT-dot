# B0 MiniMind-O 官方入口核验

核验日期：2026-10-17。仅只读访问官方仓库、模型卡、文件列表和配置/源码网页。没有下载权重、执行上游代码、安装依赖、调用模型或上传任何用户数据。

## 结论

设计中的名称对应正确：MiniMind-O 是项目/代码仓库；minimind-3o 是稠密版发布模型；minimind-3o-pytorch 提供原生 PyTorch 权重。B0 应选稠密原配路径，不是 MoE，也不是把 Qwen 换进去后仍称原配。官方说明区分英文且无视觉的 mini 与中英/视觉 full，故 mini 训练不能作为中文能力复现。[官方仓库](https://github.com/jingyaogong/minimind-o)

## 已观察到的源快照

以下 revision 来自页面实际显示及提交链接，未编造。它们是本次只读观察记录，不是经下载/哈希/运行验证的项目 lockfile。网页结果可能带缓存，正式下载前还需重新核对；不承诺其必然为服务器最新 HEAD。

| 来源 | 本次可见 revision | 证据和限制 |
|---|---|---|
| GitHub 代码仓库 master | ccab6d98520ac5ac5d7583a3fa21a65105137e28 | [提交历史](https://github.com/jingyaogong/minimind-o/commits/master/)列出的2026-08-06提交。提交详情页抓取失败，不能把branch上所读源码自动当成该hash内容已核验 |
| HF Transformers 模型 | ee3febbd08cc5b2bd41c039c825a8934232fee33 | [提交页](https://huggingface.co/jingyaogong/minimind-3o/commit/ee3febbd08cc5b2bd41c039c825a8934232fee33)，文件目录可读 |
| HF 原生 PyTorch 包 | 9e22fb0e51852359ff51bd48d91ec3a345dbd75b | [提交页](https://huggingface.co/jingyaogong/minimind-3o-pytorch/commit/9e22fb0e51852359ff51bd48d91ec3a345dbd75b)，文件目录可读 |

## 模型文件与结构

- [Transformers文件列表](https://huggingface.co/jingyaogong/minimind-3o/tree/main)包含自定义model代码、tokenizer、模板、配置与pytorch_model.bin；页面未列safetensors。需审查custom_code与二进制载入路径，不将网页的安全扫描标记当完整安全保证。
- [配置](https://huggingface.co/jingyaogong/minimind-3o/blob/main/config.json)与文档的768维Thinker、8层Thinker/4层Talker、512维语音输入、768维视觉输入、64个图像token、192维speaker条件一致。桥取第3号层（零基索引）；配置为非MoE。模型配置标注BF16不等于每个推理脚本实际用BF16。
- [原生包文件列表](https://huggingface.co/jingyaogong/minimind-3o-pytorch/tree/main)同时提供语言基座、正式omni、MoE与mini训练结果等.pth文件。稠密正式基线候选为sft_omni_768.pth；不要误选llm_768.pth或sft_zero_768.pth作为正式omni发布结果。列表本身不证明权重完整性。

## 配套资源不能忽略

[作者官方合集](https://huggingface.co/collections/jingyaogong/minimind-o)列出以下配套条目：

- jingyaogong/SenseVoiceSmall
- jingyaogong/siglip2-base-p32-256-ve
- jingyaogong/mimi
- jingyaogong/campplus
- jingyaogong/minimind-o_dataset

这些是作者原配发行入口；设计中也列了原上游模型。S0b应优先复现原配资产和布局，再比较原上游替代。不能仅凭名称相似就假定重打包文件和上游完整模型的配置、目录、processor或权重字节相同。以上配套条目的具体revision与license尚未逐一锁定。

## 影响 S0b 的源码发现

[官方eval_omni.py](https://github.com/jingyaogong/minimind-o/blob/master/eval_omni.py)显示：原生加载允许缺失键；HF加载启用自定义代码；模型主体最后转为FP16；seed随时间变化；默认文本演示走英文，中文需显式选择。因此复现封装需记录missing/unexpected keys、固定seed和输入集合、打印每个组件dtype；不能把成功退出当作完整复现。路径分支还以字符串是否含model判断原生格式，封装应避免模糊路径触发错误分支。

[官方model_omni.py](https://github.com/jingyaogong/minimind-o/blob/master/model/model_omni.py)显示：感知权重单独加载；缺文件可返回空组件，视觉加载错误也可退为空；SenseVoice在该路径为FP32。感知对象使用非普通模块注册方式并在编码时禁用梯度。S0b应在期望模态缺失时明确失败；S4不能仅修改requires_grad就宣称全感知已经解冻。

这些是源码静态发现，未复现运行，不代表已确认某个固定环境必然出现错误。

## 环境前置

[官方requirements.txt](https://github.com/jingyaogong/minimind-o/blob/master/requirements.txt)实际固定了Transformers 4.57.6、FunASR 1.3.1等，但torch/torchaudio/torchvision仅在注释中，不能视为完整环境锁定。RTX5090就绪后应另据官方PyTorch/CUDA支持选择和实测，不能照搬作者旧RTX3090配置后声称兼容。

## 下一步需要形成的真正 lockfile

1. 代码仓库固定commit及所读源码hash，模型/每个encoder/codec固定HF或ModelScope revision。
2. 所有实际下载文件的SHA256、尺寸、许可文件、来源和原上游映射。
3. tokenizer、chat template、预处理、音频码本/延迟排布、特殊token及speaker条件版本。
4. Python/PyTorch/CUDA/driver/attention及音视频系统依赖，实际设备信息。
5. 加载完整性报告、固定seed与输入、组件dtype/shape/参数账本、基线指标与失败记录。

未完成：配套资源逐文件校验、权重许可逐项归档、RTX5090环境可用性、模型forward、音频结果、显存和时延实测。官方入口可读只解除“名称和来源是否存在”的疑问，不解除这些前置条件。
