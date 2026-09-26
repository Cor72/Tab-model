# kagami-tab-model 项目长期笔记

## 本机硬件约束（影响所有技术选型）

- **无 NVIDIA 独显**，只有 Intel UHD 核显 → 所有 GPU 方案（CUDA / bitsandbytes 量化）不可用
- CPU：i5-13420H，内存 15.7 GB，D 盘剩余约 100 GB
- Python 3.13.14（managed），未装 uv，无 HF 缓存目录
- 国内网络：HuggingFace 直连不通，模型需走 ModelScope 或 hf-mirror 镜像

## 项目方向

local-completion：基于 Qwen3-0.6B 的本地 inline completion（类 Copilot），
核心引擎自己写，Web 框架/VS Code API 用现成的。

## 已确认的技术决策

- 模型加载 dtype：bfloat16 或 float32，禁用 float16
- Qwen3 必须显式关闭 thinking 模式，否则补全被思考内容污染
- CPU 量化路线：GGUF（llama.cpp）或 ONNX Runtime，不用 bitsandbytes
- 评估集要早建，不要等最后才 benchmark

## 用户偏好

- 指令简洁、改动范围边界明确，严格遵守
- 不要客套开场白，先给结论
