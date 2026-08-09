# LLAISYS HTTP 推理服务使用说明

## 1. 已实现的功能

当前服务是一个不依赖 Web 框架的轻量推理服务，包含：

- OpenAI 风格的 Chat Completions 与 Text Completions API；
- 普通 JSON 返回和 SSE 流式 Token 返回；
- 多请求请求池、优先级和逐 Token Round-Robin 调度；
- 单用户多会话及多用户多会话隔离；
- system、user、assistant 结构化消息和 Chat Template；
- 请求状态查询、等待/运行中取消、超时和 Stop Token；
- Argmax、Temperature、Top-k、Top-p、重复惩罚和随机种子；
- 健康检查、模型发现、JSON 状态和 Prometheus 文本指标；
- TTFT、请求总时延、Token 数、成功/失败/取消请求统计；
- 内存中的会话导入与导出。

按项目范围，未实现网页界面、用户登录、数据库持久化、多机分布式、
Tensor Parallel、INT4 量化和 Speculative Decoding。

## 2. 启动服务

先完成项目编译和安装，再执行：

```bash
export PYTHONPATH="$PWD/python:$PWD"
python -m llaisys.serve \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia \
  --host 0.0.0.0 \
  --port 8000 \
  --model-id deepseek-r1-qwen-1.5b
```

CPU 环境将 `--device nvidia` 改为 `--device cpu`。只允许本机访问时，
应使用默认的 `--host 127.0.0.1`。

## 3. OpenAI 风格接口

非流式聊天：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-qwen-1.5b",
    "messages": [{"role": "user", "content": "解释什么是 KV Cache"}],
    "max_tokens": 128,
    "top_k": 1
  }'
```

SSE 流式聊天：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-qwen-1.5b",
    "messages": [{"role": "user", "content": "介绍 GQA"}],
    "max_tokens": 128,
    "stream": true
  }'
```

纯文本补全使用 `POST /v1/completions`，请求体中的 `prompt` 必须是字符串。

## 4. 持续会话和异步请求

创建一个内存会话：

```bash
curl http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "study-chat",
    "user_id": "user-1",
    "system_prompt": "你是一名大模型推理课程助教"
  }'
```

异步提交新消息：

```bash
curl http://127.0.0.1:8000/sessions/study-chat/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "继续讲解 KV Cache", "max_tokens": 128}'
```

接口立即返回 `request_id`，随后可查询：

```bash
curl http://127.0.0.1:8000/requests/REQUEST_ID
curl http://127.0.0.1:8000/sessions/study-chat
```

请求完成后，assistant 消息会自动写回相应会话。取消请求：

```bash
curl -X POST http://127.0.0.1:8000/requests/REQUEST_ID/cancel
```

删除会话：

```bash
curl -X DELETE http://127.0.0.1:8000/sessions/study-chat
```

## 5. 监控接口

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics
```

`/metrics` 输出 Prometheus 文本格式，包括请求总数、当前活跃数、成功数、
失败数、取消数、输入/输出 Token 数、平均首 Token 延迟（TTFT）和平均请求时延。

## 6. 当前性能边界

当前 C++ 模型只有一套 KV Cache。为了保证多个会话结果互不污染，Round-Robin
调度器在每个 Token 步骤使用完整上下文重新推理。因此，这一版本强调接口和功能
完整性，不代表高性能 Continuous Batching。后续若追求吞吐量，需要实现每请求独立的
Paged KV Cache，并让一次 C++ Decode 同时接收多个 sequence。
