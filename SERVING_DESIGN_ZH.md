# LLAISYS 请求池、调度器与会话管理

## 1. 三个概念的区别

- **用户（User）**：使用服务的人，一个用户可以拥有多个相互独立的会话。
- **会话（Session）**：一段连续对话，保存自己的 token 历史。切换会话不会串用上下文。
- **请求（Request）**：会话中的一次生成任务，例如用户发送一条新消息。

它们的关系是：

```text
用户 user-1
├── 会话 conversation-a
│   ├── 请求 request-1
│   └── 请求 request-3
└── 会话 conversation-b
    └── 请求 request-2
```

请求池保存尚未执行的请求；调度器从请求池中取出请求交给 Qwen2；会话管理器负责在执行前拼接历史，在成功后提交新的完整 token 历史。

## 2. 当前调度策略

`RequestPool` 是线程安全的优先级 FIFO 队列：

1. `priority` 数字大的请求先执行；
2. 优先级相同时按提交顺序执行；
3. 同一个会话同时只允许一个未完成请求，防止对话轮次乱序；
4. 不同用户或不同会话可以同时进入请求池；
5. `RequestScheduler` 使用单个模型实例逐个执行请求。

当前调度器没有声称支持 Continuous Batching。原因是 C++ `LlaisysQwen2Model` 当前只有一套 `cache_len`、`key_cache` 和 `value_cache`。如果把会话 A 和会话 B 的 Decode 步骤交错执行，两段 KV Cache 会互相覆盖。

因此，当前实现保证的是**多会话语义正确和统一排队**，不是多个会话在一个 GPU Batch 中并行 Decode。

### 功能优先的逐 Token 调度

`RoundRobinScheduler` 提供另一种运行方式：每轮只让一个活跃请求生成一个 Token，然后切换到下一个请求。

```text
请求 A -> Token A1
请求 B -> Token B1
请求 C -> Token C1
请求 A -> Token A2
...
```

由于底层还没有多序列 KV Cache，每个 Token 步骤都会使用“完整上下文 + 已生成 Token”重新执行 Prefill。这种实现计算量较大，但具备完整且可验证的功能语义：

- 请求按优先级进入活跃队列，随后 Round-Robin 公平轮询；
- 每生成一个 Token 立即产生 `TokenEvent`；
- 支持同步生成器和后台工作线程；
- 支持 EOS、用户指定 Stop Token、最大生成长度；
- 支持等待中取消、生成中取消和超时；
- 取消或超时时可保留已经流式返回的部分结果；
- 后端错误不会污染上一次成功的会话历史。

```python
scheduler = llaisys.RoundRobinScheduler(
    model,
    token_decoder=lambda token_id: tokenizer.decode([token_id]),
)
scheduler.sessions.create("chat-1", user_id="user-1")
scheduler.start()

request = scheduler.submit(
    "chat-1",
    input_tokens=[101, 102, 103],
    max_new_tokens=128,
    stop_token_ids=[151643],
    timeout_seconds=60,
)

for event in scheduler.events(request.request_id, timeout=5):
    if event.text:
        print(event.text, end="", flush=True)
    if event.finished:
        print("finish_reason:", event.finish_reason)

scheduler.stop()
```

这个上层接口以后可以直接连接真正的 Batched Decode；届时只替换底层的一步生成逻辑，不需要改变请求、会话、取消和事件接口。

### 结构化消息与 Chat Template

`ChatService` 在 Token 会话之上保存 `system`、`user`、`assistant` 三种结构化消息。每一轮生成前，它使用模型 Tokenizer 的 `apply_chat_template()` 重建完整对话输入；生成完成后，再把新回复写回该会话。

```python
chat = llaisys.ChatService(scheduler, tokenizer)
chat.create_session(
    "study-chat",
    user_id="user-1",
    system_prompt="你是一名计算机系统课程助教。",
)

request = chat.submit_message(
    "study-chat",
    "解释一下 GQA。",
    max_new_tokens=128,
)

for event in chat.run_until_idle_stream():
    if event.text:
        print(event.text, end="", flush=True)
```

会话可以导出和重新导入，不依赖数据库：

```python
data = chat.export_session("study-chat")
copy = chat.import_session(data, session_id="study-chat-copy")
```

结构化消息是会话的真实来源。当前每轮重新应用 Chat Template 并 Prefill 完整历史，所以速度不如持久化多序列 KV Cache，但不会因为手工拼接 Token 而破坏角色分隔符或对话格式。

## 3. 使用示例

```python
import llaisys

model = llaisys.models.Qwen2("/path/to/model", llaisys.DeviceType.NVIDIA)
scheduler = llaisys.RequestScheduler(model)

# 同一用户创建两个互不影响的会话。
scheduler.sessions.create("work-chat", user_id="user-1")
scheduler.sessions.create("study-chat", user_id="user-1")

# input_tokens 需要包含本轮对话模板需要的角色和分隔 token。
request_1 = scheduler.submit(
    "work-chat",
    input_tokens=[101, 102, 103],
    max_new_tokens=64,
    priority=5,
)
request_2 = scheduler.submit(
    "study-chat",
    input_tokens=[201, 202],
    max_new_tokens=128,
    priority=0,
)

# 单工作线程运行，优先级较高的 request_1 先执行。
finished = scheduler.run_until_idle()

print(request_1.status, request_1.output_tokens)
print("new tokens:", request_1.generated_tokens)
print(request_2.status, request_2.output_tokens)

# 查询 user-1 的全部会话。
for session in scheduler.sessions.list(user_id="user-1"):
    print(session.session_id, len(session.token_history))
```

`Qwen2.generate()` 返回“输入上下文 + 新生成 token”，调度器会验证此前缀，并只在推理成功后更新会话历史。如果模型推理失败，会话仍保留上一次成功的历史。

## 4. 单个用户的不同会话

单用户多会话与多用户在调度层没有本质区别，隔离依据是 `session_id`，而不是 `user_id`：

```text
user-1/work-chat  -> 独立 token 历史
user-1/study-chat -> 独立 token 历史
user-2/chat       -> 独立 token 历史
```

`user_id` 用于归属、查询和权限管理，`session_id` 才决定对话上下文。删除或清空一个会话不会影响同一用户的其他会话。

## 5. 与完整推理服务的差距

要实现图中类似 vLLM 的连续批处理，下一阶段需要把模型权重与序列状态分离：

1. 每个请求拥有独立的 `cache_len` 和 KV Cache 映射；
2. 增加 KV Cache block/page 分配器；
3. C++ Infer 接口接收多个 sequence ID；
4. 调度器每轮选择多个活跃请求组成 Decode Batch；
5. 支持抢占、换出、流式 token 返回和请求结束后的 Cache 回收。

这部分会改变 Qwen2 后端的数据结构和算子输入布局，属于下一阶段的推理引擎改造，而不是简单增加一个 Python 队列。
