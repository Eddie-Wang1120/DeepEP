# Megakernel Compute Design Notes

## 背景

当前 MK-v7 megakernel 已经把 dispatch 和 combine 融合到同一个 persistent CUDA kernel 中，并通过 logical channel 做通信流水。但 compute 仍处于占位/模拟阶段，尚未真正进入 dispatch -> compute -> combine 的数据路径。

本设计文档总结第一版 compute 接入方案：优先保证真实 compute 路径闭环、信号正确和 combine 可按 token ready 推进；暂不追求复杂动态调度和极限性能。

## 重构起点：通信已对齐，compute 仍是占位

当前 `megakernel.cu` 的实际状态（区别于本文档早期设想，重构前必须对齐认知）：

```text
dispatch  : 已对齐。NVL receiver 直接把原始 token 写入 combine_input（megakernel.cu:1032），
            并在 topk 循环前一次性算好 token_compute_expected，再自旋保证 slot 数据先于
            expert_recv_count 可见（megakernel.cu:1040-1082）。
combine   : 已对齐。combine_x 直接指向 combine_input（megakernel.cu:2961），
            NVL sender 按 token 顺序自旋等待 combine_token_ready，带 timeout→trap
            （megakernel.cu:1708-1722）。
compute   : 占位。compute_worker 是「单 SM 处理一个 expert」的 round-robin
            （megakernel.cu:1192），内部只 __nanosleep ~100us（megakernel.cu:1235-1240），
            然后发 per-token-ready。完全不做 GEMM、不写任何输出。
            另存在一个已分配但未接线的 compute_output 缓冲（megakernel.cu:2761）。
```

关键事实：**combine 当前发的是 dispatch 写进 combine_input 的原始 token，而不是 FFN 输出**。compute 只是一个纯 gate（pure gate）。重构的本质就是：让 compute 真正算 FFN 并把结果接到 combine 的输入，同时维持现有信号协议不破。

本文档「## SM 划分」「## Compute Batch 粒度」「## Consumer Group 内同步」等章节描述的是 `3×32 SM consumer group` 设想，与现有「单 SM/expert」代码不一致，相关取舍见文末「## 待决策点」。

## 目标

把真实 MoE FFN compute 接入 megakernel：

```text
dispatch -> compute -> combine
```

每个 token 的本地 expert 计算完成后，通过 per-token-ready 信号放行 combine，使 combine 不必等待所有 expert 或所有 compute 完成。

## SM 划分

> 注意：以下 `3×32 consumer group` 是本文档早期设想。现有代码实际是「单 SM 处理一个 expert」的 round-robin（megakernel.cu:1192），且 launch 时 `compute = total - dispatch - combine = 148 - 24 - 24 = 100`（无 reserved）。两套方案差异很大（是否需要 group barrier、scratch 布局、SM 数都不同），见文末「## 待决策点」决策 D1。

硬件总 SM 数：

```text
total SM = 148
```

第一版静态划分：

```text
dispatch = 24 SM
combine  = 24 SM
compute  = 96 SM = 3 个 consumer group * 32 SM
reserved = 4 SM
```

其中每个 compute consumer group 固定 32 个 SM，一次处理一个 expert 的一个 batch。

## Compute Batch 粒度

主路径 batch size：

```text
128 tokens / expert / batch
```

每个 batch 执行完整 MoE FFN：

```text
W_gate GEMM
W_up GEMM
SwiGLU
W_down GEMM
```

典型形状：

```text
A: [128, hidden]
W_gate/W_up: [intermediate, hidden]
W_down: [hidden, intermediate]
hidden = 4096
intermediate = 4096
dtype = BF16
```

### GEMM 形状与转置约定（必须与现有 device_gemm_bf16 一致）

当前代码第一版仍按 `COMPUTE_BATCH_SIZE=128` 聚合/flush work unit，但每个 work unit 内先用 per-token `M=1` GEMM 跑通正确性闭环，尚未实现本文档主推的 `M=128` batch GEMM。由于 WMMA `load_matrix_sync` 会按 16 行 tile 读取，`M=1` 输入必须先拷到带 16 行 padding 的 workspace，不能直接指向 compact token buffer。

现有 `device_gemm_bf16`（megakernel.cu:251）语义是 `C = A @ B^T`：A 取 `row_major`，B 取 `col_major`，即 `C[M,N] = A[M,K] @ B[N,K]^T`。三次 GEMM 的映射：

```text
gate = A @ W_gate^T
  A      : [M, hidden]              (row_major)
  W_gate : [intermediate, hidden]   (B = [N=intermediate, K=hidden], col_major)
  -> gate: [M, intermediate]

up   = A @ W_up^T   （同 gate 形状）

down = act @ W_down^T
  act    : [M, intermediate]        (row_major)
  W_down : [hidden, intermediate]   (B = [N=hidden, K=intermediate], col_major)
  -> down: [M, hidden]
```

三次都满足 `C=A@B^T` 的 col_major B 语义，可直接复用 `device_gemm_bf16`。约束：

```text
K 维（hidden / intermediate）必须是 WMMA_K(16) 的倍数；
tail batch 的 M 非 16 倍数时，靠 device_gemm_bf16 内 out_row<M 兜底，
  但要确认不会越界读 A（A 行数 = M，tile 读取需在 K 维对齐、M 维裁剪）。
```

### Route Weight 应用位置（精度对齐高发坑）

baseline 在 expert 内、`W_down 之前` 乘 route weight，并对同 token 命中同 expert 的多个 topk slot 做 `sum`（test_megakernel_v7.py:100-101）：

```python
weights = (recv_topk_weights[idx] * mask[idx].float()).sum(dim=1, keepdim=True)
swiglu_out = swiglu_out * weights      # W_down 之前
down = swiglu_out @ W_down[e].T
```

megakernel 重构必须：

```text
route weight 只乘一次，且位置与 baseline 一致：SwiGLU 后、W_down 之前；
权重取自 combine_input_topk_weights（megakernel.cu:1060 已由 dispatch 写入）；
确认 combine 端（DeepEP combine 协议）不再二次加权，否则 bitwise 对不齐。
```

对于 `128 x 4096 @ 4096 x 4096`，如果按 `128 x 128` 的 N 维 tile 切分，N 维共有：

```text
4096 / 128 = 32 tiles
```

因此 32 个 SM 对应一个 expert batch 是合理的第一版映射。

## Compute Task Queue

建议引入 compute task queue。队列元素是 batch descriptor：

```cpp
struct ComputeTask {
    int expert_id;
    int start_slot;
    int num_tokens;          // 128 或 tail < 128
    int logical_channel_id;  // 第一版可不使用，第二版做 per-channel flush 时需要
};
```

第一版入队规则：

```text
正常阶段：
  expert_recv_count[expert] - enqueue_cursor[expert] >= 128
  -> 入队一个 128-token full batch

收尾阶段：
  dispatch 全局结束后
  expert_recv_count[expert] - enqueue_cursor[expert] > 0
  -> 入队一个 tail batch
```

`enqueue_cursor[expert]` 记录该 expert 已经入队到哪个 expert-local slot，避免重复入队。

## 不足 128 Token 的处理

不要在正常 dispatch 过程中把不足 128 的 batch 直接塞进队列，否则会打碎主路径，降低 TensorCore 利用率。

第一版策略：

```text
满 128 才进入主路径计算；
dispatch 全局结束后，不足 128 的 tail batch 才 flush。
```

这样可以避免 compute 因尾批不足 128 而死等，同时保持主路径 batch GEMM 的效率。

### ⚠ 当前测试规模触发不了主路径

现有 `compute_worker` 的阈值：

```cpp
bool should_compute = (ready_tokens >= COMPUTE_BATCH_SIZE);  // 128 (megakernel.cu:1199)
```

但 `test_megakernel_v7.py` 配置 `num_tokens=16, hidden=256, experts_per_rank=8`，单 expert 收到的 token 远不足 128，**正常 128 主路径一次都不会触发**，只有 dispatch_done 后的 tail 路径执行。后果：文档主推的「满 128 主路径 GEMM」在当前唯一测试里得不到验证。需要（见决策 D3）：

```text
要么扩大测试规模（num_tokens / experts 配比）覆盖 128 主路径；
要么明确 128 是生产配置，小规模测试只验证 tail 闭环 + 精度对齐。
```

## Dispatch 全局结束语义

当前代码中 dispatch 全局结束由以下条件表示：

```cpp
dispatch_done_count == expected_dispatch_done_count
```

其中：

```cpp
expected_dispatch_done_count = num_logical_channels * NUM_MAX_NVL_PEERS;
```

语义是：所有 logical channel 的所有 NVL receiver warp 均已完成 dispatch。此后：

```text
expert_recv_count 不会再增长；
recv_tokens 不会再新增；
recv_token_source_info 已稳定；
可以安全 flush 所有 expert 的 tail batch。
```

这个信号不依赖 compute，因此不会被 compute 卡住。

## Combine Head-of-Line Blocking 风险

第一版使用全局 dispatch_done 后 flush tail，正确性简单，但可能带来性能风险：

```text
combine 按发射顺序等待 combine_token_ready；
如果较早 token 属于冷门 expert，且该 expert 长时间凑不满 128，
combine 可能被该 token head-of-line block。
```

第一版接受这个风险，优先验证真实 compute 闭环。后续优化可引入 per-logical-channel tail flush。

## 第二版：Per-Logical-Channel Tail Flush

更理想的策略是：某个 logical channel 的 dispatch 完成后，立即 flush 该 logical channel 内每个 expert 的 tail batch。

这更贴合 combine 的发射节奏，可以减少 head-of-line blocking。

但当前 `expert_recv_count[expert]` 是跨 logical channel 累加的，expert-local token 队列没有 channel range 信息。因此不能直接安全地按 channel flush。

第二版需要新增元信息，例如：

```text
expert_lch_start[expert, lch]
expert_lch_end[expert, lch]
```

或让 task descriptor 携带：

```text
(expert_id, logical_channel_id, start_slot, count)
```

这样才能正确 flush 某个 logical channel 内的 expert tail。

### Logical Channel 与物理 Channel 的映射（补充）

```text
物理 dispatch channel 数 = num_dispatch_sms / 2（even/odd SM 配对），测试中 = 12；
num_logical_channels 可多于物理 channel 数，由物理 channel round-robin 处理：
  for (lch = channel_id; lch < num_logical_channels; lch += num_channels)  // megakernel.cu:444
expert_recv_count[expert] 是跨 logical channel 累加的（megakernel.cu:1082），
  expert-local token 队列不带 channel range 信息，
  这正是第二版 per-channel flush 需要新增元信息的原因。
```

## Buffer 设计

推荐第一版 buffer 数据流：

```text
recv_tokens:
  expert-local input buffer
  dispatch 写入，compute 按 expert batch 连续读取

combine_input:
  compact recv-token namespace output buffer
  compute 写最终 FFN 输出，combine 继续读取
```

整体路径：

```text
dispatch 写 recv_tokens 给 compute；
compute 读 recv_tokens，执行 FFN；
compute 根据 recv_token_idx 写 combine_input；
combine 等 combine_token_ready 后读 combine_input 并发射。
```

这样 combine 侧协议改动最小，仍然顺序读取 compact token namespace。

### ⚠ 现状与上述设计的所有权矛盾（重构必须解决，见决策 D2）

上面的「compute 写 combine_input」是目标态。现状是：

```text
combine_input 当前由 dispatch 的 NVL receiver 写入原始 token（megakernel.cu:1032）；
combine_x 直接指向 combine_input（megakernel.cu:2961）；
compute 不写任何输出，combine 发的是 dispatch 的原始输入，不是 FFN 输出；
另有一个已分配但未接线的 compute_output 缓冲（megakernel.cu:2761）。
```

重构时必须明确「谁是 combine 的最终来源」。两个可选方案见决策 D2：
- 方案 A：compute 把结果写入独立的 `compute_output`，并把 `combine_x` 改指向 `compute_output`。dispatch 写 combine_input 的逻辑保留为 compute 的输入来源（或改写 recv_tokens）。**推荐**：不让 compute 覆写 dispatch 的数据，输入/输出 buffer 分离，调试更清晰。
- 方案 B：compute 原地覆写 combine_input。省一块显存，但 compute 读输入和写输出共用一块 buffer，必须保证「先读完该 token 的输入再写输出」，多 expert reduce 时尤其危险。

无论哪种，dispatch 当前写 combine_input 的那段（含 token data、topk_weights、src_meta）要重新定位：如果 combine 不再直接吃它，这段应改为写 compute 的输入 buffer。

## 多本地 Expert 命中与 Local Reduce

一个 recv token 可能命中多个本地 expert。因此 compute 不能让多个 expert 并发覆盖同一个 `combine_input[recv_token_idx]`。

需要做 local reduce：

```text
多个 local expert contribution -> combine_input[recv_token_idx]
```

可选实现路径：

```text
第一版正确性优先：
  使用 scratch / 临时 output 保存 contribution，再 reduce 到 combine_input。

长期优化：
  expected == 1 的 token 直接 store 到 combine_input；
  expected > 1 的 token 走慢路径 reduce。
```

### ⚠ reduce 必须在发 ready 之前完成

现有 per-token-ready 计数（megakernel.cu:1274-1282）：每个 local expert 算完做 `atomicAdd(token_compute_done)`，`done==expected` 时发 `combine_token_ready`。重构后若多个 expert 各自直接 `store` 到 `compute_output[recv_token_idx]`（非累加），**会丢掉先到 expert 的贡献**。约束：

```text
expected > 1 的 token，多个 expert 的 contribution 必须累加（reduce），
  不能互相覆盖；
reduce 完成 + __threadfence_system() 之后，才允许发出该 token 的 combine_token_ready。
```

最后一个完成的 expert（`done==expected`）负责确保 reduce 结果已落盘并对 combine 可见。

## Compute -> Combine 信号

沿用 per-token-ready 信号：

```text
token_compute_expected[recv_token_idx]
token_compute_done[recv_token_idx]
combine_token_ready[recv_token_idx]
```

每个 local expert 算完某 token contribution 后：

```text
token_compute_done += 1
如果 done == expected:
    publish combine_token_ready = 1
```

combine 发射 token 前：

```text
wait combine_token_ready[token_idx]
```

这保证 combine 只发所有本地 expert compute/reduce 均完成的 token。

### ⚠ 新增正确性前提：先 fence output，再发 ready

现状下 compute 是 pure gate，combine 读的是 dispatch 写的 combine_input，所以 ready 信号不涉及 output 可见性。重构后 compute 要真正写 output，必须保证写出顺序：

```text
1. compute 写 compute_output[recv_token_idx]（含多 expert reduce）；
2. __threadfence_system()  // output 对 combine SM 全局可见；
3. st_release combine_token_ready[recv_token_idx] = 1。
```

否则 combine sender 可能读到尚未写完的 output。dispatch 端「先 expected 再 count、自旋保证 slot 先于 count 可见」的既有顺序（megakernel.cu:1040-1082）保持不变。

### ⚠ combine 等待带 timeout→trap，注意阈值

combine NVL sender 按 token 顺序自旋等待，超过 `NUM_TIMEOUT_CYCLES` 会 `trap()`（megakernel.cu:1708-1722）。重构后 compute 真正耗时，且存在 head-of-line blocking（见下节），必须确认：

```text
NUM_TIMEOUT_CYCLES >= 整个 dispatch + 最慢 expert compute 的总时长上界，
否则冷门 expert 凑不满 128、迟迟不 flush 时会误 trap。
```

## Consumer Group 内同步

> 注意：本节的 `group_barrier[consumer_id]` / `group_phase[consumer_id]` 仅在采用「3×32 SM consumer group」方案（决策 D1 选 group）时才需要。现有「单 SM/expert」方案下，一个 batch 的全套 GEMM 在单个 block 内由 warps 用 `device_gemm_bf16` 完成，阶段间用 `__syncthreads()` 即可，不需要跨 SM 的 global barrier。

一个 32-SM consumer group 处理一个 batch 时，阶段之间需要同步：

```text
W_gate/W_up GEMM 完成
-> SwiGLU
-> W_down GEMM
-> 写回 combine_input
-> publish ready
```

由于这是 persistent kernel，不能依赖 kernel launch 边界同步。需要为每个 consumer group 配轻量 global barrier：

```text
group_barrier[consumer_id]
group_phase[consumer_id]
```

确保 32 个 SM 都完成当前阶段后再进入下一阶段。

## Scratch Buffer

每个 consumer group 需要中间 scratch：

```text
gate_scratch[consumer, 128, intermediate]
up_scratch[consumer, 128, intermediate]
```

SwiGLU 后可覆盖 `up_scratch` 为 activation，供 `W_down` 使用。

对于 3 个 consumer group，BF16 scratch 开销大致为数 MB，第一版可以接受。

## 第一版落地范围

第一版建议只实现以下内容：

```text
3 个固定 compute consumer group；
每组 32 SM；
full batch = 128；
全局 dispatch_done 后 flush tail；
compute 输出写 combine_input；
combine 用 per-token-ready 等待；
不做 per-channel tail flush；
不做复杂动态调度。
```

## 后续优化方向

第一版跑通后，再逐步推进：

```text
per-logical-channel tail flush；
更细粒度 CTA-level work queue；
更高效的多 local expert reduce；
W_gate/W_up/SwiGLU/W_down fusion；
consumer group 内更高效 barrier；
减少 scratch buffer 和全局内存往返。
```

## 结论

第一版采用固定 3 个 compute consumer group、每组 32 SM、每个 batch 128 token、全局 dispatch_done 后 flush tail、per-token-ready 放行 combine 的方案是可行的。

该方案优先解决真实 compute 接入和协议正确性问题。性能上可能存在 combine head-of-line blocking，但可以通过后续 per-logical-channel tail flush 和更细粒度调度继续优化。

## 待决策点

下面是重构前需要你拍板的点。每条给了现状、选项和我的倾向，你决定后我再据此改代码/文档。

- **D1｜compute 的 SM 划分模型**
  - 现状：单 SM 处理一个 expert，experts 在 100 个 compute SM 上 round-robin（megakernel.cu:1192）。`experts_per_rank=8` 时只有 8 个 SM 在算，其余空转。
  - 选项 A：保持「单 SM/expert」，一个 block 内 warps 跑完整 batch 的 GEMM。改动小，无需 group barrier。冷门/热门 expert 负载不均，大 batch 时单 SM 算力不足。
  - 选项 B：改成「N 个 SM 协作一个 expert batch」（文档原设想 3×32），按 N 维 tile 切分。算力利用率高，但需要跨 SM global barrier + scratch 布局重构，复杂度高。
  - 倾向：第一版先 A（正确性闭环优先），B 列入后续优化。

- **D2｜combine 的最终数据来源 / buffer 所有权**
  - 现状：combine_input 由 dispatch 写原始 token，combine_x 指向它；compute 不写输出（megakernel.cu:1032/2961）。
  - 选项 A：compute 写独立 `compute_output`（已分配），combine_x 改指向它；dispatch 那段改为写 compute 的输入 buffer。输入输出分离，调试清晰，多占一块显存。
  - 选项 B：compute 原地覆写 combine_input。省显存，但读写同一 buffer，多 expert reduce 风险高。
  - 倾向：A。

- **D3｜测试规模是否覆盖 128 主路径**
  - 现状：测试 num_tokens=16，永远走 tail，128 主路径未被验证。
  - 选项 A：新增大规模测试用例覆盖主路径。
  - 选项 B：维持小规模，文档声明 128 为生产配置，小测试只验证 tail + 精度。
  - 倾向：先 B 跑通闭环，再补 A。

- **D4｜route weight 的应用位置**
  - baseline 在 SwiGLU 后、W_down 前乘权重并对 topk slot 求和（test_megakernel_v7.py:100-101）。
  - 需确认：megakernel 在同一位置乘、且 combine 端不二次加权。
  - 倾向：严格对齐 baseline（SwiGLU 后、W_down 前），combine 端不加权。这条若你无异议，我按倾向实现，不必单独决策。

- **D5｜reduce 实现方式**（依赖 D2）
  - expected>1 的 token 多 expert contribution 必须累加。
  - 选项 A：统一走 atomic accumulate 到输出 buffer。简单，BF16 atomic 精度/性能需验证。
  - 选项 B：scratch 暂存 + 最后 reduce；expected==1 快路径直接 store。
  - 倾向：第一版 B（正确性优先，且避免 BF16 atomic 精度问题）。

我的建议组合：**D1=A, D2=A, D3=B(先), D4=对齐baseline, D5=B**。这是「正确性闭环最快、改动最小」的一版。你确认或调整后，我开始重构 `compute_worker`。
