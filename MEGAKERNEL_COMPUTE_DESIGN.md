# Megakernel Compute Design Notes

## 背景

当前 MK-v7 megakernel 已经把 dispatch、compute 和 combine 融合到同一个 persistent CUDA kernel 中，并通过 logical channel 做通信流水。compute 已进入 dispatch -> compute -> combine 的真实 FFN 数据路径。

本设计文档总结当前 compute 接入方案：优先保证真实 compute 路径闭环、信号正确和 combine 可按 token ready 推进；compute 调度已从按 expert 静态轮询升级为 scheduler SM + 动态 task queue。

## 当前状态：通信与 compute 已接入

当前 `megakernel.cu` 的实际状态（区别于本文档早期设想，重构前必须对齐认知）：

```text
dispatch  : 已对齐。NVL receiver 直接把原始 token 写入 combine_input（megakernel.cu:1032），
            并在 topk 循环前一次性算好 token_compute_expected，再自旋保证 slot 数据先于
            expert_recv_count 可见（megakernel.cu:1040-1082）。
combine   : 已对齐。combine_x 直接指向 compute_output，NVL sender 按 token 顺序
            自旋等待 combine_token_ready，带 timeout→trap。
compute   : 已接入真实 FFN。1 个 scheduler SM 扫描 expert_recv_count 并生成 ComputeTask；
            3 个 32-SM consumer group 通过 CAS 动态 pop task，任意 group 都可处理任意
            expert batch。满 128 token 走 M=128 batched GEMM，dispatch_done 后 flush tail；
            多 local expert contribution 先 atomicAdd 到 compute_output_f，最后转 bf16
            到 compute_output 并发 per-token-ready。
```

关键事实：**combine 当前发的是 compute_output 中的 FFN 输出**。dispatch 写 combine_input 作为 compute 输入，compute 写 compute_output 作为 combine 输入，两者已分离。

本文档「## SM 划分」「## Compute Batch 粒度」「## Compute Task Queue」「## Consumer Group 内同步」等章节描述的 `scheduler + 3×32 SM consumer group` 已作为当前 compute 路线落地。

## 目标

把真实 MoE FFN compute 接入 megakernel：

```text
dispatch -> compute -> combine
```

每个 token 的本地 expert 计算完成后，通过 per-token-ready 信号放行 combine，使 combine 不必等待所有 expert 或所有 compute 完成。

## SM 划分

> 当前实现采用 `1 scheduler SM + 3×32 consumer group`：先从 reserved 中抽 1 个 SM 做 scheduler，`compute = floor((total - dispatch - combine - scheduler) / 32) * 32`。测试默认 `148 - 24 - 24 - 1 = 99`，因此启用 96 个 compute SM，剩余 3 个 SM reserved。

硬件总 SM 数：

```text
total SM = 148
```

当前划分：

```text
dispatch  = 24 SM
combine   = 24 SM
scheduler = 1 SM
compute   = 96 SM = 3 个 consumer group * 32 SM
reserved  = 3 SM
active    = 145 SM
```

其中 scheduler SM 只负责入队 compute task；每个 compute consumer group 固定 32 个 SM，一次处理一个 expert 的一个 batch。consumer group 不再绑定 expert，空闲 group 可动态领取任意 expert batch。

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

当前代码按 `COMPUTE_BATCH_SIZE=128` 聚合/flush work unit，并用 `M=batch_size` 的 batched GEMM 执行 FFN。full batch 为 `M=128`；tail batch 复用同一路径，workspace 固定预留 128 行并将 `[batch_size,128)` 行清零，避免 WMMA `load_matrix_sync` 在 M 维 tile 读取时越界。

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

当前已引入 compute task queue。队列元素是 batch descriptor：

```cpp
struct ComputeTask {
    int expert_id;
    int start_slot;
    int num_tokens;          // 128 或 tail < 128
};
```

当前入队规则：

```text
正常阶段：
  expert_recv_count[expert] - enqueue_cursor[expert] >= 128
  -> 入队一个 128-token full batch

收尾阶段：
  dispatch 全局结束后
  expert_recv_count[expert] - enqueue_cursor[expert] > 0
  -> 入队一个 tail batch
```

`enqueue_cursor[expert]` 记录该 expert 已经入队到哪个 expert-local slot，避免重复入队。scheduler 是唯一 producer，通过 `compute_task_tail` 顺序发布任务；compute groups 是多个 consumer，通过 CAS 推进 `compute_task_head` 领取任务。group 内只有 `group_sm_idx==0 && threadIdx.x==0` pop task，并通过 `compute_group_task_idx[group_id]` 广播给同组 32 个 SM；`-1` 表示暂时无任务，`-2` 表示 scheduler 已完成且队列为空，整组统一退出，避免 group barrier 分歧。

## 不足 128 Token 的处理

不要在正常 dispatch 过程中把不足 128 的 batch 直接塞进队列，否则会打碎主路径，降低 TensorCore 利用率。

第一版策略：

```text
满 128 -> 立即算
不满 128 -> 等当前 dispatch round 完成后，一次性全 flush
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

当前实现按 dispatch round flush tail，而不是等全局 dispatch_done；每个 round 对应一组 `num_dispatch_channels` 个 logical_channel，round 内全部结束后统一 flush：

```text
combine 按发射顺序等待 combine_token_ready；
如果较早 token 属于冷门 expert，且该 expert 长时间凑不满 128，
combine 可能被该 token head-of-line block。
```

当前实现接受按 round flush 的粒度风险，优先验证真实 compute 闭环。后续若要进一步前移 tail flush，可再引入 per-logical-channel tail flush。

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

当前 compute 真正写 output，ready 信号必须保证 output 可见性。写出顺序必须保持：

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

> 当前实现已采用「3×32 SM consumer group」方案。每个 consumer group 配 `compute_group_barrier[group_id]` / `compute_group_phase[group_id]`，阶段间通过 reusable global barrier 同步 32 个 SM。

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

当前已实现以下内容：

```text
1 个 scheduler SM；
3 个动态 compute consumer group；
每组 32 SM；
full batch = 128；
全局 dispatch_done 后 flush tail；
compute 输出写 compute_output；
combine 用 per-token-ready 等待；
不做 per-channel tail flush。
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

当前采用 1 个 scheduler SM、3 个动态 compute consumer group、每组 32 SM、每个 batch 128 token、按 dispatch round flush tail、per-token-ready 放行 combine 的方案。

该方案优先解决真实 compute 接入和协议正确性问题，并避免静态 expert/group 绑定导致的负载不均。性能上仍可能存在 combine head-of-line blocking，但可以通过后续 per-logical-channel tail flush 和更细粒度调度继续优化。

## 待决策点

下面是重构前需要你拍板的点。每条给了现状、选项和我的倾向，你决定后我再据此改代码/文档。

- **D1｜compute 的 SM 划分模型**
  - 当前实现：采用选项 B，固定 32 个 SM 组成一个 compute consumer group，并从 reserved 中抽 1 个 SM 做 scheduler。默认 `total=148, dispatch=24, combine=24, scheduler=1` 时启用 3 个 group / 96 个 compute SM，剩余 3 个 SM reserved。
  - 每个 group 共享一份 batch workspace，GEMM tile 由 `group_warp_id` 在 32 个 SM 的所有 warp 上切分；阶段间用 `compute_group_barrier/phase` 做 global barrier。group 通过 task queue 动态领取任意 expert batch，不再按 local expert 静态 round-robin。

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
  - 当前实现：dispatch 先发布最终 `token_compute_expected`，再发布 expert slot；compute 中 `expected==1` 直接 store 到 `compute_output`，`expected>1` 走 `compute_output_f` float atomic reduce，最后一个 expert 转 bf16 并发 ready。该实现避免 BF16 atomic 精度问题，同时保留单 expert 快路径。

当前落地组合：**D1=B + scheduler/task queue, D2=A, D3=B(先), D4=对齐baseline, D5=B**。compute 已从单 SM/expert 升级为 scheduler 动态分发、32-SM consumer group 协作一个 expert batch。

---

# 附录：Compute 计算部分性能优化（B300，借鉴 SonicMoE）

> 本章是独立于上面「接入/协议正确性」的另一个维度：**在协议已闭环的前提下，如何把 compute 的 GEMM/SwiGLU
> 计算部分在 B300（Blackwell，SM100）上做到性能最优**，主要借鉴 SonicMoE 的 IO/Tile-aware 思路。
> 适用范围：推理 forward-only。dispatch / combine 不在本章范围。

参考实现：SonicMoE（`../sonic-moe/`），论文 `SonicMOE.pdf` + 博客 `assets/2026-04-22-sonicmoe-blackwell.md`。
本仓相关代码：`device_gemm_bf16`（megakernel.cu:287-335）、`compute_worker`（megakernel.cu:1370-1575）。

## A. SonicMoE 对 MoE 计算的优化点

核心判断：**细粒度 + 高稀疏 MoE 已进入 memory-bound regime**（算术强度低），瓶颈是 IO 与调度，不是算力。

### A.1 算法层：消除 O(TKd) 中间张量（不适用 megakernel）
- 反向重排 `dS_{t,e}=⟨dO_t, A_e W_2⟩=⟨dA'_{e,t}, A_{e,t}⟩`，避免缓存 `Y`/`dY`，激活显存与粒度无关。
- megakernel 是推理 forward-only，无 backward、无激活缓存，这层不复用。

### A.2 IO 层（forward 最该复用）
- **Gather fusion**：token gather 融进 GMEM→SMEM load（cp.async / TMA gather4），不写 gathered X 到 HBM。
- **L2 局部性**：从原始 `T×d` 张量 gather（比预聚合 `T×K×d` 小 K 倍），更易常驻 L2（B300 192MB），实测 HBM load 下降、L2 命中率 74.9% vs 66.3%。
- **SwiGLU in-register epilogue 融合**：MMA 结果在寄存器/TMEM 内直接算 SwiGLU，不落 HBM 再读回。
- **Expert aggregation = gather-and-sum**：GEMM 连续打包输出，聚合时每 token gather 自己激活的专家输出求和（比 scatter-fusion 在 Hopper 快 20%，B300 仍快 3%）。

### A.3 硬件/调度层（B300 直接相关）
- **2CTA MMA**：一对 CTA 协作一条 UMMA，M_tile 翻倍到 256；B（权重）tile 在 CTA pair 间 multicast 共享，**B-side SMEM/HBM 流量减半**。
- **UMMA(`tcgen05.mma`) + TMEM 双缓冲**：UMMA 单线程异步发射，结果进 TMEM；MMA warp 填一个 stage，epilogue warps 排空另一个 → MMA 与 epilogue IO 重叠（dH kernel epilogue IO +24%，TFLOPS 仅 -11%）。
- **CLC 动态 tile scheduler**：硬件管理工作队列，无 GMEM atomic，对专家 token 数不均天然均衡。
- **Producer-consumer mainloop + 可定制 epilogue（`epi_visit_subtile`）**：fusion 逻辑集中在 epilogue 注入点。

## B. megakernel 计算部分现状的性能问题

`compute_worker` 三段 GEMM 全部走 `device_gemm_bf16`，问题：
- 用 **WMMA legacy API（16×16×16）**，不是 Blackwell 的 `tcgen05.mma`/TMEM，TensorCore 利用率远未到峰值。
- **无 SMEM tiling / 无 cp.async 流水**：`load_matrix_sync` 每个 tile 直读 GMEM，A/B 重复从 GMEM 读，无 load/MMA 重叠。
- **中间张量全部落 GMEM**：gate_buf/up_buf/down_buf 在 GMEM workspace，SwiGLU 还要从 GMEM 读回算再写回。
- **三次 GEMM 间全局 barrier 串行**（每段后 `compute_group_sync`），无 MMA/epilogue 重叠。
- **32 SM 协作 128-token batch + 16×16 碎 tile**，调度与同步开销高。

## C. 重点议题：「重复 load 权重」是不是最大瓶颈？

### C.1 现象确实存在
`compute_worker`（megakernel.cu:1474-1506）每个 task 即时从 GMEM 读专家权重；`device_gemm_bf16`（megakernel.cu:314-317）K 维循环里 `wmma::load_matrix_sync(b_frag, ...)` 每个 tile 直读 GMEM 权重 B，**无 SMEM 缓存、无跨 batch 复用**。同专家被切成多个 task 时权重重复 load，不同专家轮流上场也无法常驻。

### C.2 用算术强度定量判断
单段 GEMM `[M,K]·[K,N]`：FLOPs ≈ `2·M·K·N`，权重加载字节 ≈ `2·K·N`（bf16），
**权重复用率 ≈ FLOPs/权重字节 = M（batch token 数）**。
- 当前 `COMPUTE_BATCH_SIZE=128` → 权重每加载一次仅被 128 token 复用。
- M=128 在 B300 上对 bf16 GEMM 仍偏小（平衡点约 300+），**确实偏 memory-bound，权重 IO 占比高**——担心成立。
- 若 M 能到 256/512，权重就不再是主瓶颈。

### C.3 更准确的根因排序
比"跨 batch 重复 load 权重"更致命的是：
1. **权重连 SMEM 都没进**：直读 GMEM，连同一 batch 内不同 16×16 tile 都重复读同一块权重（无 SMEM tiling）。
2. **WMMA 16×16×16 + 无流水**：load 无法与 MMA 重叠。
3. **中间张量落 GMEM 再读回**算 SwiGLU。

**结论：权重 IO 是瓶颈的一部分，但根因是"无 SMEM 缓存 + 无 load/compute 重叠 + tile 太小"，
不仅仅是跨 batch 重复 load。需 microbench/ncu 实测确认占比再定优先级。**

## D. 可复用的 SonicMoE 思路（映射到 megakernel）

| SonicMoE 思路 | 复用性 | 说明 |
|---|---|---|
| SwiGLU 融进 epilogue（in-register） | 高，必做 | 当前 gate/up 写 GMEM 再读回算 SwiGLU 是纯浪费；在 accumulator 还在寄存器/TMEM 时直接算 silu(gate)*up |
| GEMM1+GEMM2 合并 / 中间不落 GMEM | 高 | gate、up 同 input，可合并成 `[M,2I]` GEMM 或共享 input tile；act 直接喂 down-proj |
| UMMA(tcgen05) + TMEM 替换 WMMA | 最大算力收益（B300 核心） | 16×16×16 WMMA 在 B300 严重欠速；换 `tcgen05.mma`，accumulator 进 TMEM，M_tile 用 128/256 |
| 2CTA MMA（权重 multicast） | 适配 group 模型 | group 内多 SM 共享专家权重 tile，2CTA multicast 减半权重 HBM/SMEM 流量——直接缓解 C 节的权重重复 load |
| MMA/epilogue IO 重叠（TMEM 双缓冲） | 中高 | warp specialization：MMA warp 填一个 TMEM stage，epilogue warp 排空另一个并做 SwiGLU+写回，去掉 barrier 串行 |
| cp.async / TMA 流水进 SMEM | 必做 | 替换 `load_matrix_sync` 直读 GMEM，建 producer-consumer 双缓冲 mainloop |
| Gather fusion + L2 局部性 | 部分适用 | dispatch 已把 token 路由到 expert storage；可让 GEMM 直接按 source_info 从 `combine_input` 流式 gather 进 SMEM，省掉 input_buf 拷贝 |
| gather-and-sum 聚合（避免 scatter） | 已类似 | 当前多专家 reduce 用 float atomicAdd；如 profiling 显示 atomic 热点再改 gather-and-sum |
| CLC 动态 tile scheduler | 概念可借 | megakernel 已有软件任务队列做动态均衡，作用类似 CLC；GEMM 内部 tile 调度可换 persistent/CLC 风格减少同步 |
| 反向 dS/dH 重排、token rounding、激活显存优化 | 不适用 | forward-only，无 backward、无激活缓存 |

## E. 计算逻辑改动方案（按优先级）

### P0 — 重写 `device_gemm_bf16` 为 Blackwell GEMM（决定性能上限）
- 用 `tcgen05.mma`(UMMA) 替换 `nvcuda::wmma`，accumulator 落 TMEM。
- 用 cp.async / TMA 把 input 和专家权重 tile 流水进 SMEM，建 N-stage 双缓冲 mainloop（producer 加载 / consumer MMA）。
- tile 放大到 M=128（或 2CTA 的 256），N/K 取 64/128，取代碎片化 16×16。
- 现实建议：与其裸 CUDA 手写 UMMA，**优先在 compute_worker 内调用 SonicMoE 依赖的 QuACK / CUTLASS Blackwell grouped-GEMM device 接口**，复用其 mainloop+epilogue。megakernel 的 128 token/expert batch 天然就是 grouped-GEMM 的 M 分组。

### P1 — 三段 GEMM 融合 + 中间张量不落 GMEM
- GEMM1(gate) 与 GEMM2(up) 合并为单个 `X·[W_gate|W_up]` 输出 `[M,2I]`，或至少共享 input SMEM tile。
- **SwiGLU 放进 epilogue**：accumulator 在 TMEM/寄存器内直接算 `silu(gate)*up*route_w`（route weight 位置严格对齐 baseline：SwiGLU 后、W_down 前，见 D4），产出 act tile 直接作为 down-proj 输入 tile，删除 gate_buf/up_buf GMEM workspace 与对应往返。
- 删掉 input_buf 的 GMEM 拷贝，GEMM 直接按 `recv_token_source_info` gather 从 `combine_input` 进 SMEM（gather fusion）。

### P2 — MMA/epilogue 重叠 + 权重 multicast
- TMEM 双 stage + warp specialization 重叠 MMA 与 SwiGLU/写回，去掉每段 GEMM 后的全局 `compute_group_sync`。
- group 内多 SM 用 2CTA MMA 对同一专家权重做 multicast，减半权重 SMEM/HBM 流量。

### P3 — 调度与 reduce 收尾
- `COMPUTE_GROUP_SIZE=32` SM 协作 128 token 偏重；配合大 tile 后缩小 group（如 1-2 SM/batch 用 2CTA），减少跨 SM 同步。
- 适当提高 `COMPUTE_BATCH_SIZE`（提高 M / 权重复用率），或让同专家多 batch 连续处理（权重在 SMEM/L2 跨 task 复用）。
- 多专家 reduce 的 float atomicAdd 可保留；若 ncu 显示 atomic 热点再改 gather-and-sum。

## F. 验证计划
- **microbench**：单专家三段 GEMM `[128,hidden]→[128,2I]→SwiGLU→[128,hidden]`，对比改造前后 TFLOPS 与 HBM 流量。
- **ncu 指标**：Tensor Pipe util、DRAM throughput、L2 hit rate；单独拉出 `W_gate/up/down` 的 HBM read bytes，对比 input/激活 bytes，定量回答"权重 load 占总 IO 多少"。
- **端到端**：接回 megakernel，BF16 数值对齐用 precision-alignment 流程。
- 顺序建议：先 microbench 确认当前 `device_gemm_bf16` 是 memory-bound 且权重 IO 占比 → 再按 P0→P3 改造。

## G. 演进顺序与依赖链（分步实施，勿混在一个 patch）

这些优化**不是 5 个平行独立 patch**，有依赖关系。一次只改一个因素（对齐 AGENTS.md「最小变量」原则），每加一项都对一次数值基线，便于定位 diff。

```
[已完成] SwiGLU 融进 epilogue (WMMA 版) —— 定义数值正确性基线
    │
    ├─ cp.async/TMA 流水进 SMEM   ← 可独立做（WMMA 版上即可），不强依赖 UMMA
    ├─ GEMM1+GEMM2 合并成 [M,2I]  ← 正交，任意阶段可做（需同步改 workspace layout）
    │
    └─ P0: UMMA(tcgen05)+TMEM 替换 WMMA   ← 会重写 device_gemm_swiglu_fused
            ├─ 依赖它: 2CTA MMA 权重 multicast（WMMA 无 2CTA 概念）
            └─ 依赖它: MMA/epilogue TMEM 双缓冲重叠（依赖 TMEM）
```

- **当前 `device_gemm_swiglu_fused`（WMMA 版）的定位**：它定义了 fused gate+up + epilogue SwiGLU 的**计算语义和数值基线**。UMMA 版会重写底层 MMA，但保留这套骨架（同样在 epilogue 从 TMEM 排空时算 SwiGLU）。所以这版不是白做，是后续 UMMA 改造的 reference。
- **2CTA / TMEM 双缓冲依赖 UMMA**：必须先有 `tcgen05.mma`+TMEM 才能用 `cta_group::2` 和双 stage 重叠。
- **数值基线策略**：每一步改造后，新内核输出都要和上一个稳定版本在 BF16 容差内对齐，再叠加下一项。不要把 UMMA + 2CTA + 双缓冲混进一个改动。

### 已知待处理的不兼容点（workspace layout）
当前为不动显存分配，`gemm_workspace` 仍保留 `input/gate/up/down` 四段切分，其中 `gate` 段在 fusion 后已不再使用（代码里 `(void)gate_buf` 标记）。等做到「GEMM1+GEMM2 合并成 `[M,2I]`」或 UMMA 版时：
- `gate_stride` 段可彻底删除；
- 需同步改 `compute_worker` 的 stride 计算（megakernel.cu:1482-1491）和 host 端 `gemm_workspace` 分配大小。

## H. 变更记录

### 2026-06-26｜SwiGLU 融进 epilogue（WMMA 版，已完成，待编译验证）
- **改动文件**：`csrc/kernels/megakernel.cu`（仅此文件）。
- **新增** `device_gemm_swiglu_fused`（megakernel.cu:351）：融合 gate+up GEMM。
  - 每个 16×16 N-tile 同时累加 `cg_frag`(gate) 与 `cu_frag`(up)，**A tile 每个 K 步只 load 一次**被两个 MMA 共享。
  - accumulator 经 SMEM 暂存后在 epilogue 内直接算 `silu(gate)*up*route_w`，**只写一次 act** 到 GMEM。
  - per-warp SMEM 分 gate/up 两块（`2*WMMA_M*WMMA_N` floats）；padded 行写 0。
- **`compute_worker` 改造**：
  - 新增 `s_route_w[]`，加载 token 元信息时一次性 gather route weight 进 SMEM（megakernel.cu:1546）。
  - 用一次 `device_gemm_swiglu_fused` 替换原「gate GEMM + barrier + up GEMM + barrier + SwiGLU 循环 + barrier」（megakernel.cu:1575）。
- **route weight 位置不变**：SwiGLU 后、W_down 前，与 baseline 一致（D4）。
- **收益**：省掉 gate_buf/up_buf 两趟 GMEM 写 + SwiGLU 读回两趟 GMEM 读；去掉两个 `compute_group_sync` 全局 barrier；A tile 在 gate/up 间复用，减半 A 的 GMEM load。
- **SMEM 预算**：fused 每 SM 用量约 `25 warps * 2 * 256 * 4 = 51200 B`，小于现有 `smem_size = max(8*16384, 24*9248) = 221952 B`（deep_ep.cpp:2110），安全。
- **验证方式**：用户跑 `run_megakernel_v7_test.sh` 单测做 BF16 数值对齐（编译/执行由用户负责）。若出 diff，优先排查 `s_route_w` 的 gather 索引（megakernel.cu:1546）。
- **状态**：代码已写，已通过 `run_megakernel_v7_test.sh` 验证（用户确认），已 commit `fab6030` 推到 `origin/mega_train`。

### 2026-06-26｜setup.py 加入 SM100 arch flag（已完成）
- `TORCH_CUDA_ARCH_LIST` 默认从 `'9.0'` 改为 `'9.0;10.0'`（setup.py:74），为 UMMA(tcgen05)+TMEM 路径准备 SM100 编译目标。
- **副作用提醒**：setup.py:80 的判断「arch != '9.0' 时强制 `DISABLE_AGGRESSIVE_PTX_INSTRS=1`」现在会触发，禁用 `.L1::no_allocate` 等激进 PTX。不影响 UMMA，属保守行为，暂不处理。

## I. UMMA(tcgen05)+TMEM 改造：路线选择与风险清单（P0，待逐条深挖）

> 目标：把 compute 部分的 `device_gemm_bf16` / `device_gemm_swiglu_fused` 从 `nvcuda::wmma`（16×16×16）
> 换成 Blackwell UMMA（`tcgen05.mma`）+ TMEM accumulator，拿到 B300 的算力上限。
> 用户已确认：环境 OK、compute 是瓶颈、arch flag 已加。本节记录路线与风险，逐条深挖后再动手。

### I.0 调研结论（前置事实，已确认）
- **编译**：`setup.py` 用 `TORCH_CUDA_ARCH_LIST` 驱动，已加 `10.0`（setup.py:74）；`nvcc -rdc=true`。**未链接 CUTLASS/CuTe**，CUDA 自带 `include/cccl` 可能有部分 CuTe（setup.py:55-57）。
- **权重布局**：`W_gate/W_up = [E_local, intermediate, hidden]`，`W_down = [E_local, hidden, intermediate]`，连续 BF16，host 端无转置（megakernel.cu:181-183；test_megakernel_v7.py 用 `W_gate[e].T` 做参考）。
- **GEMM 调用点**：`compute_worker` 内两处——`device_gemm_swiglu_fused`(gate+up+SwiGLU) 和 `device_gemm_bf16`(down)（megakernel.cu:1575/1581）。
- **线程组织**：`COMPUTE_GROUP_SIZE=32` SM 协作一个 batch，`blockDim=kMegaKernelNumThreads=(24+1)*32=800`（25 warps/SM），`group_num_warps=800`，跨 SM 用全局 `compute_group_sync`（megakernel.cu:45-47, 1454-1475）。
- **现有低层原语**：仓库已有 TMA / `cp.async`（combine 路径 internode.cu）、`__threadfence` / `st_*_release` / `atomicCAS`；**无 tcgen05 / mbarrier(cluster) / wgmma / cute:: 的现成用法**。

### I.1 路线选择（待定，深挖后决策）
- **路线 A：裸 PTX 手写 `tcgen05.mma` + TMEM**
  - 优点：完全控制 TMEM 生命周期和跨 SM 调度，不迁就 CUTLASS 的 kernel 级假设，和 persistent 模型最搭；不引入新编译依赖。
  - 缺点：工作量大、易错，要自己写 `tcgen05.alloc/mma/ld/dealloc`、mbarrier 同步。
- **路线 B：引入 C++ CuTe/CUTLASS Blackwell GEMM**
  - 优点：2CTA / TMEM 双缓冲 / tile scheduler 现成（SonicMoE 同款思路）。
  - 缺点：与 persistent megakernel 集成有多项风险（见 I.2），且要把并行模型往 CUTLASS 的 CTA 模型靠。
- **路线 A'（折中）：用 CuTe 低层 atom（SM100 tiled mma + TMEM helper），kernel 编排自己写**
  - 拿到 tcgen05 封装又保留 persistent 控制权；要求对 CuTe 较熟。

### I.2 CuTe/CUTLASS 与 persistent megakernel 集成的风险清单（逐条深挖）

**风险 R1：CUTLASS GEMM 是 kernel 级实体，不是 device 函数**
- `cutlass::gemm::device::GemmUniversal` / `CollectiveMma` 设计成完整 `__global__`，自带 grid/tile scheduler/epilogue，**不能在 `compute_worker` 里当 `__device__` 函数直接调**。
- 能复用的是底层 CuTe 原语（`TiledMMA`、tcgen05 atom、`cute::copy` for TMA）+ collective mainloop，但要自己在 worker 内手工编排。**省 PTX 细节，不省 kernel 编排。**
- 深挖点：CUTLASS 4.x 里 collective mainloop 能否以纯 device 函数形式被复用？SonicMoE 在 QuACK 里是怎么把 collective 拆成可嵌入片段的？

**风险 R2：线程组织不匹配（最大风险）**
- 现状：32 SM × 800 线程协作一个 batch，跨 SM 全局 barrier。
- CUTLASS Blackwell：协作单位是 1 CTA（或 2CTA cluster），warp specialization（1 producer + 1 MMA + 多 epilogue）在**单 CTA 内**用 mbarrier 同步，不跨 SM。
- 两套并行模型对不上。可选：(a) 把 GEMM 单元从「32 SM 协作」缩成「1 CTA / 2CTA cluster 算一个 tile」，让 CUTLASS 的 CTA 内 warp specialization 自然工作（更对的方向，但要重写 compute group 调度）；(b) 强行让 32 SM 各跑 CUTLASS CTA 逻辑，则 `compute_group_sync` 与 mbarrier 语义打架。
- 深挖点：把 compute group 从 32 SM 改成 1-2 CTA/batch，对 task queue 调度、batch 划分、其余 SM 利用率的影响。

**风险 R3：TMEM 分配与 persistent 复用**
- TMEM(256KB/SM) 需 `tcgen05.alloc` 显式分配、`tcgen05.dealloc` 释放；CUTLASS 假设 kernel 退出自动回收。
- `compute_worker` 是 `while(true)` 反复处理 task：每 batch alloc/dealloc TMEM，还是循环外分配一次复用？弄错会泄漏或撞另一个 batch 的 TMEM。
- 深挖点：TMEM 在 persistent 循环里的正确生命周期管理；dispatch/combine SM 不用 TMEM，是否要按角色条件分配。

**风险 R4：寄存器/SMEM 预算挤压**
- megakernel `__launch_bounds__(800, 1)`，SMEM 已被 dispatch/combine 的 TMA buffer 占（`smem_size≈217KB`，deep_ep.cpp:2110）。
- CUTLASS Blackwell mainloop 自己要一大块 SMEM 做 A/B multistage + mbarrier。同 kernel 内 dispatch SM 和 compute SM 共享同一 `__launch_bounds__` 和动态 SMEM 上限（B300 上限约 228KB/SM）。
- 深挖点：compute SM 实际 SMEM 需求 vs combine 路径 217KB，会不会叠加超限；800 线程/block 下 UMMA 版的寄存器压力与 occupancy。

**风险 R5：编译复杂度**
- 引入 CUTLASS 要加 include（CUDA 自带 cccl 有部分，Blackwell GEMM 要完整 CUTLASS 4.x），`-rdc=true` + CuTe 模板 + nvshmem device link 三者一起编，编译时间和符号冲突风险上升。
- 深挖点：能否只 include CuTe 头（header-only）而不引入完整 CUTLASS；与现有 nvshmem device link 是否冲突。

### I.3 建议的低风险验证步骤（待用户确认是否执行）
- 先写一个**独立、单 CTA 的 tcgen05 BF16 GEMM microkernel** `[128,K]@[K,N]`，脱离 megakernel 单独编译跑通，确认：(1) sm_100 PTX 能编；(2) tcgen05+TMEM 数值正确；(3) 单 CTA 跑一个 tile 的 SMEM/TMEM 预算。
- 跑通后再决定裸 PTX(A) / CuTe(B) / 折中(A')，以及怎么把它嵌进 compute group（关联 R2/R3）。
- 全程以「已 commit 的 WMMA 版 fused SwiGLU」为数值基线。

### I.4 路线决策（已定）：A'（CuTe 低层 atom，1CTA 先行）

经讨论，采用**路线 A'**：复用 CuTe 的低层算子积木（L1-L5），kernel 编排自己写；不走完整 CUTLASS collective（L6/L7）。

**CuTe/CUTLASS 分层复用矩阵**（评估在 persistent megakernel 里的可复用性；类名已实读 cutlass_ref v4.5.2 核实）：

- **L1 数据类型/布局**（`cute::Tensor`/`Layout`/swizzle、`UMMA::tile_to_mma_shape` + `UMMA::Layout_K_SW128_Atom`）：✅ 几乎全复用，header-only。
- **L2 MMA atom**（`SM100_MMA_F16BF16_SS`(1CTA) / `SM100_MMA_F16BF16_2x1SM_SS`(2CTA)，经 `make_tiled_mma` → `TiledMMA`）：✅ 可复用——**最大价值，省掉最难写的 tcgen05 PTX**。**FP16/BF16 共用同一 atom**，`TypeA/B=bfloat16_t`、accumulator `float` 即可。
- **L3 Copy atom**（`cooperative_copy<128>` 直拷 GMEM→SMEM；或 TMA `make_tma_atom`+`tma_partition`；TMEM→reg 用 `make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, ...)`）：✅ 可复用。
- **L4 TMEM 管理**（`cute::TMEM::Allocator1Sm`(1CTA) / `Allocator2Sm`(2CTA)，`allocate(Sm100TmemCapacityColumns,&base)` / `free` / `release_allocation_lock`；accumulator 由 `cta_mma.make_fragment_C(tCgC)` 生成 TMEM tensor，`tCtAcc.data()=base` 绑地址）：✅ 可复用，**1CTA 下生命周期清晰（配对 allocate/free），难点在阶段 4 persistent 循环复用**（R3）。
- **L5 mbarrier/同步**（`cute::initialize_barrier` / `umma_arrive` / `wait_barrier` + phase_bit；2CTA 加 `cluster_sync`）：⚠️ 可复用但需适配 persistent `while` 循环。
- **L6 Collective Mainloop**（`CollectiveMma`+`CollectiveEpilogue`）：❌ 不用——接口假设 Params/tile scheduler/特定 SMEM 布局，与 persistent + gather + 混布角色冲突（R1）。
- **L7 Kernel/Tile Scheduler**（`GemmUniversal`/CLC，见 `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`、`sm100_tile_scheduler.hpp`）：❌ 用不上——megakernel 有自己的 task queue + 角色分派（R2）。

**复用量估计**：路线 A' 复用 L1-L5 约 70%，kernel 编排/调度/gather+SwiGLU epilogue/TMEM 生命周期自己写。

**关键参考蓝本**：`cutlass_ref/examples/cute/tutorial/blackwell/01~05_*.cu`——裸 CuTe（不用 L6/L7），是渐进式手写 GEMM，正好对应我们的分阶段计划（见 I.4b）。

### I.4b CuTe Blackwell tutorial 五级演进表（实施直接蓝本）

`examples/cute/tutorial/blackwell/` 5 个文件从简到繁，每级在前一级上叠加，与我们 (2a)→(3) 精确对应：

- **01_mma_sm100.cu** —— tcgen05.mma + TMEM 最简 GEMM。`cooperative_copy<128>` 直拷 GMEM→SMEM（无 TMA），`Allocator1Sm`，`gemm()` 累加进 TMEM，epilogue 用 `make_tmem_copy`(tcgen05.ld 读回 reg) + `axpby` 写回。`cluster_shape=(1,1,1)` 等价不开 cluster。→ **(2a) 直接照它改 BF16**。
- **02_mma_tma_sm100.cu** —— 用 TMA 替换 cooperative_copy：`make_tma_atom`/`tma_partition`/`set_barrier_transaction_bytes`/`copy(tma_atom.with(barrier),...)`，双 barrier(tma+mma)。→ (2b) 后可选加速。
- **03_mma_tma_multicast_sm100.cu** —— TMA multicast。
- **04_mma_tma_2sm_sm100.cu** —— **2SM MMA + 2SM multicast TMA**：`SM100_MMA_F16BF16_2x1SM_SS`、`Allocator2Sm`、`block_rank_in_cluster()`、`elect_one_cta`(leader 执行 MMA / peer 只 multicast 数据)、`create_tma_multicast_mask`、`cluster_sync()`、`initialize_barrier(...,num_ctas=num_mcast_participants)`。→ **(3) 2CTA 照它**（这就是 SonicMoE leader/peer+relay 的 CUTLASS 原生版，不用自己发明）。
- **05_mma_tma_epi_sm100.cu** —— 输出也走 TMA epilogue。→ 后期可选。

**SwiGLU 插入点**：01 的 epilogue `axpby(alpha, tDrAcc, beta, tDrC)` 处——accumulator 已在寄存器(`tDrAcc`)，把它换成 `silu(gate)*up*route_w` 即 (2b)。gate/up 两个 accumulator 可用两次 MMA 或更大 N 拼。

**编译约束（实读确认）**：
- tutorial 要求 `mma_tiler 整除 ProblemShape`（"OOB accesses are not supported"）。我们 batch=128、hidden/intermediate=4096 都是 128 的倍数能整除；tail batch(<128) 需 padding 到 128（与现有 WMMA zero-pad 一致）。
- 01 用 `launch_kernel_on_cluster`+`ClusterLaunchParams`（dim=1）。独立 microkernel 无碍；**阶段 4 嵌入 megakernel 时，1CTA 路径应避免 cluster launch，2CTA 才面对 I.5 的 gridDim 整除**。
- arch flag：B30Z 是 **cc 10.3**，应为 `sm_103a`/`compute_103a`（之前 setup.py 写的 `10.0` 需在阶段 2a 编译时确认/修正）；SM100 atom 通用于 cc10.3。
- include 入口：`cute/tensor.hpp`、`cute/arch/tmem_allocator_sm100.hpp`、`cute/algorithm/cooperative_copy.hpp`、`cutlass/arch/barrier.h`、`cutlass/cluster_launch.hpp`。

### I.5 cluster_dim=2 风险校正（R2 相关，收窄）

> 之前把「cluster_dim=2 会改 dispatch/combine 行为 / 要拆 kernel」说重了，校正如下。

- **功能正确性不变**：cluster launch 只是把相邻 block 两两绑成 co-scheduled cluster，多出 `block_rank` 和 cluster-scope barrier/DSMEM。**`blockIdx.x`/`blockDim`/`__syncthreads`/SMEM/GMEM 语义全不变**。dispatch/combine 只要不调用 cluster API（`cluster.sync()`/DSMEM），逻辑感知不到 cluster，行为不变。
- **真实硬约束（躲不掉）**：`gridDim.x 必须被 cluster_dim 整除`。当前 `scheduler=1` 是奇数，总 block 数大概率非 2 的倍数，直接 cluster launch 会报错 → 需把 SM 布局对齐成偶数，且让 cluster 配对不跨 compute 边界。
- **次要影响**：给全 grid 开 cluster 会给 dispatch/combine 强加「2 block 同时驻留」约束，可能轻微影响调度，不改正确性。
- **dispatch/combine 的 even/odd SM 配对**（`sm_id%2`）与 cluster `block_rank` 是两套独立编号，只要不混用就不冲突。
- **结论**：2CTA 的真正待办收窄为「gridDim 整除 + SM 布局偶数对齐 + 验证 cluster 化 grid 下 dispatch/combine 驻留无退化」，比「拆 kernel」轻得多。**且这些只在阶段 3（2CTA）才面对，阶段 2（1CTA）完全不开 cluster。**

### I.6 实施流程（已定，分阶段）

```
(1)  [完成] clone NVIDIA/cutlass v4.5.2 → cutlass_ref/，产出 CuTe 分层复用矩阵(L1-L7)
     + 五级 tutorial 蓝本(I.4b)。确认 SM100 头齐全、Blackwell examples 可参照。
(2a) 独立 __global__ 单 CTA microkernel：基于 tutorial 01 改 BF16，[128,K]@[K,N] 纯 GEMM，
     脱离 megakernel 单独编译跑，对齐 torch/cuBLAS 参考，验证 编译(sm_103a)/数值/SMEM-TMEM-寄存器预算
(2b) 在 01 的 epilogue(axpby 处)接 gate+up+SwiGLU+route_w，改成 __device__ 形态，
     与 baseline（已 commit 的 WMMA fused 版）BF16 精度对齐
(3)  1CTA → 2CTA：照 tutorial 04（2x1SM atom + Allocator2Sm + multicast mask + cluster_sync），
     仍独立验证；处理 I.5 的 gridDim 整除 + SM 布局偶数对齐
(4)  嵌入 megakernel compute_worker：**把 compute 调度从 expert 级重构为 tile 级**（见 I.8），
     TMEM persistent 生命周期、SMEM 预算(与 combine 217KB 共处)、gridDim 整除
```

- 每阶段独立可验证、风险递增；全程以已 commit 的 WMMA fused SwiGLU 为数值基线。
- 关键洞察：UMMA 主要收益（TMEM/大 tile/warp specialization/load-MMA 重叠）在**阶段 2（1CTA）即可拿到**，2CTA 只是额外省权重 IO，是锦上添花、非前提。

### I.7 实施进展
- 2026-06-26：路线定为 A'（1CTA 先行）。`setup.py` 已加 SM100 arch（见 H 节，待按 cc10.3 确认 `sm_103a`）。
- 2026-06-26：(1) 完成——clone cutlass v4.5.2 到 `cutlass_ref/`（调研用，不应提交进仓库，建议加 .gitignore）。实读 tutorial 01 全文 + 02/04 要点，核实真实 CuTe API（已替换之前子 agent 的推测）；确认 5 级 tutorial 是 (2a)→(3) 的直接蓝本。
- 2026-06-26：**(2a) 完成（编译+数值通过）**。产物 `compute_ref/umma_gemm_1cta.cu`（独立 microkernel + WMMA 基线 + FP32 ref harness）。
  - 编译：`nvcc -std=c++17 -arch=sm_103a -O3 -I../cutlass_ref/include -I../cutlass_ref/tools/util/include --expt-relaxed-constexpr`，通过（只修了一个 `TypeA`→`OutType` 笔误）。**sm_103a 可用，CuTe SM100 atom 在 B30Z cc10.3 上能编。**
  - 数值：M=128,K=4096,N=4096，UMMA 与 WMMA 相对 FP32 参考误差均 = 0.00166（完全一致）→ PASS。**tcgen05+TMEM 数值正确，路线 A' 可行性确认。**
  - 性能：UMMA 2.17 TFLOPS(1.98ms) vs WMMA 6.39 TFLOPS(0.67ms)，**UMMA 慢 3x**。
  - **慢的根因（预期，非 bug）**：① M=128 只有 1 个 M-tile，N=4096 切 16 个 N-tile → 仅 16 个 CTA 干活，B30Z 上百 SM 大量闲置；② tutorial 01 是最简版（无 TMA、无 pipeline、无 double buffer），load/MMA 串行；③ 单 CTA 内 800 线程对单线程异步的 UMMA 是浪费。WMMA 版按 megakernel 方式 32 block×25 warp 各啃 16×16 小 tile，小 M 下并行度反而更高。
  - **结论**：2a 验证目标（编译+数值）达成。性能慢印证了核心判断——**UMMA 的优势要靠 tile 级多 CTA 并行 + TMA 流水 + fused epilogue + 2CTA 权重 multicast 才显现，裸 1CTA-01 在 M=128 瘦长 shape 下拿不到**。这直接催生 I.8 的 tile 级调度方向。

### I.8 Stage 4 架构方向：compute 调度从 expert 级重构为 tile 级（R2 的彻底解法）

**洞察来源**：2a 实测 UMMA 慢 3x 的根因是「expert 级调度」与「UMMA tile 级计算」粒度不匹配——32 SM 绑死协作一个 expert batch，无法把 batch 内的 tile 摊到所有 SM。

**核心改动**：把 compute 调度单位从「一个 expert 的 128-token batch」降到「(expert_id, m_tile, n_tile) 三元组」，做一个**全局 tile 工作队列**，所有 compute SM 各自从队列领 tile，每个 CTA（或 2CTA cluster）用 UMMA 算一个 output tile。

这等价于自己实现一个 grouped-GEMM 的 persistent tile scheduler（CUTLASS `sm100_tile_scheduler` / CLC、`75_blackwell_grouped_gemm` 的思路），但保留 megakernel 的软件队列形态。

**收益**：
- SM 利用率：所有 tile 摊到所有 SM，CLC 式全活跃（解决 2a 闲置问题）。
- **直接吃掉 R2**：降到 tile 级后「32 SM 协作」消失，每 CTA 独立算 tile，TMEM per-SM 自洽，`compute_group_sync` 全局 barrier 不再需要。
- 负载均衡：tile 是最小粒度，冷门 expert 不再拖累。
- 权重复用：同 expert 的 tile 连续调度，权重驻留 L2/SMEM。

**待设计的挑战（下一轮专门深挖）**：
1. **M 维太小的现实约束**：M=128 在 M 方向只 1 个 tile，并行度主要来自 N 维(16 tile) + **多 expert 的 tile 同时在队列**。tile 级威力在「多 expert、每 expert token 不满」的真实 MoE 场景才完全发挥。
2. **K 维 reduction 归属**：单 tile 的 K=4096 在 CTA 内 mainloop 累加即可。**先不 split-K**（避免跨 CTA reduce）。
3. **三段 GEMM 的 tile 依赖（最需设计）**：down GEMM 的 tile 依赖 gate/up 的哪些 tile 算完——需要 tile 级依赖管理，或保持「一个 CTA 串行算完某 token 块的三段」。这是 fused（我们当前）vs SonicMoE 8-kernel 拆法的权衡点。
4. **gather 落点**：token gather 从 batch 粒度移到 tile 的 M 方向 load 时按 source_info gather（对应 SonicMoE gather fusion）。

**定位**：这是 stage 4 的目标形态，不是立刻做。先完成 2b（fused SwiGLU epilogue，小而确定），再单独花一轮把 tile 级调度（尤其挑战 3 的三段 GEMM tile 依赖）设计清楚再动手。

- 下一步：(2b) 在 1CTA microkernel 的 epilogue 加 fused SwiGLU，与 WMMA fused 版数值对齐。
- 2026-06-26：**(2b) 完成（编译+数值通过）**。产物 `compute_ref/umma_gemm_swiglu_1cta.cu`。
  - 实现 `act = silu(A@Wg^T)*(A@Wu^T)*route_w`，gate/up 在 tcgen05+TMEM 上算，SwiGLU 在 epilogue（RMEM）内融合，只写 act 到 GMEM。
  - 数值：M=128,K=4096,N=4096，相对 host fused-SwiGLU 参考（复现 WMMA fused 版语义，bf16-round 输入 fp32 累加）误差 = **2.3e-4** → PASS。**fused gate+up+SwiGLU epilogue 在 TMEM 上正确，路线 A' 完整验证可行。**
  - 实现踩坑（已解决，记录备查）：
    1. **route_w 坐标映射**：epilogue 按全局行 m 取 route_w[m]，用 `make_identity_tensor(shape(mD))` → `local_tile` → `cta_mma.partition_C` → `thr_t2r.partition_D` 拿到与 accumulator 同构的 (m,n) 坐标。
    2. **双 TMEM accumulator misaligned**：手动算 TMEM 列偏移（2D DP×COL 编码）触发 misaligned address。改为**单 accumulator 复用**：gate 跑完 K-mainloop 排空到 RMEM，清零再跑 up 排空到 RMEM，两者在 RMEM 做 SwiGLU。
  - 代价：gate/up 串行两趟 mainloop、A 加载两次。性能优化留后。
- 下一步：(3) 1CTA → 2CTA，照 tutorial 04（`SM100_MMA_F16BF16_2x1SM_SS` + `Allocator2Sm` + multicast mask + cluster_sync），独立验证。
- 2026-06-26：**(3) 纯 GEMM 2CTA 完成（编译+数值通过）**。产物 `compute_ref/umma_gemm_2cta.cu`。
  - 照 tutorial 04 实现 2x1SM tcgen05.mma + 2SM multicast TMA。M_tile=256（用户确认 COMPUTE_BATCH_SIZE 可调到 256，满载不 padding）。
  - 数值：M=256,K=4096,N=4096，相对 FP32 参考误差 = **1.66e-3** → PASS（与 1CTA 同量级）。
  - 验证的 2CTA 机制（全部可用）：`SM100_MMA_F16BF16_2x1SM_SS<...256,256...>`、`Allocator2Sm`、`make_tma_atom_A/B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{})`、`cluster_shape=(2,1,1)` + `launch_kernel_on_cluster`、`block_rank_in_cluster`/`elect_one_cta`(leader 发 MMA / 两 CTA 都发 TMA)、`create_tma_multicast_mask`、`cluster_sync`、`umma_arrive_multicast_2x1SM`、TMA/MMA 双 barrier。
  - **关键收益机制确认**：B(权重) 经 `SM100_TMA_2SM_LOAD_MULTICAST` 在 cluster 2 CTA 间 multicast 共享 → 这就是省权重 IO 的来源。
  - **重要前提（已与用户确认）**：batch 能稳定凑满 256，故 M_tile=256 不浪费。megakernel 接入时需把 `COMPUTE_BATCH_SIZE` 从 128 调到 256（连带：tail 凑满更慢、workspace/SMEM 翻倍，见 I.8 接入注意）。
- 下一步：(3') 2CTA + fused SwiGLU，与 WMMA fused 版对精度 + 性能。
- 2026-06-26：**(3') 2CTA + fused SwiGLU 完成（数值 PASS + 性能反转）**。产物 `compute_ref/umma_swiglu_2cta.cu`。
  - 把 2b 的 gate/up 双 pass + SwiGLU epilogue 搬进 04 的 2CTA 框架；A 走 multicast TMA，Wg/Wu 各一份 multicast TMA，gate/up 两 pass 复用单 TMEM acc，SwiGLU 在 RMEM epilogue 内做。
  - 数值：M=256,K=4096,N=4096，2CTA-UMMA 与 WMMA 相对 fused-SwiGLU 参考误差均 = **1.99e-4** → PASS。
  - 性能：**2CTA-UMMA 0.241ms / 71.4 TFLOPS vs WMMA 1.97ms / 8.7 TFLOPS → UMMA 快 8.2x**（TFLOPS 按 gate+up 两 GEMM = 2*2*M*N*K 计）。
  - **从 2a 慢 3x 反转为快 8.2x 的原因**（印证全部判断）：① M=128→256 满载 2x1SM 不 padding、权重复用率翻倍；② N=4096 在 2CTA 下并行度上来；③ `SM100_TMA_2SM_LOAD_MULTICAST` 让 B(权重)在 cluster 2 CTA 间共享，命中"权重重复 load"痛点；④ fused SwiGLU 在 TMEM epilogue 内做，gate/up 不落 GMEM。
  - **诚实局限**：8.2x 是独立单 tile microkernel 对比（都算一个 M=256 tile）。端到端收益须 stage 4 嵌入 megakernel 后用 `run_megakernel_v7_test.sh` 测整个 compute 阶段才算数（含 tile 级调度、TMEM persistent 生命周期、gather、与 dispatch/combine SM 共享）。但已证明计算内核本身的巨大性能空间。
- **路线 A' 独立 microkernel 阶段全部通过**（2a/2b/3/3'）。下一步：stage 4 / I.8 —— 设计 tile 级调度，把 2CTA fused SwiGLU 内核嵌进 megakernel。

## I.9 Stage 4 实现规格：group 内 tile 级调度（已与用户对齐，待实现）

> 这是 stage 4 的正式实现规格，由多轮讨论收敛而成。核心思想：**保留现有 32-SM group 调度框架，
> 只把 group 内部从「32 SM 协作一个 batch」改成「32 SM = 16 个 2-CTA cluster 并行算 16 个 tile」**。
> 改动集中在 group 内 worker 逻辑，调度框架（scheduler + task queue）几乎不动，风险最低。

### I.9.0 固定前提（用户确认）
- 测试配置 **hidden = intermediate = 4096**，故 up-proj 的 16 个 i-tile 与 down-proj 的 16 个 d-tile **16↔16 完美对应**。先基于此版本开发。
- **`COMPUTE_BATCH_SIZE` 从 128 调到 256**，batch 能稳定凑满 256（用户确认），故 2x1SM 的 M_tile=256 满载不 padding。
- act 中间结果**落 group GMEM workspace**，靠 **L2 常驻**省 hardware IO（每 batch act = [256,4096] ≈ 2MB，B300 L2 192MB 装得下）。先用此方案试速度，不上 distributed-SMEM。

### I.9.1 三个权重的 tile 依赖关系（设计基础）
```
gate = X@Wg^T  [M,d]@[I,d]→[M,I]   K=d, N=I, 沿 I 切 tile
up   = X@Wu^T  [M,d]@[I,d]→[M,I]   K=d, N=I, 沿 I 切 tile
act  = silu(gate)*up*route         element-wise（不跨 tile）
out  = act@Wd^T [M,I]@[d,I]→[M,d]  K=I, N=d, 沿 d 切 tile，K=I 全规约
```
- **gate↔up**：同位置（同 i-tile）配对，互不依赖；SwiGLU 需同位置 gate+up 都到位 → **打包进一个 up-proj task 内部消化**（cluster 内 2 次 MMA + SwiGLU，X 只 load 一次）。
- **act→down**：down 的**每个** d-tile 的 K=I 要规约 act 的**全部 16 个 i 段** → **down 依赖该 m_block 的全部 16 个 up-proj tile 完成**（barrier 依赖）。
- **down tile 之间**：互不依赖，可并行。
- **关键纠正**：SwiGLU 是 element-wise，**随每个 up-proj tile 就地完成**，不是 16 个 tile 算完后单独的阶段；真正需要"等 16 个齐"的是 **down**（因 K=I 全规约），不是 SwiGLU。

### I.9.2 调度模型（保留 32-SM group，内部拆 16 cluster）
- **scheduler 不变**：仍发 batch 粒度 task（一个 256-token M-block），`compute_task_head/tail` CAS、`compute_group_task_idx` 广播全保留。
- **`COMPUTE_GROUP_SIZE=32` 不变**，但 group 内 32 个 block 在 `cluster_dim=2` 下自动配成 **16 个 2-CTA cluster**。
- **up→down 依赖门 = 现有 `compute_group_sync`**（32 SM 全局 barrier）：16 个 up tile 同 group，barrier 一下就齐，**无需额外的 `up_tiles_done` 计数器**——这是此方案最干净的地方。
- 16 个 i-tile 计算量相同（各 [256,256] gate+up+SwiGLU），负载天然均衡，barrier 等待倾斜小。

### I.9.3 group 内 worker 流程
```
group task = batch (256, 4096)              ← 调度不变
group 内 32 SM = 16 个 2-CTA cluster：

阶段1（up-proj，16 cluster 并行）:
  cluster_c 领 i_tile = c  (c = 0..15)
  - A(token) 用 multicast TMA 按 recv_token_source_info gather 进 SMEM（gather fusion，省 input_buf 拷贝）
  - gate = X@Wg[c]^T, up = X@Wu[c]^T （2x1SM tcgen05，K=d=4096）
  - SwiGLU: act_c = silu(gate)*up*route  （RMEM epilogue 内）
  - 写 act[:, c*256:+256] 到 group GMEM workspace
compute_group_sync(32 SM) + __threadfence_system()   ← up→down 依赖门（保证 16 act 全局可见）

阶段2（down-proj，16 cluster 并行）:
  cluster_c 领 d_tile = c
  - 读 act[256, 全 4096]（L2 常驻）算 out[:, c*256:+256] = act @ Wd[c]^T （K=I=4096 全规约）
  - 写最终 output（含多 local expert reduce，沿用现有 compute_output_f atomic 逻辑）
```

### I.9.4 复用与新增
- **直接复用**：阶段1 up-proj = 已验证的 `compute_ref/umma_swiglu_2cta.cu` 内核（gate+up+SwiGLU，2CTA，输出 256×256 act）。
- **新增**：阶段2 down-proj = `compute_ref/umma_gemm_2cta.cu` 的纯 2CTA GEMM，输入换成 act、权重换成 Wd。
- **改造 compute_worker**：cluster 内 tile 领取（i_tile/d_tile = cluster 在 group 内的序号）、两阶段 + 中间 group barrier、act workspace 读写。

### I.9.5 待解决的工程约束（实现时处理）
1. **cluster launch（I.5）**：整个 megakernel grid 要 cluster 化（cluster_dim=2）。需 SM 布局重排成偶数对齐、gridDim 被 2 整除、cluster 配对不跨 compute group 边界。dispatch/combine 不调 cluster API、行为不变。
2. **TMEM persistent 生命周期（R3）**：`compute_worker` 的 `while(true)` 里，每个 cluster 每 tile alloc/free TMEM（`Allocator2Sm`），或循环外分配一次复用——实现时确认不泄漏、不撞下一 tile。
3. **SMEM 预算（R4）**：compute SM 的 cluster mainloop SMEM（A/B multistage + mbarrier）要与 combine 路径 217KB 共处一个 `__launch_bounds__(800)` 和 228KB/SM 上限内。需实测。
4. **act workspace 布局**：group 共享的 `[256,4096]` bf16 buffer（2MB/group），16 个 up cluster 写、16 个 down cluster 读。确认 L2 命中（ncu lts hit rate）。
5. **多 local expert reduce + per-token-ready**：阶段2 输出后，沿用现有 `compute_output_f` float atomic reduce + `token_compute_done==expected` 发 `combine_token_ready` 的逻辑，位置不变。

### I.9.6 验证
- 数值：接回 megakernel 后用 `run_megakernel_v7_test.sh`，与现有 WMMA fused 版（已 commit `fab6030`）BF16 对齐。
- 性能：同脚本测整个 compute 阶段耗时 vs WMMA 版（这才是端到端 KPI，独立 microkernel 的 8.2x 仅供参考）。
- 顺序：先 hidden=intermediate=4096 跑通对齐，再考虑非对称尺寸（d-tile≠16）。

### I.9.10 阶段1 UMMA 内核接入 megakernel：TMA 路线 B1 vs B2

> 把 host-launched + CuTe-TMA 的 microkernel（2b/3'）塞进 device-side persistent compute_worker，
> 核心障碍是 TMA 数据加载怎么做（R1）。调研发现两条路，本质区别是**有没有权重 multicast**，
> 不是"有没有 host descriptor"——descriptor 只是 B2 实现 multicast/多维 tile 的载体。

**关键发现**：本仓 `csrc/kernels/utils.cuh:394-417` 已有 device 端 1D TMA helper（`tma_load_1d`/`tma_store_1d`），
基于裸 PTX `cp.async.bulk`，**不需要 host `CUtensorMap` descriptor**，参数是 `(smem, gmem_ptr, bytes, mbarrier)`，
gmem_ptr 当场算 → 天然支持 gather。DeepEP dispatch 末尾存 hidden_states 就用它（internode.cu:1176 按 token 索引算地址 TMA store）。

**B1 vs B2 本质区别表**：
- 拷贝形状：B1 = 1 维线性 bytes；B2 = 多维 tile（带 stride/swizzle）。
- 地址：B1 = device 端裸算 gmem_ptr；B2 = host `CUtensorMap` 编码 tensor shape/stride。
- **multicast（决定性）**：B1 **无**（2 CTA 各 load 自己的权重，HBM load 两遍）；B2 **有**（leader 一次 multicast 进 2 CTA SMEM，权重 HBM 流量减半）。
- swizzle：B1 手动；B2 descriptor 自带（喂 tcgen05 最优布局）。
- cluster launch：B1 **不需要**（无 multicast）→ 可关 cluster，dispatch/combine 性能恢复；B2 **需要** → dispatch/combine 继续陪绑降速。
- descriptor：B1 不需要；B2 需要（CuTe `make_tma_atom`，host 建好进 state，device 用 `tma_partition`+坐标）。

**B1（1D TMA，无 multicast，本质 1CTA + TMA 异步加速）**：
- device 端 `tma_load_1d` 搬 A（按 token gather 算地址）+ W（按 expert_id 偏移）进 SMEM → tcgen05+TMEM+SwiGLU。
- 纯 device、无 descriptor、gather 天然、复用本仓成熟 helper、不需 cluster。
- = 文档 I.9.9 的 1CTA 回退路线，但用异步 TMA 替代 cooperative_copy（load/MMA 可重叠）。放弃 2CTA 权重 multicast（少 1.2~1.5x）。

**B2（CuTe tiled TMA + descriptor，有 2CTA multicast）**：
- host（`megakernel.cu` 的 `__host__` 函数，非 deep_ep.cpp，避免 g++ 编 CuTe）为 **每个 expert 各建一个 2D `[I,d]` `SM100_TMA_2SM_LOAD_MULTICAST` atom**（W_gate/W_up 各 E 个），存进 `ComputeTmaAtoms`（device 可见），device 按 expert_id 选 atom。
- **关键决策（避免 3D TMA）**：不建 3D `[E,I,d]` descriptor（CuTe 3D batched multicast TMA 无确证样例）。改为 **per-expert 2D descriptor**——每个就是 tutorial 04 / stage 3' 验证过的 2D `[N,K]` multicast atom，零新未知。代价：host 建 2E 个 descriptor、state 存 2E 个（每个 128B，E≤64 可接受）。
- **A 的处理（选甲，已定）**：A 也走 TMA，但**指向 per-group 连续 input_buf workspace**（沿用现有 gather→input_buf 拷贝）。device 内核结构 = 已验证的 3' microkernel 几乎照搬（A/Wg/Wu 三个 TMA + 双 pass + SwiGLU）。**不做 gather fusion**（见下方后续优化）。
- 拿到 2CTA 权重 multicast（B-side HBM/SMEM 流量减半）。需 cluster launch（S4.2 那套，dispatch/combine 降速）。
- 实现要点：① per-expert 2D descriptor，device 按 expert_id 选；② `ComputeTmaAtoms` 按值含 Copy_Atom 数组，随 state 在 device global mem 可见；③ A 的 TMA atom 指向 input_buf（per-group workspace，地址 launch 前确定，host 建）。

**决策（用户定）**：**先做 B2**（对 multicast 收益有信心、一步到位拿完整 2CTA）。若端到端净收益不及预期（multicast 增益 < cluster 对 dispatch/combine 的拖累），按 I.9.9 回退 B1（关 cluster，dispatch/combine 恢复，保留 tcgen05 架构大头收益）。

**后续优化 TODO（gather fusion，S4.4 之后）**：当前 A 走「gather→input_buf 连续拷贝→TMA」，保留了一趟 GMEM 拷贝。SonicMoE 的 gather fusion 是直接在 load 时按 `recv_token_source_info` 索引 gather 进 SMEM，省掉 input_buf。待 B2 跑通对齐后再做：A 改用 cooperative_copy/1D-TMA 按 token 索引直接 gather（device 内核 mainloop 需混合 cooperative gather + W multicast TMA 两种同步），消除 input_buf 那趟拷贝 + 提升 L2 局部性。

### I.9.7 Stage 4 最小步骤实施计划（一步一步，每步独立可验证/可 commit）

> 原则（对齐 AGENTS.md「最小变量」）：每步只改一个因素，编译+单测通过再进下一步；每步可单独 commit/回退。
> 全程数值基线 = 已 commit `fab6030` 的 WMMA fused 版。每步跑 `run_megakernel_v7_test.sh` 验证。

- **S4.1 调大 batch 到 256（不改计算逻辑）**
  - 改 `COMPUTE_BATCH_SIZE` 128→256；同步检查 `gemm_workspace` 分段（input/gate/up/down 按 padded_m=256 翻倍分配）、`s_recv_token_idx[]` 等 `__shared__` 数组、SMEM 预算。
  - 计算仍走现有 WMMA `device_gemm_swiglu_fused`，**只验证 batch=256 端到端数值不变 + 不 OOM/不超 SMEM**。
  - 成功判据：单测 PASS，数值与 fab6030 一致。

- **S4.2 SM 布局偶数对齐 + 开启 cluster launch（不改计算逻辑）**
  - 重排 SM 分配使 `dispatch+combine+scheduler` 凑偶数边界、compute 区为偶数、gridDim 被 2 整除、cluster 配对不跨 compute group 边界。
  - megakernel launch 改用 cluster launch（cluster_dim=2）；dispatch/combine/compute 现有逻辑**不调 cluster API**，行为应不变。
  - 成功判据：cluster 化 grid 下单测仍 PASS（验证 I.5 的"行为不变"判断），dispatch/combine 无 hang/退化。

- **S4.3 group 内引入 act workspace（仍用 WMMA，拆两阶段验证依赖门）**
  - 给每个 group 分配 `[256,4096]` bf16 act workspace；compute_worker 拆成「阶段1 写 act → `compute_group_sync` → 阶段2 读 act 算 down」两段，**仍用 WMMA 内核**。
  - 验证：拆两阶段 + group barrier 依赖门的正确性（数值仍与 fab6030 一致），act workspace 读写无误。
  - 成功判据：单测 PASS。此步把"调度拆两阶段"与"换 UMMA 内核"解耦，先确保调度对。

- **S4.4 阶段1 换 UMMA 2CTA fused SwiGLU（接入已验证内核）**
  - 把阶段1 的 WMMA gate+up+SwiGLU 换成 `umma_swiglu_2cta.cu` 内核（16 cluster 各算一个 i-tile，输出 act 到 workspace）。
  - 处理 TMEM persistent 生命周期（R3）、A 的 multicast TMA gather（gather fusion）。
  - 成功判据：阶段1 输出的 act 与 WMMA 版 act 在 BF16 容差内一致（可加临时 act dump 比对），端到端单测 PASS。

- **S4.5 阶段2 换 UMMA 2CTA GEMM（down-proj）**
  - 把阶段2 的 WMMA down 换成 `umma_gemm_2cta.cu` 内核（16 cluster 各算一个 d-tile，读 act workspace）。
  - 保留现有多 local expert reduce（`compute_output_f` atomic）+ per-token-ready 逻辑。
  - 成功判据：端到端单测 PASS，数值对齐 fab6030。

- **S4.6 性能测量 + ncu**
  - `run_megakernel_v7_test.sh` 测整个 compute 阶段耗时 vs WMMA 版（端到端 KPI）。
  - ncu 看 Tensor Pipe util、DRAM throughput、**act workspace 的 L2 hit rate**（验证 I.9.0 的"落 L2 省 IO"假设）、权重 multicast 效果。
  - 实验台账记录每次运行（含失败/hang/OOM）到 `.agents/memory/experiments/YYYY-MM-DD.md`。

- **当前状态**：S4.1 待开始。

### I.9.8 Stage 4 实施进展
- 2026-06-26：**S4.1 改动完成（待用户编译验证）**。改 `megakernel.cu:45` `COMPUTE_BATCH_SIZE` 128→256，仅此一处常量。
  - 核查：所有依赖处都是符号引用，自动跟随——
    - `padded_m`/`gemm_workspace` 分段（megakernel.cu:1482-1488）随 256 翻倍。
    - host 端 `per_group_elems = COMPUTE_BATCH_SIZE*(2*hidden+2*intermediate)`（megakernel.cu:3277）自动翻倍。
    - `max_compute_tasks = num_local_experts*(max_tokens_per_expert/COMPUTE_BATCH_SIZE+2)`（megakernel.cu:3233）task 数减半，分配更小，安全。
    - scheduler 阈值（1407/1432）、`s_*` SMEM 数组（1494-1500）随 256。
  - SMEM 核查：`s_*` 静态数组 256→约 5KB（<48KB 静态预算）；动态 `smem_wmma_buf`(~50KB) 与 batch 无关。安全。
  - **计算仍走 WMMA `device_gemm_swiglu_fused`，逻辑不变，只验证 batch=256 端到端数值不变 + 不 OOM/超 SMEM。**
  - 验证方式：用户编译跑 `run_megakernel_v7_test.sh`，数值应与 fab6030 一致。
  - 下一步：S4.2（SM 布局偶数对齐 + cluster launch）。
- 2026-06-26：**S4.2 改动完成（待用户编译验证）**。开启 cluster launch（cluster_dim=2），仍用 WMMA 计算逻辑不变。
  - `megakernel.cu` launch：启用 `cudaLaunchKernelEx`，`clusterDim.x=2`（不再 `total_sms%2?2:1` 回退），`EP_HOST_ASSERT(total_sms%2==0)`——奇数布局直接报错。
  - **偶数对齐方案（按用户建议，弃用 padding hack）**：`COMPUTE_SCHEDULER_SMS` 1→2（megakernel.cu:47、deep_ep.cpp）。B300 布局：`dispatch24 + combine24 + scheduler2 + compute96 = 146`（偶），剩 2 SM reserved。
  - **关键守卫**：scheduler 角色加 `if(role_idx==0)`（megakernel.cu:2774）——**只有 scheduler #0 入队，#1 idle**。否则两个 scheduler 都扫 `expert_recv_count` 会 double-enqueue，task queue 错乱、token 算两遍。
  - **验证目的（I.5 的"行为不变"判断）**：cluster 化 grid 下 dispatch/combine/compute（仍 WMMA、不调 cluster API）数值与 fab6030 一致、无 hang。
  - 验证方式：用户编译跑 `run_megakernel_v7_test.sh`。看 launch 日志 `active_total_sms` 应为偶数。
  - 下一步：S4.3（act workspace + 拆两阶段 + group barrier 依赖门，仍 WMMA）。
- 2026-06-26：**S4.2 验证结果 — 正确性 PASS，但 dispatch/combine 性能下滑（暂可接受，先继续）**。
  - 现象：cluster 化 grid 后，计算数值与 fab6030 一致（印证 I.5「行为不变」），但 dispatch/combine 阶段耗时明显上升。
  - **疑似原因（待分离验证，未确认主因）**：
    1. **cooperative launch**：S4.2 的 `cudaLaunchKernelEx` 同时设了 `cudaLaunchAttributeCooperative`。若 fab6030 原来是普通 `<<<>>>`（cooperative 那段是注释），则本次同时引入 cooperative(grid-wide co-residency) + cluster，下滑可能主要来自 cooperative。
    2. **cluster co-schedule**：cluster_dim=2 强制 2 block 同 GPC 共驻、作为不可分割单位调度，降低 dispatch(NIC 密集、SM 占用低) 的调度自由度与通信并发。
    3. DSMEM 窗口预留可能影响重 SMEM 的通信 kernel occupancy。
  - **本质教训**：cluster 是 **grid 全局属性**，只有 compute 需要它，dispatch/combine 被迫陪绑 → 「行为不变 ≠ 性能不变」。这正是最初 R2 担心的「cluster 不能只给 compute」。
  - **待做的分离验证**（确认主因后再决定缓解）：测试A 回到普通 launch 看是否恢复；测试B 留 cluster 去 cooperative 看是否恢复；ncu 看 dispatch kernel occupancy。
  - **可能的缓解（看净收益）**：若 compute 占比大，UMMA 提速 8x 可覆盖 dispatch/combine 降速，净赚；若 dispatch/combine 占大头，考虑拆 kernel（compute 单独 cluster kernel + dispatch/combine 普通 kernel），但破坏 megakernel overlap 初衷。
  - **当前决定**：性能下降暂可接受，先继续 S4.3 推进主线；分离验证与缓解留作后续 TODO。
  - 下一步：S4.3（act workspace + 拆两阶段 + group barrier 依赖门，仍 WMMA）。
- 2026-06-26：**S4.3 跳过/合并**——现有 compute_worker 已是「up-proj+SwiGLU(写 up_buf) → group barrier → down-proj(读 up_buf)」两阶段结构，act workspace=up_buf、依赖门=compute_group_sync 都已存在（fab6030 即此结构），无独立改动价值，并入 S4.4。
- 2026-06-26：**S4.4（route B2）代码完成，运行时调试中**。把 UMMA 2CTA fused gate+up+SwiGLU 接入 compute_worker 阶段1（down 仍 WMMA）。
  - **新增文件 `csrc/kernels/megakernel_compute_umma.cuh`**：隔离所有 CuTe/CUTLASS（只被 megakernel.cu 这个 nvcc TU include，deep_ep.cpp 用 g++ 不碰）。含：
    - **per-expert 2D multicast TMA atom**（避免 3D）：`make_weight_tma_atom(W+e*I*d, I, d)` 每 expert 一个 `SM100_TMA_2SM_LOAD_MULTICAST`；`ComputeTmaAtoms` 存 2E 个（wgate/wup）。A 走 `make_input_tma_atom`（指向 per-group input_buf，沿用 gather→input_buf 拷贝，gather fusion 留后）。
    - **device 内核 `umma_up_swiglu_tile`**：照搬已验证的 3' microkernel（2CTA tcgen05 + TMEM + 双 pass + SwiGLU），per-call TMEM alloc/free，输出 act[:, i_tile*256:+256] 到 up_buf。
  - **MegaKernelState** 加 `compute_tma`/`group_input_tma`/`num_compute_groups`；host 端 build + cudaMemcpy 上传（仅 hidden==intermediate 且 E≤64 时启用，否则 nullptr → WMMA fallback）。`MegaKernelState` 对 deep_ep.cpp 不透明（api.cuh 仅前向声明），故塞 CuTe 类型不污染 g++ 编译。
  - **compute_worker 阶段1**：`state->compute_tma != nullptr` 时，group 32 SM = 16 个 2-CTA cluster，cluster c 算 i_tile=c 的 act tile；否则 WMMA fallback。
  - **编译**：`TORCH_CUDA_ARCH_LIST="10.3"`（注意：实际生成 `compute_103,code=sm_103`，**无 a 后缀**，tcgen05 是否运行时合法待观察）。修了 3 个编译错：`make_mma_tiler` 重复 `inline`、`make_sA/sB_layout` 需 `CUTE_HOST_DEVICE`、cluster launch `cudaLaunchConfig_t{0}`→`{}`（-Werror）。**编译通过**。
  - **运行时 bug 1（已修）**：thread 768 越界写 + `tcgen05_guardrail_trap_phase_invalid_during_alloc`。根因：800 线程全进了 128 线程的 UMMA 内核。修复：调用处加 `thread_id < 128` guard，只前 128 线程进内核。
  - **运行时 bug 2（已修）**：UMMA 内核内部 `__syncthreads()` 在 `if(thread_id<128)` 分支内 → 等全 800 线程 → 死锁。修复：新增 `umma_sync128()`（`barrier.sync 1, 128` named barrier，只同步参与的 128 线程），内核内两处 `__syncthreads` 换成它；`cluster_sync()` 保留。
  - **待验证的残留风险**：① arch `sm_103`(无 a) 下 tcgen05 是否合法（否则要 `10.3a` 重编）；② `cluster_sync()` 在"每 CTA 仅 128/800 线程活跃"下是否正确（可能仍 hang）；③ TMA Copy_Atom memcpy 到 device 后 descriptor 是否有效；④ ClusterSharedStorage 是否超 SMEM 预算；⑤ 测试 num_tokens 是否够凑满 256 触发 UMMA 路径（否则走 WMMA fallback，验证不到）。
  - 下一步：重编重跑，按现象（出数/OOB/hang）继续定位。

### I.9.9 2CTA vs 1CTA 决策与回退预案

**2CTA 相对 1CTA 的优势来源**：
1. **权重 B multicast 共享（核心）**：cluster 2 CTA 协作算一个 M=256 tile，B(权重) 经 `SM100_TMA_2SM_LOAD_MULTICAST` 只 load 一份 multicast 到 2 CTA → B-side HBM/SMEM 流量减半。直接命中"权重重复 load"痛点。
2. **M_tile 翻倍到 256**：`2x1SM` 一条 UMMA 算 M=256（1CTA `_SS` 是 128），算术强度（权重复用率≈M）翻倍，更接近 compute-bound。
3. **TMEM 容量翻倍**（`Allocator2Sm`），放更大 accumulator。

**优势有多大（诚实评估）**：
- 2CTA 优势主要在**省 IO（带宽侧）**，不是加倍算力。memory-bound（细粒度 MoE、权重 IO 主导）收益显著；compute-bound 收益有限。
- **microkernel 实测的 8.2x 是「UMMA vs WMMA」总差距，不是「2CTA vs 1CTA」增量**——8.2x 大头来自 WMMA→tcgen05 架构换代 + M=128→256 + fused SwiGLU 省 GMEM，2a/3' 的 M/shape 不同不能直接相减。
- **2CTA vs 1CTA 的干净增量未实测**。基于 SonicMoE 数据与算术强度**预估**：M=256/K=N=4096 偏 memory-bound 下约 **1.2~1.5x**（权重流量减半 + M 摊销），非数倍。

**代价**：2CTA 需 cluster launch，正是 S4.2 中 dispatch/combine 性能下滑的根源（cluster 是 grid 全局属性，dispatch/combine 被迫陪绑）。

**决策与回退预案（当前路线）**：
- **先做 2CTA 实现**（S4.3→S4.6），因 microkernel 已验证、收益机制清晰。
- **若端到端效果不理想**（2CTA 计算增益 < cluster 给 dispatch/combine 的拖累），**回退 1CTA 方案**：
  - 1CTA = `SM100_MMA_F16BF16_SS`（M_tile=128），**不需要 cluster launch** → 不开 cluster，dispatch/combine 调度自由度恢复，S4.2 的性能下滑消失。
  - 1CTA 仍拿到 UMMA vs WMMA 的绝大部分收益（架构换代是大头），只是少了权重 multicast 那 1.2~1.5x。
  - 1CTA 是不依赖 cluster 的安全基线，这正是当初坚持「1CTA 先行、2CTA 后置」的原因。
- **待补 TODO**：做 1CTA vs 2CTA 的**控制变量 microbench**（M=256 同条件，仅 atom/cluster 不同，ncu 看 DRAM bytes 验证权重流量减半），量化 2CTA 真实增量——作为"是否值得为 2CTA 开 cluster"的决策依据。