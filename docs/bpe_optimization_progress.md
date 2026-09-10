# BPE 优化进度与后续路线

更新时间：2026-09-10

本文档用于在长对话、换会话或换机器后快速恢复 CS336 Assignment 1 当前进度。BPE training 的实现与主要优化已经完成；最终 AutoDL 性能验收暂不阻塞学习主线。下一阶段直接进入 tokenizer encoding / decoding，OpenWebText BPE training 暂缓。

## 1. 当前结论

### BPE correctness

- `train_bpe()` 已实现。
- 测试命令：

```bash
uv run pytest tests/test_train_bpe.py -v
```

- 当前结果：`3 passed`。
- heap 优化后再次通过 `3 passed`，说明 vocab / merges 以及 tie-breaking 行为保持正确。
- heap 优化提交：`b8da7b2aa3694daa24a3032907f630998be23335`（`使用heap优化best_pair寻找`）。

### TinyStories training script

脚本：

```text
scripts/train_bpe_tinystories.py
```

配置：

- input：`data/TinyStoriesV2-GPT4-train.txt`
- vocab size：`10000`
- special token：`<|endoftext|>`
- 输出：
  - `outputs/tinystories_vocab.pkl`
  - `outputs/tinystories_merges.pkl`

## 2. 已完成的 TinyStories 实验

### 初始完整训练

结果：

- Training time：`892.06 s`（约 14 分 52 秒）
- Vocab size：`10000`
- Number of merges：`9743`
- Longest token：`" accomplishment"`
- Longest token length：`15 bytes`
- 运行中通过 `top` 观察到主 Python 进程约 `10.6 GB RSS`

Sanity check：

```text
9743 = 10000 - 256 byte tokens - 1 special token
```

最长 token `" accomplishment"` 合理：TinyStories 是英语儿童故事语料，高频完整单词经过多轮 BPE merge 后成为单 token 很正常；前导空格来自 GPT-2 风格 pre-tokenization。

### 8-process pre-tokenization 版本

真实非 profiler 结果：

- Pretokenization：`115.13 s`
- BPE build/merge：`72.12 s`
- Total training：`187.28 s`

相对初版：

```text
892.06 s -> 187.28 s
```

约 `4.76x` wall-clock speedup。

## 3. 已完成优化 A：并行 pre-tokenization

当前实现新增：

- `count_chunk(...)`
- `find_chunk_boundaries(...)`
- `parallel_count_pretokens(...)`

设计：

1. 使用 `<|endoftext|>` 作为安全文档边界。
2. `find_chunk_boundaries()` 以 binary mode 查找切点。
3. 将 `[start, end)` chunk 分发给 `multiprocessing.Pool`。
4. worker 独立读取 chunk、UTF-8 decode、执行 `count_pretokens()`。
5. 主进程使用 `Counter.update()` 汇总每个 worker 的 pre-token counts。
6. BPE pair bookkeeping 保持增量更新。

AutoDL 实例：

- CPU：16 cores
- RAM：80 GB
- GPU：RTX 4090D（BPE 基本不用 GPU）

当前 `train_bpe()` 中 `num_processes` 已从 `8` 调到 `16`，用于最终 TinyStories 性能实验。

## 4. Profiling 结论

使用过：

```bash
uv run python -m cProfile \
-o outputs/bpe_profile.prof \
scripts/train_bpe_tinystories.py
```

注意：`cProfile` instrumentation overhead 很明显，不可把 profiler run 的 wall-clock 当作真实性能基线。

Profiler run：

- Pretokenization：`244.27 s`
- BPE build/merge：`144.25 s`
- Total：`388.55 s`
- function calls：`381,962,867`

`tottime` 关键结果：

```text
15323 calls     builtins.max       69.577 s tottime / 134.812 s cumtime
369218707       <lambda>           65.235 s tottime
277780          merge_pair          1.171 s
1               build_pair_data     0.341 s
```

结论：旧实现中最大热点是每轮 merge 都执行：

```python
best_pair = max(
    pair_counts,
    key=lambda pair: (pair_counts[pair], pair),
)
```

这会在约 `9743` 次 merge 中反复全量扫描 `pair_counts`，最终产生约 `3.69 亿` 次 lambda 调用。

`merge_pair()` 和 `build_pair_data()` 并不是主要瓶颈。

## 5. 已完成优化 B：heap + lazy invalidation

当前实现已经用 `heapq` 替代每轮全量 `max()`。

核心结构：

- `HeapEntry` 保存：
  - `pair`
  - `count_snapshot`
- `HeapEntry.__lt__()` 将 `(count_snapshot, pair)` 的比较方向整体反转，使 Python min-heap 保持原来 `max((count, pair))` 的语义：
  - count 更大者优先；
  - count 相同，pair 字典序更大者优先。
- 初始阶段：所有 `pair_counts.items()` 构造 entries 后 `heapq.heapify()`。
- 每轮取 best pair：
  - `heappop()`；
  - pair 已不存在 -> stale -> `continue`；
  - `count_snapshot != pair_counts[pair]` -> stale -> `continue`；
  - 否则该 entry 为当前 best pair。
- merge 后只对 `touched_pairs` 增量处理：
  - count < 0 -> RuntimeError；
  - count == 0 -> 从 `pair_counts` 删除；
  - count > 0 -> push 最新 `HeapEntry`。
- 不在 heap 中原地寻找和修改旧 entry；stale entry 到达堆顶时再丢弃。

这版实现已经通过 `tests/test_train_bpe.py` 的全部 3 个测试。

## 6. 尚未完成：最终 BPE 性能验收

2026-09-10 状态：

- 16-process + heap 版本已经准备好并开始用于 AutoDL 最终实验。
- 截至本次交接，尚未拿到最终完整日志结果；该实验暂不阻塞后续学习。
- 后续任意时间拿到结果后，只需补记：
  - pretokenization time
  - BPE build/merge time
  - total training time
  - vocab size / merge count
  - longest token
  - 如有可靠测量，再补 memory

目标参考：handout 提示 TinyStories BPE training 经 multiprocessing 等优化后应可做到 `< 120 s`。若最终仍略高于 2 分钟，但 correctness 正确且性能已显著改善，不再无限优化；优先继续 Assignment 主线。

## 7. BPE 最终清理（以后一次性做）

在正式把 BPE 标记为完全 DONE 前检查：

1. `uv run pytest tests/test_train_bpe.py -v` -> `3 passed`。
2. 根据最终实验决定是否移除 `train_bpe()` 内临时 profiling prints：
   - `pretokenization time`
   - `BPE build merge time`
3. 检查 `num_processes = 16` 是否应改成更合适的内部默认策略，避免对小测试/不同机器过度开进程。
4. 检查 split delimiter `b"<|endoftext|>"` 的硬编码；`train_bpe()` 是通用接口，不应永久只适配 TinyStories。
5. 保存 TinyStories writeup 信息：time、memory、longest token、why it makes sense。

除非最终性能明显异常，否则不再优先优化 `merge_pair()`。

## 8. 路线调整：暂缓 OpenWebText BPE training

当前决定：**先跳过 OpenWebText 的 32K BPE 全量训练，直接进入 tokenizer encoding / decoding。**

原因：

- OpenWebText BPE 是重型实验，不是理解 tokenizer 编解码接口的前置条件。
- TinyStories 已经生成一套可用 vocab + merges，可用于后续 tokenizer 学习和实验。
- 先实现 `Tokenizer` 能更快进入 Assignment 的下一组核心代码与测试。
- OpenWebText training 以后需要比较两个 tokenizer 或做完整 tokenizer experiments 时再补。

因此不要把“OWT 尚未训练”误判成 blocker。

## 9. 下一阶段：Tokenizer encoding / decoding

下一次继续时，不再从 heap 开始，直接进入 tokenizer。

目标模块至少包括：

- tokenizer initialization：保存 vocab / merges / special tokens，并建立 encode/decode 所需的反向映射和 merge rank。
- `encode(text: str) -> list[int]`
- `decode(ids: list[int]) -> str`
- special-token handling：special tokens 必须整体保留，不参与普通 BPE merge；需要正确处理连续和 overlapping special tokens。
- Unicode / byte-level round trip。
- `encode_iterable(...)`：为大文件流式编码准备。

对应主要测试文件：

```text
tests/test_tokenizer.py
```

建议学习/实现顺序：

1. 先读 tokenizer problem 原文和 `tests/test_tokenizer.py`。
2. 画清楚 decode：token id -> bytes -> 拼接 -> UTF-8 decode。
3. 再做 encode：text -> special-token 分段 -> GPT-2 regex pre-tokenization -> bytes -> 按 merge rank 应用 BPE -> token IDs。
4. 先过最小 round-trip / ASCII tests。
5. 再处理 Unicode、special tokens、overlapping special tokens。
6. 最后实现 `encode_iterable` 并跑整个 `tests/test_tokenizer.py`。

## 10. 下一次继续时从这里开始

直接读取：

```text
docs/bpe_optimization_progress.md
```

然后开始：

> CS336 Assignment 1 — BPE Tokenizer Encoding and Decoding：先读题目与 `tests/test_tokenizer.py`，理解 `Tokenizer` 类需要保存哪些状态，以及 `decode()` 为什么比 `encode()` 更容易先实现。

不要先跑 OpenWebText BPE；不要重新做 heap profiling；最终 AutoDL BPE 结果以后补记即可。
