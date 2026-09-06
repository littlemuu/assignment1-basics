# BPE 优化进度与后续路线

更新时间：2026-09-06

本文档用于在长对话、换会话或换机器后快速恢复 CS336 Assignment 1 中 BPE tokenizer training 的当前状态。重点记录已经验证过的实现、AutoDL 实验结果、profiling 结论和下一步优化路线。

## 1. 当前完成状态

### 正确性

- `train_bpe()` 已实现并通过：
  - `uv run pytest tests/test_train_bpe.py -v`
  - 当前结果：`3 passed`
- 已完成 TinyStories BPE training script：
  - `scripts/train_bpe_tinystories.py`
- TinyStories 配置：
  - vocab size：`10000`
  - special token：`<|endoftext|>`
  - input：`data/TinyStoriesV2-GPT4-train.txt`
- 训练产物：
  - `outputs/tinystories_vocab.pkl`
  - `outputs/tinystories_merges.pkl`

### TinyStories 训练结果

第一次完整训练结果：

- Training time：`892.06 s`（约 14 分 52 秒）
- Vocab size：`10000`
- Number of merges：`9743`
- Longest token：`" accomplishment"`
- Longest token length：`15 bytes`
- 运行中通过 `top` 观察到主 Python 进程约 `10.6 GB RSS`

`9743 = 10000 - 256 byte tokens - 1 special token`，与预期完全一致。

最长 token `" accomplishment"` 合理：TinyStories 是英语儿童故事语料，高频完整单词可能经过多轮 BPE merge 成为单 token；前导空格来自 GPT-2 风格 pre-tokenization。

## 2. 已完成优化：并行 pre-tokenization

### 设计

当前实现已新增：

- `count_chunk(...)`
- `find_chunk_boundaries(...)`
- `parallel_count_pretokens(...)`

核心思路：

1. 使用 `<|endoftext|>` 作为安全文档边界。
2. `find_chunk_boundaries()` 以 binary mode 查找切点。
3. 将每个 `[start, end)` chunk 分发给 `multiprocessing.Pool`。
4. 每个 worker 独立执行 `count_chunk()`。
5. 主进程使用 `Counter.update()` 汇总各 chunk 的 pre-token counts。
6. 后续 pair counting / merge 逻辑保持不变。

### 当前参数

`train_bpe()` 内当前临时硬编码：

```python
num_processes = 8
```

当前 AutoDL 实例：

- CPU：16 cores
- RAM：80 GB
- GPU：RTX 4090D（BPE 训练基本不用 GPU）

### 8-process 实验

正常运行（非 profiler）结果：

- Pretokenization：`115.13 s`
- BPE build/merge：`72.12 s`
- Total training：`187.28 s`

相对旧版：

- `892.06 s -> 187.28 s`
- 约 `4.76x` wall-clock speedup

阶段时间之和：

- `115.13 + 72.12 = 187.25 s`
- 与总时间 `187.28 s` 基本一致，说明阶段划分准确、额外开销很小。

## 3. 当前主要瓶颈：merge 阶段寻找 best pair

使用：

```bash
uv run python -m cProfile \
-o outputs/bpe_profile.prof \
scripts/train_bpe_tinystories.py
```

注意：`cProfile` 带来很明显的 instrumentation overhead，不应使用 profiler run 的 wall-clock time 作为性能基线。

Profiler run：

- Pretokenization：`244.27 s`
- BPE build/merge：`144.25 s`
- Total：`388.55 s`
- function calls：`381,962,867`

相较正常运行基本约慢 2 倍，因此该 run 只用于定位热点。

### `tottime` 关键结果

```text
15323 calls   builtins.max       69.577 s tottime / 134.812 s cumtime
369218707      <lambda>           65.235 s tottime
277780         merge_pair          1.171 s
1              build_pair_data     0.341 s
```

结论非常明确：

当前最大瓶颈是每轮 merge 都执行：

```python
best_pair = max(
    pair_counts,
    key=lambda pair: (pair_counts[pair], pair),
)
```

这一操作会在每轮 merge 对整个 `pair_counts` 做全量扫描，并为每个 pair 调用一次 Python lambda。TinyStories 一共需要 `9743` 次 merge，因此最终产生约 `3.69 亿` 次 lambda 调用。

当前暂时不值得优先优化：

- `merge_pair()`
- `build_pair_data()`
- `set.add()` / `dict.get()` 等局部更新逻辑

它们相对 best-pair selection 的耗时小得多。

## 4. 下一步优化路线

### Phase A：用 heap / priority queue 替代每轮全量 `max()`

目标：

- 不再每轮扫描整个 `pair_counts`
- 只针对当前 merge 后发生变化的 `touched_pairs` 更新候选状态

计划结构：

1. 引入 `heapq`。
2. 初始化时把所有 pair 的优先级放入 heap。
3. Python `heapq` 是 min-heap，需要设计能够表达“count 最大优先”的 key。
4. 必须保持原始 tie-breaking 语义：
   - count 更大者优先；
   - count 相同，pair 字典序更大的优先。
5. pair count 改变时，不尝试从 heap 中原地删除旧记录。
6. 对 `touched_pairs` 直接 push 新记录。
7. 采用 lazy invalidation：
   - heap 顶部记录如果与 `pair_counts` 当前 count 不一致，则视为 stale，pop 丢弃；
   - 直到找到与当前状态一致的记录，才作为 `best_pair`。

### Phase A 验收

每次结构性修改后必须先执行：

```bash
uv run pytest tests/test_train_bpe.py -v
```

要求：

- `3 passed`
- vocab / merges 行为与原实现一致
- 特别注意 tie-breaking 不能改变

### Phase B：扩大 pre-tokenization 并行度

在 heap 优化正确后：

- AutoDL CPU 为 16 cores
- 将 `num_processes` 从 `8` 调到 `16`
- 重新跑完整 TinyStories

理论目标：

- pre-tokenization：从约 `115 s` 进一步降低
- merge：经过 heap 优化后显著低于当前 `72 s`
- 总时间目标：`< 120 s`

handout 明确提示：通过 multiprocessing pre-tokenization 等优化，TinyStories BPE training 应能够做到 2 分钟以内。

### Phase C：重新 profiling（仅在必要时）

如果 Phase A + 16 processes 后仍明显高于 2 分钟：

1. 优先使用低开销 profiler（如 `py-spy`）或针对 merge 局部计时。
2. 不要再次把 cProfile wall-clock 当作真实性能数据。
3. 再判断是否需要优化：
   - heap stale-entry 数量
   - touched-pair 更新成本
   - pair bookkeeping 数据结构

## 5. 当前实验代码中的临时项

以下内容目前为了 profiling / TinyStories 实验暂时保留，最终应清理或重构：

### `train_bpe()` 内 profiling print

当前内部打印：

- `pretokenization time`
- `BPE build merge time`

用途：性能定位。

优化完成后应考虑移除，避免通用 `train_bpe()` 带实验日志副作用。

### `num_processes = 8` 硬编码

当前是实验参数，不应长期硬编码在通用算法接口中。

后续可考虑：

- 根据 CPU 数量选择合理默认值；或
- 由内部 helper 统一管理；或
- 在不破坏 assignment 要求的前提下设计可配置方式。

### `<|endoftext|>` boundary 硬编码

当前：

```python
find_chunk_boundaries(input_path, num_processes, b"<|endoftext|>")
```

适用于 TinyStories，但 `train_bpe()` 本身是通用接口。完成性能优化后需重新检查这一设计是否应从 `special_tokens` 中安全推导 split delimiter，避免把 TinyStories 特定假设永久写死在通用函数里。

## 6. AutoDL 工作流备忘

仓库：

```bash
cd /root/autodl-tmp/assignment1-basics
source /etc/network_turbo
git pull
```

BPE 测试：

```bash
uv run pytest tests/test_train_bpe.py -v
```

完整 TinyStories：

```bash
nohup uv run python scripts/train_bpe_tinystories.py \
> outputs/train_bpe_tinystories.log 2>&1 &
```

查看日志：

```bash
tail -f outputs/train_bpe_tinystories.log
```

查看 Python worker：

```bash
ps -eo pid,ppid,%cpu,%mem,rss,cmd | grep python
```

## 7. 后续 Assignment 路线

BPE 性能达到可接受水平后：

1. 完成 `train_bpe_tinystories` writeup：
   - time
   - memory
   - longest token
   - why it makes sense
2. 准备 OpenWebText BPE experiment：
   - vocab size `32000`
   - 资源要求显著更高
3. 实现 BPE tokenizer encoding / decoding。
4. 进入 tokenizer experiments：compression ratio、throughput、dataset encoding 等。

## 8. 下次继续时从这里开始

下一课直接从以下问题继续：

> 如何用 `heapq` 保存 pair priority，同时严格保留当前 `max(pair_counts, key=lambda pair: (pair_counts[pair], pair))` 的 tie-breaking 语义，并通过 lazy invalidation 处理 pair count 更新？

不要先优化 `merge_pair()`；profiling 已经证明当前主要热点是 best-pair selection。
