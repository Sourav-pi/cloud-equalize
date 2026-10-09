# Algorithms And Implementation

## Intensity Mapping

For target intensity `I`, count the values at or below `I` (`CDF(I)`) and the values at the minimum occupied intensity (`Cmin`). Remap using integer division:

```text
I_new = 255 * max(0, CDF(I) - Cmin) / (m - Cmin)
```

If `m == Cmin`, keep the original intensity. KNN neighborhoods include the target, so `m = k + 1`. K-means uses the cluster's population. All output coordinates retain their original order.

The original assignment specifies neighbor-only KNN histograms and `m = k`; this repository preserves the implemented self-inclusive convention. It also restricts coordinates to integers, whereas the specification allows real coordinates.

## Exact KNN

Each CUDA thread processes one target point and scans every other point. Blocks of 256 threads cooperatively load 256 aligned, 16-byte point records into shared memory, with synchronization before consumption and before reuse. Invalid threads in the final partial block still participate in barriers.

A per-thread max-heap retains the best `k` candidates, ordered by squared distance and then neighbor `(x, y, z)`. Unique positions make that ordering unambiguous. Selection and remapping share one kernel; remapping scans the chosen intensities instead of constructing 256 histogram bins. Separate device input and output prevent updates from affecting other threads' neighborhoods. Host point arrays use `cudaMallocHost`; copies are synchronous.

The distance scan is quadratic in `n`; heap maintenance is at most `O(n^2 log k)` total work, plus `O(n k)` remapping. Heap arrays have capacity 128. Large heaps may spill to local memory; shared tiling alone does not establish a measured speedup.

## Morton Approximate KNN

CPU preprocessing offsets each coordinate by 10,000 and interleaves 21 bits per axis into a 63-bit Morton code. OpenMP parallelizes code generation and sorted-index extraction; `std::sort` itself is sequential. Sorting `(code, original_index)` gives deterministic ordering. A gather kernel builds the sorted GPU point array.

For sorted position `i`, all 128 threads in its block share this candidate interval:

```text
center = floor(i / 128) * 128 + 64
start  = max(0, center - 7500)
end    = min(n - 1, start + 15000)     # inclusive
```

The interval contains at most 15,001 entries, before excluding the target. This is intentionally the CUDA code's interval, including its forward shift near the beginning and shortened interval near the end. It is not a symmetric per-point radius search. Blocks load 896-point chunks (`128 * (2 * 3 + 1)`) into shared memory. A bounded heap selects neighbors within that window, and the fused kernel scatters remapped points back to original order.

Host sorting costs `O(n log n)`; candidate processing costs at most `O(n W log k)` for `W <= 15001`, plus remapping. Small clouds fitting entirely within every block's window have the exact neighborhood. Larger clouds can lose neighbors at Morton discontinuities. Intensity MAE measures output distortion, not neighbor recall, and depends on the intensity distribution.

## K-means

Initial centroids are the first `k` input coordinates. Assignments minimize squared distance, then centroid `(x, y, z)` lexicographically; identical centroids retain the first cluster index. A 256-thread assignment kernel reads centroids from constant memory and accumulates cluster populations.

The update kernel accumulates coordinate sums and counts into block-local shared arrays, then merges them into global buffers using atomics. Signed 64-bit sums use a CAS-based atomic helper. The host copies sums/counts back, divides integer sums toward zero, retains empty centroids, and stops when centroids no longer change or `T` iterations finish. At `T = 0`, one assignment pass still runs. The final histogram uses the last assignment pass, without an extra reassignment after the last centroid update.

The histogram kernel atomically fills `k * 256` bins. Each point then scans its cluster's bins to compute `Cmin` and its target CDF; a final kernel remaps intensity. This repeats CDF work across points rather than computing one reusable CDF per cluster. Assignment costs `O(T n k)` in the worst case; histogram/remapping costs `O(256 n)`.