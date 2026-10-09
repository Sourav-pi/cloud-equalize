#ifndef KNN_HEAP_UTILS_CUH
#define KNN_HEAP_UTILS_CUH

struct Neighbor {
    int dist;
    int idx;
};

static __device__ __forceinline__ bool better(const Neighbor& a, const Neighbor& b,
                                                        const int* x, const int* y, const int* z) {
    if (a.dist != b.dist) return a.dist < b.dist;
    if (x[a.idx] != x[b.idx]) return x[a.idx] < x[b.idx];
    if (y[a.idx] != y[b.idx]) return y[a.idx] < y[b.idx];
    if (z[a.idx] != z[b.idx]) return z[a.idx] < z[b.idx];
    return a.idx < b.idx;
}

static __device__ __forceinline__ bool worse(const Neighbor& a, const Neighbor& b,
                                                       const int* x, const int* y, const int* z) {
    return better(b, a, x, y, z);
}

static __device__ __forceinline__ void bubble_up(Neighbor* heap, int pos,
                                                           const int* x, const int* y, const int* z) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (!worse(heap[pos], heap[parent], x, y, z)) break;
        Neighbor tmp = heap[parent];
        heap[parent] = heap[pos];
        heap[pos] = tmp;
        pos = parent;
    }
}

static __device__ __forceinline__ void bubble_down(Neighbor* heap, int size, int pos,
                                                             const int* x, const int* y, const int* z) {
    while (true) {
        int left = 2 * pos + 1;
        int right = left + 1;
        int worst = pos;

        if (left < size && worse(heap[left], heap[worst], x, y, z)) worst = left;
        if (right < size && worse(heap[right], heap[worst], x, y, z)) worst = right;

        if (worst == pos) break;
        Neighbor tmp = heap[pos];
        heap[pos] = heap[worst];
        heap[worst] = tmp;
        pos = worst;
    }
}

#endif
