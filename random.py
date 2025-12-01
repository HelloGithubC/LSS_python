import numpy as np

def generate_sparse_distribution(x_edges, hist_array, size=1000):
    """
        生成不插值的给定概率密度函数的随机数
    """
    heap_size = int(size * 1.2)
    heap_list = []
    hist_array = hist_array.astype(np.float64, copy=False)
    if not np.isclose(np.sum(hist_array), 1.0):
        hist_array /= np.sum(hist_array)
    if x_edges[0] > x_edges[-1]:
        x_edges = x_edges[::-1]
        hist_array = hist_array[::-1]
    for i in range(len(hist_array)):
        if hist_array[i] < 0:
            hist_array[i] = 0
        if hist_array[i] > 1:
            hist_array[i] = 1 
        heap_list.append(np.random.uniform(x_edges[i], x_edges[i+1], int(hist_array[i] * heap_size)))
    heap_array = np.concatenate(heap_list)
    return np.random.choice(heap_array, size=size, replace=False)