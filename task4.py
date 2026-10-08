import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

BIG = 10 ** 6  # затычка для поиска min ребра


def hungarian(a):
    # венгерский метод через потенциалы, ищем минимум, возвращаем (суммарный вес, match), match[i] - столбец для строки i
    n = len(a)
    INF = float("inf")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)  # p[j] - какая строка заняла столбец j
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    match = [0] * n
    for j in range(1, n + 1):
        match[p[j] - 1] = j - 1
    s = 0
    for i in range(n):
        s += a[i][match[i]]
    return s, match


def split_bipartite(n, edges):
    # обычный BFS в 2 цвета, если не получилось - возвращаем None
    adj = [[] for _ in range(n)]
    for x, y, w in edges:
        adj[x].append(y)
        adj[y].append(x)

    color = [-1] * n
    for s in range(n):
        if color[s] != -1:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if color[y] == -1:
                    color[y] = 1 - color[x]
                    q.append(y)
                elif color[y] == color[x]:
                    return None
    return color


def matching_of_graph(n, edges):
    # красим граф и строим матрицу весов
    color = split_bipartite(n, edges)
    if color is None:
        print("Граф не двудольный, венгерский алгоритм не применим")
        return None

    left = [v for v in range(n) if color[v] == 0]
    right = [v for v in range(n) if color[v] == 1]

    if len(left) != len(right):
        print("Доли разные, полного паросочетания не будет")
        return None

    li = {}
    for i, v in enumerate(left):
        li[v] = i
    ri = {}
    for i, v in enumerate(right):
        ri[v] = i

    a = [[BIG] * len(right) for _ in left]
    for x, y, w in edges:
        if color[x] == 1:
            x, y = y, x
        a[li[x]][ri[y]] = w

    total, match = hungarian(a)
    if total >= BIG:
        print("Полного паросочетания нет")
        return None

    pairs = []
    for i in range(len(left)):
        pairs.append((left[i], right[match[i]]))
    return total, pairs, (left, right)


def draw_matching(n, edges, pairs, parts, filename):
    # рисуем две колонки, красным - выбранные ребра
    left, right = parts
    pos = {}
    for k, v in enumerate(left):
        pos[v] = (0, -k)
    for k, v in enumerate(right):
        pos[v] = (1, -k)

    chosen = set()
    for p in pairs:
        chosen.add(frozenset(p))

    fig, ax = plt.subplots(figsize=(6, 5))
    for x, y, w in edges:
        sel = frozenset((x, y)) in chosen
        if sel:
            ax.plot([pos[x][0], pos[y][0]], [pos[x][1], pos[y][1]],
                    "r-", lw=3, zorder=1)
        else:
            ax.plot([pos[x][0], pos[y][0]], [pos[x][1], pos[y][1]],
                    "0.7", lw=1, zorder=1)
        mx = pos[x][0] * 0.75 + pos[y][0] * 0.25
        my = pos[x][1] * 0.75 + pos[y][1] * 0.25
        ax.text(mx, my, str(w), fontsize=8, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.5))

    for v in pos:
        x, y = pos[v]
        ax.scatter(x, y, s=400, c="lightblue", edgecolors="k", zorder=2)
        ax.text(x, y, str(v), ha="center", va="center", zorder=3)
    ax.axis("off")
    fig.savefig(filename, dpi=120)
    plt.close(fig)


# а: матрица весов, строки = левая доля
w = [[4, 1, 3],
     [2, 0, 5],
     [3, 2, 2]]

total, match = hungarian(w)
print("а) минимум:", total, "назначения:", list(enumerate(match)))

# максимум = минимум от -w
neg = []
for row in w:
    neg.append([-x for x in row])
total_max, _ = hungarian(neg)
print("   максимум:", -total_max)

# б: граф списком ребер
n = 6
edges = [(0, 3, 2), (0, 4, 5), (1, 3, 4), (1, 5, 1), (2, 4, 3), (2, 5, 6)]

res = matching_of_graph(n, edges)
if res is not None:
    total, pairs, parts = res
    print("б) вес:", total, "ребра:", pairs)
    draw_matching(n, edges, pairs, parts, "matching.png")

