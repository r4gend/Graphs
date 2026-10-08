import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_cycle(adj):
    # ищем любой цикл в графе обычным DFS
    par = {}

    def dfs(v, p):
        par[v] = p
        for w in adj[v]:
            if w == p:
                continue
            if w in par:
                # наткнулись на уже посещенную - цикл
                cyc = [v]
                x = v
                while x != w:
                    x = par[x]
                    cyc.append(x)
                return cyc
            r = dfs(w, v)
            if r is not None:
                return r
        return None

    start = next(iter(adj))
    return dfs(start, None)


def planar_faces(adj):
    # гамма-алгоритм (Демукрон-Мальгранж-Пуаза)
    cyc = find_cycle(adj)
    if cyc is None:
        return None

    faces = [cyc[:], cyc[:]]  # цикл делит плоскость на 2 грани
    emb_v = set(cyc)
    emb_e = set()
    for i in range(len(cyc)):
        emb_e.add(frozenset((cyc[i], cyc[(i + 1) % len(cyc)])))

    all_e = set()
    for a in adj:
        for b in adj[a]:
            all_e.add(frozenset((a, b)))

    while emb_e != all_e:
        # строим еще не уложенные сегменты
        segs = []

        # 1) отдельные ребра между уже уложенными вершинами
        for e in all_e - emb_e:
            a, b = tuple(e)
            if a in emb_v and b in emb_v:
                segs.append(({a, b}, set()))

        # 2) целые компоненты из неуложенных вершин
        seen = set()
        for s in adj:
            if s in emb_v or s in seen:
                continue
            comp = set()
            stack = [s]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                for y in adj[x]:
                    if y not in emb_v and y not in comp:
                        stack.append(y)
            seen |= comp
            contacts = set()
            for x in comp:
                for y in adj[x]:
                    if y in emb_v:
                        contacts.add(y)
            segs.append((contacts, comp))

        # ищем сегмент с наименьшим числом подходящих граней
        best = None
        for c, comp in segs:
            ok = []
            for i in range(len(faces)):
                if c <= set(faces[i]):
                    ok.append(i)
            if len(ok) == 0:
                return None  # некуда класть - граф не планарен
            if best is None or len(ok) < len(best[2]):
                best = (c, comp, ok)
            if len(ok) == 1:
                break

        contacts, comp, ok = best
        fi = ok[0]

        # строим путь внутри сегмента между двумя контактами
        a = next(iter(contacts))
        if len(comp) == 0:
            b = next(iter(contacts - {a}))
            path = [a, b]
        else:
            path = None
            queue = []
            for x in adj[a]:
                if x in comp:
                    queue.append([a, x])
            while queue and path is None:
                cur = queue.pop(0)
                last = cur[-1]
                for y in adj[last]:
                    if y in comp and y not in cur:
                        queue.append(cur + [y])
                    elif y in emb_v and y != a:
                        path = cur + [y]
                        break
        b = path[-1]

        # разрезаем выбранную грань на две
        f = faces[fi]
        k = f.index(a)
        f = f[k:] + f[:k]
        j = f.index(b)
        f1 = f[:j + 1] + path[-2:0:-1]
        f2 = [a] + path[1:-1] + f[j:]
        faces[fi] = f1
        faces.append(f2)

        emb_v |= set(path)
        for i in range(len(path) - 1):
            emb_e.add(frozenset((path[i], path[i + 1])))

    return faces


def find_blocks(adj):
    # разбиваем граф на блоки - алгоритм Тарьяна
    disc = {}
    low = {}
    timer = [0]
    edge_stack = []
    blocks = []

    def dfs(v, par_edge):
        disc[v] = timer[0]
        low[v] = timer[0]
        timer[0] += 1
        for w in adj[v]:
            e = frozenset((v, w))
            if e == par_edge:
                continue  # не возвращаемся сразу по тому же ребру к родителю
            if w not in disc:
                edge_stack.append((v, w))
                dfs(w, e)
                low[v] = min(low[v], low[w])
                if low[w] >= disc[v]:  # точка сочленения - снимаем со стека ребер один блок целиком
                    block = []
                    while True:
                        edge = edge_stack.pop()
                        block.append(edge)
                        if edge == (v, w):
                            break
                    blocks.append(block)
            elif disc[w] < disc[v]:   # обратное ребро - в стек, используется при подсчёте low
                edge_stack.append((v, w))
                low[v] = min(low[v], disc[w])

    for s in adj:
        if s not in disc:
            dfs(s, None)

    return blocks


def check_planar(adj):
    # проверка планарности произвольного графа
    blocks_info = []
    is_planar = True
    for block_edges in find_blocks(adj):
        if len(block_edges) == 1:
            blocks_info.append((block_edges, None))  # мост планарен всегда
            continue
        block_adj = {}
        for a, b in block_edges:
            if a not in block_adj:
                block_adj[a] = []
            if b not in block_adj:
                block_adj[b] = []
            block_adj[a].append(b)
            block_adj[b].append(a)
        faces = planar_faces(block_adj)
        if faces is None:
            is_planar = False
        blocks_info.append((block_edges, faces))
    return is_planar, blocks_info


def draw_planar(adj, faces, filename):
    # укладка Татта
    outer = max(faces, key=len)
    vs = list(adj)
    idx = {}
    for i, v in enumerate(vs):
        idx[v] = i
    n = len(vs)

    pos = np.zeros((n, 2))
    fixed = set(outer)
    for k, v in enumerate(outer):
        ang = 2 * np.pi * k / len(outer)
        pos[idx[v]] = (np.cos(ang), np.sin(ang))

    A = np.zeros((n, n))
    B = np.zeros((n, 2))
    for v in vs:
        i = idx[v]
        A[i, i] = 1
        if v in fixed:
            B[i] = pos[i]
        else:
            for w in adj[v]:
                A[i, idx[w]] -= 1.0 / len(adj[v])
    pos = np.linalg.solve(A, B)

    fig, ax = plt.subplots(figsize=(6, 6))
    for v in vs:
        for w in adj[v]:
            if idx[v] < idx[w]:
                xs = [pos[idx[v]][0], pos[idx[w]][0]]
                ys = [pos[idx[v]][1], pos[idx[w]][1]]
                ax.plot(xs, ys, "k-", zorder=1)
    for v in vs:
        x, y = pos[idx[v]]
        ax.scatter(x, y, s=400, c="lightblue", edgecolors="k", zorder=2)
        ax.text(x, y, str(v), ha="center", va="center", zorder=3)
    ax.axis("equal")
    ax.axis("off")
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def draw_tree(adj, filename):
    # рисуем дерево, листья слева направо - x, родитель - над серединой своих детей
    root = next(iter(adj))
    parent = {root: None}
    order = [root]
    i = 0
    while i < len(order):
        v = order[i]
        i += 1
        for w in adj[v]:
            if w not in parent:
                parent[w] = v
                order.append(w)

    children = {}
    for v in adj:
        children[v] = []
    for v in order[1:]:
        children[parent[v]].append(v)

    depth = {root: 0}
    for v in order[1:]:
        depth[v] = depth[parent[v]] + 1

    x = {}
    next_x = [0]

    def place(v):
        if len(children[v]) == 0:
            x[v] = next_x[0]
            next_x[0] += 1
        else:
            for w in children[v]:
                place(w)
            xs = [x[w] for w in children[v]]
            x[v] = sum(xs) / len(xs)

    place(root)

    fig, ax = plt.subplots(figsize=(6, 6))
    for v in order[1:]:
        p = parent[v]
        ax.plot([x[v], x[p]], [-depth[v], -depth[p]], "k-", zorder=1)
    for v in order:
        ax.scatter(x[v], -depth[v], s=400, c="lightblue", edgecolors="k", zorder=2)
        ax.text(x[v], -depth[v], str(v), ha="center", va="center", zorder=3)
    ax.axis("equal")
    ax.axis("off")
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def make_graph(edge_list):
    adj = {}
    for a, b in edge_list:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)
    return adj


def check(name, edge_list, filename):
    adj = make_graph(edge_list)
    is_planar, blocks_info = check_planar(adj)
    if not is_planar:
        print(name, "- не планарен")
        return

    print(name, "- планарен, блоков:", len(blocks_info))

    # draw_planar умеет рисовать только двусвязный блок
    num = 0
    for edges, faces in blocks_info:
        if faces is None:
            continue
        block_adj = {}
        for a, b in edges:
            if a not in block_adj:
                block_adj[a] = []
            if b not in block_adj:
                block_adj[b] = []
            block_adj[a].append(b)
            block_adj[b].append(a)
        if num == 0:
            fname = filename
        else:
            fname = filename.split(".")[0] + "_" + str(num) + ".png"
        draw_planar(block_adj, faces, fname)
        num += 1

    if num == 0:
        # циклов нет вообще - это дерево, рисуем его отдельной раскладкой
        draw_tree(adj, filename)


cube = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)]
k5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
k33 = [(i, j) for i in range(3) for j in range(3, 6)]
k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
c4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
tree = [(0, 1), (1, 2)]

# если граф не планарен, то файл не создается
check("Куб", cube, "planar_cube.png")
check("K5", k5, "planar_k5.png")
check("K3,3", k33, "planar_k3,3.png")
check("Тетраэдр", k4, "planar_k4.png")
check("Цикл", c4, "planar_c4.png")
check("Дерево", tree, "planar_tree.png")