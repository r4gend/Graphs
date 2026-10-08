import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def decode_tree(code):
    # 0 - шаг вниз (новая вершина), 1 - шаг вверх к родителю, возвращаем (сколько вершин, список ребер)
    par = [None]
    edges = []
    cur = 0
    for c in code:
        if c == "0":
            v = len(par)
            par.append(cur)
            edges.append((cur, v))
            cur = v
        elif c == "1":
            if cur == 0:
                raise ValueError("Выход выше корня")
            cur = par[cur]
        else:
            raise ValueError("В коде не только 0 и 1")
    if cur != 0:
        raise ValueError("Обход не вернулся в корень")
    return len(par), edges


def export_dot(n, edges, filename):
    # экспорт дерева в дот формат
    f = open(filename, "w")
    f.write("graph Tree {\n")
    for v in range(n):
        f.write("    %d;\n" % v)
    for a, b in edges:
        f.write("    %d -- %d;\n" % (a, b))
    f.write("}\n")
    f.close()


def draw_tree(n, edges, filename):
    # считаем координаты: листья в ряд, родитель - среднее детей
    ch = [[] for _ in range(n)]
    for a, b in edges:
        ch[a].append(b)
    pos = {}
    cnt = [0]  # следующий свободный x для листа

    def go(v, d):
        if len(ch[v]) == 0:
            pos[v] = (cnt[0], -d)
            cnt[0] += 1
        else:
            for c in ch[v]:
                go(c, d + 1)
            xs = [pos[c][0] for c in ch[v]]
            pos[v] = (sum(xs) / len(xs), -d)

    go(0, 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "k-")
    for v in pos:
        x, y = pos[v]
        ax.scatter(x, y, s=500, c="lightblue", edgecolors="k", zorder=2)
        ax.text(x, y, str(v), ha="center", va="center", zorder=3)
    ax.axis("off")
    fig.savefig(filename, dpi=120)
    plt.close(fig)


code = "0100101011001011"  # 16 бит, 9 вершин
n, edges = decode_tree(code)
print("Вершин:", n)
print("Ребра:", edges)
export_dot(n, edges, "tree.dot")
draw_tree(n, edges, "tree_manual.png")
