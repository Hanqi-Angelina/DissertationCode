import os
import numpy as np
import pandas as pd
from graphviz import Digraph

def build_cpadag_directed_only(
    A, S,
    # thresholds
    thr_dir_strong=0.6, thr_dir_mid=0.5,
    # colors
    col_strong="#E78019",   # strong direction (> thr_dir_strong)
    col_mid="#296ca2",      # mid direction (thr_dir_mid .. thr_dir_strong]
    col_base="#a5a4a4",     # baseline gray
    # layout / appearance
    engine="dot", rankdir="LR", splines="line",   # straight edges
    nodesep=0.9, ranksep=1.1,                 
    node_fontsize=16, node_size=0.75, fontname="Helvetica",
    base_penwidth=1.0, mid_penwidth=2.8, strong_penwidth=3.6,
    arrowhead="vee", arrowsize_base=1.05, arrowsize_mid=1.2, arrowsize_strong=1.35,
    latent_prefix="L", dpi=300,
    # labels
    underscore_mode="keep",  # "keep" | "space" | "remove"  (affects labels only)
    wrap_width=None,       
    # page fitting
    fit_to_page=True,
    page="A4",               # "A4" or "Letter"
    orientation="landscape", # "landscape" or "portrait"
    margin_in=0.25           # page margin (inches)
):
    """
    Draw a CPDAG with only directed-edge coloring.

    Parameters
    ----------
    A : pandas.DataFrame
        CPDAG adjacency. Encoding:
        u->v: A[u,v] = -1 and A[v,u] = 1; undirected u--v: -1/-1; none: 0/0.
    S : pandas.DataFrame
        Summary with columns at least ['u','v','edge_type','prop'] where
        edge_type in {'u->v','v->u'} and 'prop' is the selection probability.
    """

    # --- sanitize A ---
    if not isinstance(A, pd.DataFrame):
        raise TypeError("A must be a pandas.DataFrame with CPDAG adjacency.")
    A_df = A.copy()
    A_df.index = A_df.index.astype(str).str.strip()
    A_df.columns = A_df.columns.astype(str).str.strip()
    names = list(A_df.columns)
    A_np = A_df.values
    n = len(names)

    # --- sanitize S & build directional prop map ---
    S = S.copy()
    for c in ("u", "v"):
        S[c] = S[c].astype(str).str.strip()
    S["prop"] = pd.to_numeric(S.get("prop", 0.0), errors="coerce").fillna(0.0)

    # (from,to) -> prop for both 'u->v' and 'v->u'
    S_dir = S[S["edge_type"].isin(["u->v", "v->u"])].copy()
    S_dir["from"] = np.where(S_dir["edge_type"] == "u->v", S_dir["u"], S_dir["v"])
    S_dir["to"]   = np.where(S_dir["edge_type"] == "u->v", S_dir["v"], S_dir["u"])
    p_dir = {(r["from"], r["to"]): float(r["prop"]) for _, r in S_dir.iterrows()}

    # --- label formatter (labels only; IDs unchanged) ---
    def format_label(s: str) -> str:
        s = str(s)
        if underscore_mode == "space":
            s = s.replace("_", " ")
        elif underscore_mode == "remove":
            s = s.replace("_", "")
        if wrap_width and wrap_width > 0:
            s = "\n".join(s[i:i+wrap_width] for i in range(0, len(s), wrap_width))
        return s

    # --- Graphviz setup ---
    g = Digraph(engine=engine)
    g.attr("graph",
           splines=splines,
           overlap="false",
           nodesep=str(nodesep),
           ranksep=str(ranksep),
           rankdir=rankdir,
           dpi=str(dpi))
    g.attr("node",
           shape="circle",
           style="filled",
           fillcolor="white",
           color="#555555",
           fontsize=str(node_fontsize),
           fontname=fontname,
           width=str(node_size),
           height=str(node_size),
           fixedsize="false")
    g.attr("edge",
           color=col_base,
           penwidth=str(base_penwidth),
           arrowsize=str(arrowsize_base),
           arrowhead=arrowhead,
           fontname=fontname)

    for v in names:
        fill = "#f5e6e6" if (latent_prefix and isinstance(v, str) and v.startswith(latent_prefix)) else "white"
        g.node(v, label=format_label(v), fillcolor=fill)

    # --- draw each unordered pair once according to CPDAG orientation ---
    for i in range(n):
        for j in range(i + 1, n):
            u, v = names[i], names[j]
            a, b = A_np[i, j], A_np[j, i]
            if a == 0 and b == 0:
                continue

            # base u -> v
            if a == -1 and b == 1:
                p = p_dir.get((u, v), 0.0)
                if p > thr_dir_strong:
                    g.edge(u, v, color=col_strong, penwidth=str(strong_penwidth),
                           arrowsize=str(arrowsize_strong), arrowhead=arrowhead)
                elif p > thr_dir_mid:
                    g.edge(u, v, color=col_mid, penwidth=str(mid_penwidth),
                           arrowsize=str(arrowsize_mid), arrowhead=arrowhead)
                else:
                    g.edge(u, v)  # thin baseline
                continue

            # base v -> u
            if a == 1 and b == -1:
                p = p_dir.get((v, u), 0.0)
                if p > thr_dir_strong:
                    g.edge(v, u, color=col_strong, penwidth=str(strong_penwidth),
                           arrowsize=str(arrowsize_strong), arrowhead=arrowhead)
                elif p > thr_dir_mid:
                    g.edge(v, u, color=col_mid, penwidth=str(mid_penwidth),
                           arrowsize=str(arrowsize_mid), arrowhead=arrowhead)
                else:
                    g.edge(v, u)
                continue

            # base undirected (-1,-1): draw thin gray, no coloring
            if a == -1 and b == -1:
                g.edge(u, v, dir="none")
                continue

            # any other combination -> thin gray
            g.edge(u, v, dir="none")

    # --- fit to one page (optional) ---
    if fit_to_page:
        if page.lower() == "a4":
            w, h = 8.27, 11.69
        elif page.lower() == "letter":
            w, h = 8.5, 11.0
        else:
            w, h = 8.27, 11.69  # default A4
        if orientation.lower() == "landscape":
            w, h = max(w, h), min(w, h)
        else:
            w, h = min(w, h), max(w, h)
        g.graph_attr.update(size=f"{w},{h}!", margin=str(margin_in), center="true", ratio="compress")

    return g

based_input_path = f'{os.getcwd()}/DataAnalysis' # should work with correct cwd, sub in absolute path otherwise
A = pd.read_csv(f"{based_input_path}/MainResults/worry_sleep_mistrustB_CPDAG.csv", index_col=0)
S = pd.read_csv(f"{based_input_path}/MainResults/mean_subsample_results.csv")
G = build_cpadag_directed_only(
    A, S,
    # layout
    engine="dot", rankdir="LR", splines="line",
    nodesep=0.85, ranksep=1.0,
    node_fontsize=16, node_size=0.70,
    arrowsize_base=1.05, arrowsize_mid=1.2, arrowsize_strong=1.35,
    # labels & underscores
    underscore_mode="remove",  # "keep" | "space" | "remove"
    wrap_width=10,
    # force one page
    fit_to_page=True, page="A4", orientation="landscape", margin_in=0.25
)
G.render("cpadag_onepage", format="pdf", cleanup=True)

