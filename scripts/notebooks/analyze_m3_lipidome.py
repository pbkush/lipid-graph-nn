# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "umap-learn",
#     "hdbscan",
#     "torch",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import warnings
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import marimo as mo

    warnings.filterwarnings("ignore")

    REPO = Path(__file__).resolve().parents[2]
    ITP_DIR_PRIMARY = REPO / "resources" / "martini3" / "itp"
    ITP_DIR_STEROLS = REPO / "emil_extra" / "simulation_parameters" / "toppar"
    FF_PARAMS_PATH = REPO / "resources" / "martini_ff_params.json"
    DATA_DIR = REPO / "data" / "membrane_only"
    FIG_DIR = REPO / "results" / "figures" / "m3_lipidome"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    CURRENT_LIPIDS = [
        "POPC", "DOPC", "DIPC", "DPPC",
        "POPE", "DOPE", "DPPE",
        "DOPS", "POPS",
        "CHOL",
    ]

    def save_fig(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=150)
        return fig

    return (
        AgglomerativeClustering,
        CURRENT_LIPIDS,
        Counter,
        DATA_DIR,
        FF_PARAMS_PATH,
        FIG_DIR,
        ITP_DIR_PRIMARY,
        ITP_DIR_STEROLS,
        PCA,
        REPO,
        StandardScaler,
        json,
        mo,
        np,
        pd,
        plt,
        re,
        save_fig,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # M3 Lipidome Analysis

        Characterisation of the Martini 3 lipid library before any new simulations.
        Plan: [`docs/m3_lipidome_analysis_plan.md`](../../docs/m3_lipidome_analysis_plan.md).

        Two layers:
        - **(A) Lipid space** — each lipid is a point with descriptor vectors built
          from ITP/INSANE metadata, bead composition, and bead physics.
        - **(B) Composition space** — each membrane is a mole-fraction-weighted
          centroid of its lipids in the lipid-space embedding.

        The 70 currently-simulated compositions and the 10-lipid training pool are
        marked on every relevant figure.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Section 0 — ITP inventory")
    return


@app.cell
def _(Counter, ITP_DIR_PRIMARY, ITP_DIR_STEROLS):
    def _classify_source(filename):
        f = filename.lower()
        if "etherphospholipids" in f: return "ether"
        if "plasmalogens" in f:       return "plasmalogen"
        if "phospholipids_sm" in f:   return "SM"
        if "phospholipids_cl" in f:   return "CL"
        if "phospholipids_pc" in f:   return "PC"
        if "phospholipids_pe" in f:   return "PE"
        if "phospholipids_ps" in f:   return "PS"
        if "phospholipids_pa" in f:   return "PA"
        if "phospholipids_pg" in f:   return "PG"
        if "phospholipids_pi" in f:   return "PI"
        if "ceramides" in f:          return "ceramide"
        if "diglycerides" in f:       return "DG"
        if "monoglycerides" in f:     return "MG"
        if "triglycerides" in f:      return "TG"
        if "dotap" in f:              return "DOTAP"
        if "bmp" in f:                return "BMP"
        if "sterol" in f:             return "sterol"
        return "other"

    def _parse_insane(line):
        m = {}
        for part in line.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                m[k.strip().lstrip(";").lstrip("@INSANE ")] = v.strip()
        return m

    def _parse_itp(path):
        text = path.read_text()
        records = []
        pending_insane = None
        i = 0
        lines = text.splitlines()
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith(";@INSANE"):
                pending_insane = _parse_insane(stripped)
            elif stripped.startswith("[") and "moleculetype" in stripped:
                j = i + 1
                molname = None
                while j < len(lines):
                    s = lines[j].strip()
                    if s and not s.startswith(";"):
                        molname = s.split()[0]
                        break
                    j += 1
                atoms = []
                in_atoms = False
                k = j
                while k < len(lines):
                    s = lines[k].strip()
                    if s.startswith("[") and "atoms" in s:
                        in_atoms = True
                        k += 1
                        continue
                    if in_atoms:
                        if s.startswith("["):
                            break
                        if s and not s.startswith(";"):
                            toks = s.split()
                            if len(toks) >= 7:
                                atoms.append({
                                    "id":      int(toks[0]),
                                    "type":    toks[1],
                                    "residue": toks[3],
                                    "atom":    toks[4],
                                    "charge":  float(toks[6]),
                                })
                    k += 1
                if molname and atoms:
                    records.append({
                        "molname":     molname,
                        "source":      path.name,
                        "family":      _classify_source(path.name),
                        "atoms":       atoms,
                        "insane":      pending_insane or {},
                    })
                pending_insane = None
                i = k
                continue
            i += 1
        return records

    _files = sorted(ITP_DIR_PRIMARY.glob("martini_v3.0.0_*.itp"))
    _sterol_file = ITP_DIR_STEROLS / "martini_v3.0_sterols_v1.0.itp"
    if _sterol_file.exists():
        _files.append(_sterol_file)
    _keep = []
    for _f in _files:
        _name = _f.name.lower()
        if any(s in _name for s in ("ffbonded", "ions", "solvents", "fattyacids", "hydrocarbons")):
            continue
        _keep.append(_f)
    raw_lipids = []
    for _f in _keep:
        raw_lipids.extend(_parse_itp(_f))

    _family_counts = Counter(r["family"] for r in raw_lipids)
    print(f"Parsed {len(raw_lipids)} molecules from {len({r['source'] for r in raw_lipids})} ITP files")
    print(f"Families: {dict(_family_counts)}")
    return (raw_lipids,)


@app.cell
def _(CURRENT_LIPIDS, FF_PARAMS_PATH, json, np, pd, raw_lipids):
    ff_params = json.loads(FF_PARAMS_PATH.read_text())

    BILAYER_FAMILIES = {"PC", "PE", "PS", "PA", "PG", "PI", "SM", "CL",
                        "ether", "plasmalogen", "sterol", "ceramide",
                        "DOTAP", "BMP"}

    def _features_from_record(rec):
        ins = rec["insane"]
        atoms = rec["atoms"]
        head_tokens = ins.get("alhead", "").split()
        link_tokens = ins.get("allink", "").split()
        tail_tokens = ins.get("altail", "").split()

        n_beads = len(atoms)
        total_charge = sum(a["charge"] for a in atoms)
        try:
            insane_charge = float(ins.get("charge", "0").strip())
        except ValueError:
            insane_charge = total_charge

        tail_lengths = [len(t.replace("t", "")) for t in tail_tokens]
        unsat_counts = [sum(1 for c in t if c == "D") for t in tail_tokens]

        return {
            "molname":         rec["molname"],
            "family":          rec["family"],
            "source":          rec["source"],
            "n_beads":         n_beads,
            "n_tails":         len(tail_tokens),
            "tail_len_mean":   float(np.mean(tail_lengths)) if tail_lengths else 0.0,
            "tail_len_total":  int(sum(tail_lengths)),
            "tail_len_asym":   (max(tail_lengths) - min(tail_lengths)) if tail_lengths else 0,
            "n_double_bonds":  int(sum(unsat_counts)),
            "frac_unsat":      float(sum(unsat_counts) / max(1, sum(tail_lengths))),
            "n_head_beads":    len(head_tokens),
            "n_link_beads":    len(link_tokens),
            "total_charge":    float(total_charge),
            "insane_charge":   insane_charge,
            "bead_types":      [a["type"] for a in atoms],
            "has_insane":      bool(ins),
        }

    _rows = [_features_from_record(r) for r in raw_lipids]
    lipid_df = pd.DataFrame(_rows)
    lipid_df = lipid_df[lipid_df["family"].isin(BILAYER_FAMILIES)].reset_index(drop=True)
    lipid_df["is_current"] = lipid_df["molname"].isin(CURRENT_LIPIDS)

    print(f"Lipid count after bilayer-only filter: {len(lipid_df)}")
    print(f"Of which currently simulated: {lipid_df['is_current'].sum()} / {len(CURRENT_LIPIDS)} target")
    _missing = sorted(set(CURRENT_LIPIDS) - set(lipid_df["molname"]))
    if _missing:
        print(f"Current lipids missing from M3 ITPs: {_missing}")
    return ff_params, lipid_df


@app.cell
def _(lipid_df, mo, np, pd, plt, save_fig):
    _fam_order = lipid_df["family"].value_counts().sort_values(ascending=False).index.tolist()
    _counts = lipid_df["family"].value_counts().reindex(_fam_order)
    _current_in_fam = lipid_df.groupby("family")["is_current"].sum().reindex(_fam_order).fillna(0).astype(int)

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _x = np.arange(len(_fam_order))
    _ax.bar(_x, _counts.values, color="#888", label="M3 lipids")
    _ax.bar(_x, _current_in_fam.values, color="#c0392b", label="currently simulated")
    _ax.set_xticks(_x); _ax.set_xticklabels(_fam_order, rotation=30, ha="right")
    _ax.set_xlabel("headgroup / linker family")
    _ax.set_ylabel("number of lipids")
    _ax.set_title(f"M3 lipid count per family ({len(lipid_df)} bilayer-forming lipids, {lipid_df['is_current'].sum()} in current training pool)")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    save_fig(_fig, "fig00_family_inventory")

    _summary = pd.DataFrame({
        "n_lipids": _counts,
        "in_current_pool": _current_in_fam,
    }).reset_index().rename(columns={"index": "family"})
    mo.vstack([_fig, _summary])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 1 — Lipid feature representations

        Three descriptor types built and compared in parallel.

        1. **Structural** — family one-hot, bead/tail counts, tail length and
           asymmetry, unsaturation, head/link bead counts, net charge.
        2. **Bead composition** — count of each Martini 3 bead type per lipid.
        3. **Bead physics** — mean + sum of `[mass, charge, σ, ε]` over beads.
        """
    )
    return


@app.cell
def _(StandardScaler, ff_params, lipid_df, np, pd):
    _struct_cols = [
        "n_beads", "n_tails", "tail_len_mean", "tail_len_total",
        "tail_len_asym", "n_double_bonds", "frac_unsat",
        "n_head_beads", "n_link_beads", "total_charge",
    ]
    _family_dummies = pd.get_dummies(lipid_df["family"], prefix="fam").astype(float)
    _structural = pd.concat([lipid_df[_struct_cols].astype(float), _family_dummies], axis=1)
    X_structural = StandardScaler().fit_transform(_structural.values)

    _all_beads = sorted({b for beads in lipid_df["bead_types"] for b in beads})
    _bead_to_idx = {b: i for i, b in enumerate(_all_beads)}
    _bead_count_mat = np.zeros((len(lipid_df), len(_all_beads)), dtype=float)
    for _i, _beads in enumerate(lipid_df["bead_types"]):
        for _b in _beads:
            _bead_count_mat[_i, _bead_to_idx[_b]] += 1
    X_beadcomp = StandardScaler().fit_transform(_bead_count_mat)

    _phys_keys = ["mass", "charge", "sigma", "epsilon"]
    _phys_rows = []
    for _beads in lipid_df["bead_types"]:
        _vals = []
        for _k in _phys_keys:
            _seq = [ff_params[_b][_k] for _b in _beads if _b in ff_params]
            if not _seq:
                _vals += [0.0, 0.0]
                continue
            _vals += [float(np.mean(_seq)), float(np.sum(_seq))]
        _phys_rows.append(_vals)
    X_physics = StandardScaler().fit_transform(np.array(_phys_rows))

    descriptors = {
        "structural": X_structural,
        "beadcomp":   X_beadcomp,
        "physics":    X_physics,
    }
    _missing_ff = sorted({b for b in _all_beads if b not in ff_params})
    if _missing_ff:
        print(f"WARNING: bead types missing from ff_params (zeroed out): {_missing_ff}")
    print({k: v.shape for k, v in descriptors.items()})
    return (descriptors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 2 — Dimensionality reduction & clustering

        PCA + UMAP per descriptor; HDBSCAN on the PCA-reduced points.
        Ward hierarchical clustering on the structural descriptor seeds the
        composition-space archetypes downstream.
        """
    )
    return


@app.cell
def _(PCA, descriptors, pd):
    try:
        import umap
        HAS_UMAP = True
    except Exception as _e:
        print(f"umap-learn not available: {_e}")
        HAS_UMAP = False

    try:
        import hdbscan
        HAS_HDBSCAN = True
    except Exception as _e:
        print(f"hdbscan not available: {_e}")
        HAS_HDBSCAN = False

    embeds = {}
    for _name, _X in descriptors.items():
        _pca = PCA(n_components=min(10, _X.shape[1])).fit(_X)
        _Xp = _pca.transform(_X)
        _out = {
            "pca_full":  _Xp,
            "pca_2d":    _Xp[:, :2],
            "explained": _pca.explained_variance_ratio_,
        }
        if HAS_UMAP:
            _out["umap_2d"] = umap.UMAP(
                n_neighbors=15, min_dist=0.1, random_state=0
            ).fit_transform(_X)
        if HAS_HDBSCAN:
            _out["hdb_labels"] = hdbscan.HDBSCAN(
                min_cluster_size=5, min_samples=3
            ).fit_predict(_Xp)
        embeds[_name] = _out

    _expl = pd.DataFrame(
        {
            k: list(v["explained"][:5]) + [None] * (5 - len(v["explained"][:5]))
            for k, v in embeds.items()
        },
        index=[f"PC{i+1}" for i in range(5)],
    )
    print("PCA explained variance (first 5 PCs):")
    print(_expl.round(3))
    return HAS_HDBSCAN, HAS_UMAP, embeds


@app.cell
def _(embeds, lipid_df, plt, save_fig):
    _fams = sorted(lipid_df["family"].unique())
    _cmap = plt.cm.tab20
    _color_for = {f: _cmap(i / max(1, len(_fams) - 1)) for i, f in enumerate(_fams)}

    _keys = [k for k in ["pca_2d", "umap_2d"] if any(k in v for v in embeds.values())]
    _ncol = len(embeds)
    _fig, _axes = plt.subplots(
        len(_keys), _ncol, figsize=(4.5 * _ncol, 4.2 * len(_keys)), squeeze=False,
    )

    for _col, (_dname, _info) in enumerate(embeds.items()):
        for _row, _ek in enumerate(_keys):
            _ax = _axes[_row][_col]
            if _ek not in _info:
                _ax.axis("off")
                continue
            _X2 = _info[_ek]
            for _f in _fams:
                _mask = (lipid_df["family"] == _f).values
                _ax.scatter(
                    _X2[_mask, 0], _X2[_mask, 1], s=12, alpha=0.55,
                    color=_color_for[_f],
                    label=_f if (_row == 0 and _col == 0) else None,
                )
            _cur = lipid_df["is_current"].values
            if _cur.any():
                _ax.scatter(
                    _X2[_cur, 0], _X2[_cur, 1], s=80, facecolor="none",
                    edgecolor="black", linewidth=1.4,
                    label="training pool" if (_row == 0 and _col == 0) else None,
                )
            _ax.set_title(f"{_dname} — {_ek}")
            _ax.set_xlabel("dim 1"); _ax.set_ylabel("dim 2")

    if _axes[0][0].get_legend_handles_labels()[0]:
        _axes[0][0].legend(
            bbox_to_anchor=(0, 1.25), loc="lower left", ncol=6,
            fontsize=7, frameon=False,
        )
    _fig.suptitle(
        "Lipid-space 2D embeddings (colour = family, ring = current training pool)",
        y=1.02,
    )
    _fig.tight_layout()
    save_fig(_fig, "fig01_lipid_embeddings_by_descriptor")
    return


@app.cell
def _(Counter, HAS_HDBSCAN, embeds, mo, pd):
    if not HAS_HDBSCAN:
        _ = mo.callout("HDBSCAN unavailable — skipping cluster summary.", kind="warn")
    else:
        _rows = []
        for _dname, _info in embeds.items():
            _labels = _info.get("hdb_labels")
            if _labels is None:
                continue
            _cnt = Counter(_labels)
            _rows.append({
                "descriptor": _dname,
                "n_clusters": sum(1 for k in _cnt if k != -1),
                "n_noise":    _cnt.get(-1, 0),
                "largest":    max((v for k, v in _cnt.items() if k != -1), default=0),
            })
        print(pd.DataFrame(_rows).to_string(index=False))
    return


@app.cell
def _(AgglomerativeClustering, descriptors, lipid_df, np, plt, save_fig):
    from scipy.cluster.hierarchy import dendrogram, linkage

    _X = descriptors["structural"]
    _Z = linkage(_X, method="ward")

    _fig, _ax = plt.subplots(figsize=(13, 4.5))
    dendrogram(
        _Z, ax=_ax, labels=lipid_df["molname"].values,
        color_threshold=0.5 * np.max(_Z[:, 2]),
        leaf_font_size=5, leaf_rotation=90,
    )
    _ax.set_title("Ward hierarchical clustering on structural descriptors")
    _ax.set_ylabel("linkage distance")
    _fig.tight_layout()
    save_fig(_fig, "fig02_ward_dendrogram_structural")

    K_ARCHETYPES = 8
    archetype_labels = AgglomerativeClustering(
        n_clusters=K_ARCHETYPES, linkage="ward"
    ).fit_predict(_X)

    _members = (
        lipid_df.assign(archetype=archetype_labels)
        .groupby("archetype")["molname"]
        .apply(lambda s: ", ".join(sorted(s.tolist())[:8]) + (" …" if len(s) > 8 else ""))
    )
    print(f"Ward → {K_ARCHETYPES} archetypes. Member previews:")
    print(_members.to_string())
    return K_ARCHETYPES, archetype_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 3 — Composition space

        A composition is a mole-fraction vector over lipids. The natural
        fixed-length coordinate is the **mole-fraction-weighted centroid** of
        its lipids in the lipid PCA space, independent of how many lipids the
        membrane contains.
        """
    )
    return


@app.cell
def _(DATA_DIR, pd, re):
    _pat = re.compile(r"([A-Z]+)(\d+)")

    def _parse_comp(name):
        matches = _pat.findall(name)
        if not matches:
            return None
        comp = {}
        for lip, pct in matches:
            comp[lip] = comp.get(lip, 0.0) + float(pct) / 100.0
        s = sum(comp.values())
        if s == 0:
            return None
        return {k: v / s for k, v in comp.items()}

    _records = []
    for _d in sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()):
        _c = _parse_comp(_d)
        if _c is None:
            print(f"could not parse composition for {_d}")
            continue
        _records.append({"name": _d, "fractions": _c})

    sim_df = pd.DataFrame(_records)
    print(f"Parsed {len(sim_df)} simulated compositions")
    print(f"Unique lipids used: {sorted({l for c in sim_df['fractions'] for l in c})}")
    return (sim_df,)


@app.cell
def _(embeds, lipid_df, np):
    lipid_coords = embeds["structural"]["pca_full"]
    name_to_idx = {n: i for i, n in enumerate(lipid_df["molname"].tolist())}

    def composition_to_centroid(fractions):
        vec = np.zeros(lipid_coords.shape[1])
        total = 0.0
        for lip, frac in fractions.items():
            idx = name_to_idx.get(lip)
            if idx is None:
                continue
            vec += frac * lipid_coords[idx]
            total += frac
        if total == 0:
            return None
        return vec / total

    return (composition_to_centroid,)


@app.cell
def _(CURRENT_LIPIDS, composition_to_centroid, np):
    _comps = []
    for _lip in CURRENT_LIPIDS:
        _comps.append({"name": f"{_lip}100", "fractions": {_lip: 1.0}, "kind": "pure"})

    _steps = [0.1 * i for i in range(1, 10)]
    _seen = set()
    for _i, _a in enumerate(CURRENT_LIPIDS):
        for _b in CURRENT_LIPIDS[_i + 1:]:
            for _f in _steps:
                _key = tuple(sorted([(_a, round(_f, 2)), (_b, round(1 - _f, 2))]))
                if _key in _seen:
                    continue
                _seen.add(_key)
                _comps.append({
                    "name": f"{_a}{int(round(_f * 100))}_{_b}{int(round((1 - _f) * 100))}",
                    "fractions": {_a: _f, _b: 1 - _f},
                    "kind": "binary",
                })

    _rng = np.random.default_rng(0)
    for _ in range(2000):
        _k = int(_rng.integers(2, 6))
        _lips = list(_rng.choice(CURRENT_LIPIDS, size=_k, replace=False))
        _ws = _rng.dirichlet(np.ones(_k))
        _frac = {l: float(w) for l, w in zip(_lips, _ws) if w > 0.05}
        if not _frac:
            continue
        _s = sum(_frac.values())
        _frac = {k_: v / _s for k_, v in _frac.items()}
        _comps.append({"name": None, "fractions": _frac, "kind": "dirichlet"})

    candidate_comps = []
    _cand_coords_list = []
    _cand_kinds_list = []
    for _c in _comps:
        _v = composition_to_centroid(_c["fractions"])
        if _v is None:
            continue
        candidate_comps.append(_c)
        _cand_coords_list.append(_v)
        _cand_kinds_list.append(_c["kind"])

    cand_coords = np.array(_cand_coords_list)
    _kc, _kn = np.unique(_cand_kinds_list, return_counts=True)
    print(f"Candidate compositions: {len(cand_coords)}  ({dict(zip(_kc, _kn))})")
    return cand_coords, candidate_comps


@app.cell
def _(cand_coords, composition_to_centroid, np, sim_df):
    _coords = []
    _names = []
    for _, _row in sim_df.iterrows():
        _v = composition_to_centroid(_row["fractions"])
        if _v is None:
            continue
        _coords.append(_v)
        _names.append(_row["name"])
    sim_coords = np.array(_coords)
    sim_names = _names
    all_coords = np.vstack([cand_coords, sim_coords])
    print(f"Composition cloud: {len(cand_coords)} candidates + {len(sim_coords)} simulated")
    return all_coords, sim_coords, sim_names


@app.cell
def _(PCA, all_coords, cand_coords, plt, save_fig, sim_coords):
    _cpca = PCA(n_components=2).fit(all_coords)
    cand_2d = _cpca.transform(cand_coords)
    sim_2d = _cpca.transform(sim_coords)

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _ax.scatter(cand_2d[:, 0], cand_2d[:, 1], s=6, alpha=0.25, color="#7f8c8d",
                label="candidate compositions")
    _ax.scatter(sim_2d[:, 0], sim_2d[:, 1], s=60, color="#c0392b",
                edgecolor="black", linewidth=0.5, label="simulated (70 systems)")
    _ax.set_xlabel("PC1 (composition centroid)")
    _ax.set_ylabel("PC2 (composition centroid)")
    _ax.set_title(
        f"Composition-space PCA — {len(cand_coords)} candidates + "
        f"{len(sim_coords)} simulated, training-pool mixtures only"
    )
    _ax.legend(frameon=False, loc="best")
    _fig.tight_layout()
    save_fig(_fig, "fig03_composition_pca")
    return cand_2d, sim_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Section 4 — Composition-space coverage""")
    return


@app.cell
def _(HAS_HDBSCAN, all_coords, mo, np, pd, sim_coords):
    if not HAS_HDBSCAN:
        _ = mo.callout("HDBSCAN unavailable — skipping coverage analysis.", kind="warn")
        comp_labels = None
        comp_centroids = None
        coverage_df = None
    else:
        import hdbscan as _hdbscan
        comp_labels = _hdbscan.HDBSCAN(
            min_cluster_size=30, min_samples=10
        ).fit_predict(all_coords)
        _n_cand = all_coords.shape[0] - sim_coords.shape[0]
        _sim_labels = comp_labels[_n_cand:]

        comp_centroids = {}
        _rows = []
        for _cl in sorted(set(comp_labels)):
            if _cl == -1:
                continue
            _members = all_coords[comp_labels == _cl]
            _centroid = _members.mean(axis=0)
            comp_centroids[_cl] = _centroid
            _n_sim = int((_sim_labels == _cl).sum())
            if _n_sim > 0:
                _sm = sim_coords[_sim_labels == _cl]
                _d = float(np.linalg.norm(_sm - _centroid, axis=1).min())
            else:
                _d = float("nan")
            _rows.append({
                "cluster":               _cl,
                "size":                  int((comp_labels == _cl).sum()),
                "n_simulated":           _n_sim,
                "nearest_sim_to_center": round(_d, 3),
            })
        coverage_df = pd.DataFrame(_rows).sort_values("size", ascending=False)
        print(coverage_df.to_string(index=False))
    return comp_centroids, comp_labels, coverage_df


@app.cell
def _(cand_2d, comp_labels, mo, plt, save_fig, sim_2d, sim_coords):
    if comp_labels is None:
        _ = mo.callout("Skipping cluster overlay (no labels).", kind="warn")
    else:
        _n_cand = cand_2d.shape[0]
        _cand_lab = comp_labels[:_n_cand]
        _sim_lab = comp_labels[_n_cand:_n_cand + sim_2d.shape[0]]
        _ids = sorted(set(_cand_lab) | set(_sim_lab))
        _palette = plt.cm.tab10

        _fig, _ax = plt.subplots(figsize=(7.5, 6))
        for _cl in _ids:
            _color = "lightgray" if _cl == -1 else _palette(_cl % 10)
            _mc = _cand_lab == _cl
            _ax.scatter(cand_2d[_mc, 0], cand_2d[_mc, 1], s=4, alpha=0.3, color=_color)
            _ms = _sim_lab == _cl
            if _ms.any():
                _ax.scatter(sim_2d[_ms, 0], sim_2d[_ms, 1], s=60, color=_color,
                            edgecolor="black", linewidth=0.6)
        _ = sim_coords  # ensure dependency for caching
        _ax.set_title("Composition clusters (HDBSCAN); simulated = ringed, candidates = dots")
        _ax.set_xlabel("PC1"); _ax.set_ylabel("PC2")
        _fig.tight_layout()
        save_fig(_fig, "fig04_composition_clusters")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 5 — Selection rules

        Two rules previewed: cluster-centroid pick and stratified shells at
        the 33rd, 66th, and 95th percentile of distance-to-centroid within
        each cluster. No simulations triggered.
        """
    )
    return


@app.cell
def _(cand_coords, candidate_comps, comp_centroids, comp_labels, mo, np, pd):
    if comp_centroids is None or not comp_centroids:
        _ = mo.callout("No clusters → skipping selection rule preview.", kind="warn")
        shortlist = None
    else:
        _n_cand = cand_coords.shape[0]
        _cand_lab = comp_labels[:_n_cand]
        _rows = []
        for _cl, _ctr in comp_centroids.items():
            _mask = _cand_lab == _cl
            if not _mask.any():
                continue
            _d = np.linalg.norm(cand_coords[_mask] - _ctr, axis=1)
            _local = int(np.argmin(_d))
            _gi = int(np.flatnonzero(_mask)[_local])
            _c = candidate_comps[_gi]
            _rows.append({
                "cluster":   _cl,
                "rule":      "centroid",
                "shell":     0,
                "distance":  float(_d[_local]),
                "fractions": _c["fractions"],
                "kind":      _c["kind"],
            })
            if _mask.sum() >= 5:
                _qs = np.quantile(_d, [0.33, 0.66, 0.95])
                for _shell, _qd in enumerate(_qs, start=1):
                    _idx = int(np.argmin(np.abs(_d - _qd)))
                    _gi2 = int(np.flatnonzero(_mask)[_idx])
                    _c2 = candidate_comps[_gi2]
                    _rows.append({
                        "cluster":   _cl,
                        "rule":      "shell",
                        "shell":     _shell,
                        "distance":  float(_d[_idx]),
                        "fractions": _c2["fractions"],
                        "kind":      _c2["kind"],
                    })

        shortlist = pd.DataFrame(_rows)
        shortlist["fraction_str"] = shortlist["fractions"].apply(
            lambda d: ", ".join(f"{k}:{v:.2f}" for k, v in sorted(d.items()))
        )
        print(f"Shortlist: {len(shortlist)} compositions "
              f"({shortlist['rule'].value_counts().to_dict()})")
        print(shortlist[["cluster", "rule", "shell", "distance", "fraction_str"]]
              .to_string(index=False))
    return (shortlist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 6 — GNN embedding probe (cheap tie-back to the model)

        Loads a trained checkpoint (`model_best.pt`, fall back to
        `model_final.pt`) and forwards a sample of the 70 simulated systems
        through the trunk. The descriptor-based composition embedding and
        the model's internal embedding can then be compared directly.

        Falls back to a notice if no checkpoint is present yet — re-run after
        a Tier C sweep submitted with the 2026-05-16 checkpoint code lands.
        """
    )
    return


@app.cell
def _(REPO, mo):
    import glob as _glob
    _candidates = sorted(_glob.glob(str(REPO / "logs" / "training" / "*" / "*" / "model_best.pt")))
    if not _candidates:
        _candidates = sorted(_glob.glob(str(REPO / "logs" / "training" / "*" / "*" / "model_final.pt")))
    if not _candidates:
        _ = mo.callout(
            "No `model_best.pt` or `model_final.pt` found under `logs/training/**`. "
            "Re-run a Tier C sweep after the 2026-05-16 checkpoint changes, "
            "then `python scripts/python/download_wandb_runs.py --group <name>`.",
            kind="warn",
        )
        ckpt_path = None
    else:
        ckpt_path = _candidates[0]
        print(f"Using checkpoint: {ckpt_path}")
    return (ckpt_path,)


@app.cell
def _(ckpt_path, mo):
    if ckpt_path is None:
        _ = mo.callout("No checkpoint → skipping Section 6.", kind="info")
        gnn_ckpt = None
    else:
        try:
            import torch
            from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN
            gnn_ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            _model = MembranePropertyGNN(**gnn_ckpt["model_kwargs"])
            _model.load_state_dict(gnn_ckpt["state_dict"])
            _model.eval()
            print(f"Loaded model — properties: {gnn_ckpt['properties']}, "
                  f"epoch: {gnn_ckpt.get('epoch')}")
            _ = mo.callout(
                "Model loaded. Per-system embedding extraction is a stub here: "
                "add a forward-hook on the post-trunk readout (see "
                "`lipid_gnn/membrane_prop_gnn.py`), feed the 70-system chunks, "
                "and project the resulting embeddings into `all_coords`'s PCA "
                "basis. The geometry mismatch is what answers the "
                "extrapolation question.",
                kind="info",
            )
        except Exception as _e:
            _ = mo.callout(f"Section 6 load failed: {_e}", kind="warn")
            gnn_ckpt = None
    return (gnn_ckpt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Open questions

        - Does the descriptor-based composition embedding actually predict
          which compositions the GNN extrapolates badly? Section 6's
          forward-hook extension answers this once retrained checkpoints land.
        - Are the Ward archetypes stable under bootstrap? (todo: resampled-fit
          robustness panel.)
        - The simplex-over-archetypes composition representation (Rep. 1 in
          the plan) is not implemented — only the mole-fraction-weighted
          centroid. Add if the centroid view turns out to be too smooth.
        - The GNN single-lipid-probe descriptor (descriptor 5 in the plan)
          requires bead-vocab decoupling from `LIPID_TYPES`; tracked as Phase 2.
        """
    )
    return


if __name__ == "__main__":
    app.run()
