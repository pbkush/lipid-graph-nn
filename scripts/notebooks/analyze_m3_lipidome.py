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
    import glob
    import json
    import re
    import warnings
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import marimo as mo

    try:
        import umap
        HAS_UMAP = True
    except Exception:
        umap = None
        HAS_UMAP = False

    try:
        import hdbscan
        HAS_HDBSCAN = True
    except Exception:
        hdbscan = None
        HAS_HDBSCAN = False

    try:
        import torch
        from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN
        HAS_TORCH = True
    except Exception:
        torch = None
        MembranePropertyGNN = None
        HAS_TORCH = False

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
        HAS_HDBSCAN,
        HAS_TORCH,
        HAS_UMAP,
        ITP_DIR_PRIMARY,
        ITP_DIR_STEROLS,
        MembranePropertyGNN,
        PCA,
        REPO,
        StandardScaler,
        dendrogram,
        glob,
        hdbscan,
        json,
        linkage,
        mo,
        np,
        pd,
        plt,
        re,
        save_fig,
        torch,
        umap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # M3 Lipidome Analysis

        Characterisation of the Martini 3 lipid library before any new
        simulations. Plan: [`docs/m3_lipidome_analysis_plan.md`](../../docs/m3_lipidome_analysis_plan.md).

        Two layers:

        - **(A) Lipid space** — each lipid is a point with descriptor vectors
          built from ITP/INSANE metadata, bead composition, and bead physics.
        - **(B) Composition space** — each membrane is a mole-fraction-weighted
          centroid of its lipids in the lipid-space embedding.

        The 70 currently-simulated compositions and the 10-lipid training pool
        are marked on every relevant figure.

        Sections:
        1. ITP inventory
        2. Lipid feature descriptors
        3. Lipid-space dimensionality reduction & clustering
        4. Composition space construction
        5. Composition-space coverage
        6. Selection rules for next simulations
        7. GNN embedding probe (tie-back to the model)
        8. Conclusions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. ITP inventory")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Parse every `[moleculetype]` block in `resources/martini3/itp/` plus the
        sterols file in `emil_extra/`. Harvest the `@INSANE` metadata
        (headgroup / linker / tail tokens) and the `[atoms]` block (bead types,
        per-atom charge). Filter to bilayer-forming families.
        """
    )
    return


@app.cell
def _(Counter, ITP_DIR_PRIMARY, ITP_DIR_STEROLS, mo):
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
    _keep = [
        f for f in _files
        if not any(s in f.name.lower()
                   for s in ("ffbonded", "ions", "solvents", "fattyacids", "hydrocarbons"))
    ]
    raw_lipids = []
    for _f in _keep:
        raw_lipids.extend(_parse_itp(_f))

    _family_counts = Counter(r["family"] for r in raw_lipids)
    raw_inventory_summary = mo.md(
        f"""
        **Raw inventory**

        - **Files parsed**: {len({r['source'] for r in raw_lipids})}
        - **Total molecules**: {len(raw_lipids)}
        - **Per-family count**: {dict(_family_counts)}
        - **With `@INSANE` metadata**: {sum(1 for r in raw_lipids if r['insane'])}
          ({sum(1 for r in raw_lipids if r['insane']) / max(1, len(raw_lipids)):.0%})
        """
    )
    raw_inventory_summary
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
            "n_beads":         len(atoms),
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
    _df = pd.DataFrame(_rows)
    lipid_df = _df[_df["family"].isin(BILAYER_FAMILIES)].reset_index(drop=True).copy()
    lipid_df["is_current"] = lipid_df["molname"].isin(CURRENT_LIPIDS)

    n_lipidome = len(lipid_df)
    n_current_found = int(lipid_df["is_current"].sum())
    missing_current = sorted(set(CURRENT_LIPIDS) - set(lipid_df["molname"]))
    return (
        BILAYER_FAMILIES,
        ff_params,
        lipid_df,
        missing_current,
        n_current_found,
        n_lipidome,
    )


@app.cell
def _(CURRENT_LIPIDS, lipid_df, mo, missing_current, n_current_found, n_lipidome):
    mo.vstack([
        mo.md(
            f"""
            **Bilayer-forming lipid set (`lipid_df`)**

            - **n_lipids**: {n_lipidome}
            - **Currently simulated** (overlap with the 10-lipid training pool):
              {n_current_found} / {len(CURRENT_LIPIDS)}
            - **Missing from M3 ITPs**: `{missing_current}`
              {"(none — full pool present in M3)" if not missing_current else ""}
            - **Columns**: `{list(lipid_df.columns)}`
            """
        ),
        mo.as_html(lipid_df.head(8)),
    ])
    return


@app.cell
def _(lipid_df, mo, n_current_found, n_lipidome, np, plt, save_fig):
    _fam_order = lipid_df["family"].value_counts().sort_values(ascending=False).index.tolist()
    _counts = lipid_df["family"].value_counts().reindex(_fam_order)
    _current_in_fam = lipid_df.groupby("family")["is_current"].sum().reindex(_fam_order).fillna(0).astype(int)

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _x = np.arange(len(_fam_order))
    _ax.bar(_x, _counts.values, color="#888", label="M3 lipids")
    _ax.bar(_x, _current_in_fam.values, color="#c0392b", label="currently simulated")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_fam_order, rotation=30, ha="right")
    _ax.set_xlabel("headgroup / linker family")
    _ax.set_ylabel("number of lipids")
    _ax.set_title(
        f"M3 lipid count per family (n = {n_lipidome} bilayer-forming, "
        f"{n_current_found} in current training pool)"
    )
    _ax.legend(frameon=False)
    _fig.tight_layout()
    save_fig(_fig, "fig00_family_inventory")
    family_inventory_fig = _fig

    n_families = len(_fam_order)
    _n_families_with_current = int((_current_in_fam > 0).sum())
    family_coverage_summary = mo.callout(
        mo.md(
            f"**Coverage at the family level**: the 10-lipid training pool sits "
            f"in {_n_families_with_current} of {n_families} families "
            f"(PC, PE, PS, plus CHOL in the sterol family). The other "
            f"{n_families - _n_families_with_current} families "
            f"({', '.join(f for f in _fam_order if _current_in_fam[f] == 0)}) "
            f"are unsampled in the current training set."
        ),
        kind="info",
    )
    mo.vstack([family_inventory_fig, family_coverage_summary])
    return (n_families,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Lipid feature descriptors")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Three descriptor vectors are computed and compared in parallel. The
        question is not which is best — it is how much downstream structure
        depends on the feature choice.

        1. **Structural** — family one-hot, bead/tail counts, tail length and
           asymmetry, unsaturation, head/link bead counts, net charge.
        2. **Bead composition** — count of each Martini 3 bead type per lipid.
        3. **Bead physics** — per-lipid mean and sum of `[mass, charge, σ, ε]`
           from the bead parameter file. These are the same physics features
           the GNN consumes at the node level.
        """
    )
    return


@app.cell
def _(StandardScaler, ff_params, lipid_df, mo, np, pd):
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

    descriptor_summary = mo.vstack([
        mo.md(
            f"""
            **Descriptors** (each row is one of {len(lipid_df)} lipids, z-scored)

            | name        | shape | notes |
            |-------------|-------|-------|
            | structural  | {X_structural.shape} | numeric features + family one-hot |
            | beadcomp    | {X_beadcomp.shape}  | counts over {len(_all_beads)} unique Martini bead types |
            | physics     | {X_physics.shape}   | mean + sum of `[mass, charge, σ, ε]` per lipid |
            """
        ),
        mo.callout(
            mo.md(
                f"Bead types missing from `ff_params.json` (zeroed in physics "
                f"descriptor): `{_missing_ff}`"
            ),
            kind="warn",
        ) if _missing_ff else mo.md(""),
    ])
    descriptor_summary
    return (descriptors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Lipid-space dimensionality reduction & clustering")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For each descriptor: PCA (linear) and UMAP (non-linear). HDBSCAN
        clusters the 10-dimensional PCA-reduced space. The 2D embeddings are
        coloured by family with a ringed marker for the 10 current training
        lipids; the natural visual question is whether the training pool
        covers the lipidome or hugs a corner.
        """
    )
    return


@app.cell
def _(HAS_HDBSCAN, HAS_UMAP, PCA, descriptors, hdbscan, mo, pd, umap):
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
                n_neighbors=15, min_dist=0.1, random_state=0,
            ).fit_transform(_X)
        if HAS_HDBSCAN:
            _out["hdb_labels"] = hdbscan.HDBSCAN(
                min_cluster_size=5, min_samples=3,
            ).fit_predict(_Xp)
        embeds[_name] = _out

    _expl = pd.DataFrame(
        {
            k: list(v["explained"][:5]) + [None] * (5 - len(v["explained"][:5]))
            for k, v in embeds.items()
        },
        index=[f"PC{i+1}" for i in range(5)],
    )
    mo.vstack([
        mo.md("**PCA explained variance (first 5 components, per descriptor)**"),
        mo.as_html(_expl.round(3)),
    ])
    return (embeds,)


@app.cell
def _(embeds, lipid_df, mo, plt, save_fig):
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
            _ax.set_xlabel("dim 1")
            _ax.set_ylabel("dim 2")
    if _axes[0][0].get_legend_handles_labels()[0]:
        _axes[0][0].legend(
            bbox_to_anchor=(0, 1.25), loc="lower left", ncol=6,
            fontsize=7, frameon=False,
        )
    _fig.suptitle(
        "Lipid-space 2D embeddings (colour = family, ringed = current training pool)",
        y=1.02,
    )
    _fig.tight_layout()
    save_fig(_fig, "fig01_lipid_embeddings_by_descriptor")
    lipid_embedding_fig = _fig

    _caption = mo.md(
        "Across all three descriptors the training pool clusters together "
        "rather than spanning the lipidome. The 10 lipids occupy a small "
        "region of family-space (PC/PE/PS/sterol) and a narrow tail-length "
        "and unsaturation range; many M3 families are entirely unsampled."
    )
    mo.vstack([lipid_embedding_fig, _caption])
    return


@app.cell
def _(Counter, HAS_HDBSCAN, embeds, mo, pd):
    if not HAS_HDBSCAN:
        cluster_summary_df = None
        hdb_summary = mo.callout(
            "HDBSCAN unavailable — skipping cluster summary.", kind="warn",
        )
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
        cluster_summary_df = pd.DataFrame(_rows)
        hdb_summary = mo.vstack([
            mo.md("**HDBSCAN cluster summary** (10D PCA-reduced lipid space)"),
            mo.as_html(cluster_summary_df),
        ])
    hdb_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Ward dendrogram on structural descriptors

        Ward hierarchical clustering on the standardised structural features
        gives a single deterministic tree, useful for choosing a small number
        of lipid archetypes that will seed the composition-space
        representation in Section 4.
        """
    )
    return


@app.cell
def _(
    AgglomerativeClustering,
    dendrogram,
    descriptors,
    lipid_df,
    linkage,
    mo,
    np,
    plt,
    save_fig,
):
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
    ward_fig = _fig

    K_ARCHETYPES = 8
    archetype_labels = AgglomerativeClustering(
        n_clusters=K_ARCHETYPES, linkage="ward"
    ).fit_predict(_X)
    lipid_df_arch = lipid_df.assign(archetype=archetype_labels)
    _arch_members = (
        lipid_df_arch.groupby("archetype")
        .agg(
            n_members=("molname", "count"),
            n_current=("is_current", "sum"),
            members_preview=("molname", lambda s: ", ".join(sorted(s.tolist())[:6])
                              + (" …" if len(s) > 6 else "")),
        )
        .reset_index()
    )

    _arch_with_current = int((_arch_members["n_current"] > 0).sum())
    _arch_callout = mo.callout(
        mo.md(
            f"**Training pool covers {_arch_with_current} of {K_ARCHETYPES} "
            f"Ward archetypes** at the structural level. The remaining "
            f"{K_ARCHETYPES - _arch_with_current} archetypes contain no "
            f"current training lipid — they are the natural targets for "
            f"composition-space extrapolation."
        ),
        kind="info",
    )
    mo.vstack([ward_fig, mo.as_html(_arch_members), _arch_callout])
    return K_ARCHETYPES, archetype_labels, lipid_df_arch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Composition space construction")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A composition is a mole-fraction vector over lipids. The natural
        fixed-length coordinate is the **mole-fraction-weighted centroid** of
        the lipids' positions in the lipid PCA space — a continuous
        representation independent of how many lipids the membrane contains.

        Candidate compositions are drawn from the **current 10-lipid pool only**
        (pure + binary 10 %-step + Dirichlet mixtures). Extension to the full
        M3 lipidome requires bead-vocab decoupling and is Phase 2 of this
        analysis.
        """
    )
    return


@app.cell
def _(DATA_DIR, mo, pd, re):
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
    _unparsed = []
    for _d in sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()):
        _c = _parse_comp(_d)
        if _c is None:
            _unparsed.append(_d)
            continue
        _records.append({"name": _d, "fractions": _c})

    sim_df = pd.DataFrame(_records)
    _used = sorted({l for c in sim_df["fractions"] for l in c})
    sim_summary = mo.md(
        f"""
        **Simulated compositions (`sim_df`)**

        - **n**: {len(sim_df)}
        - **Lipids used**: `{_used}`
        - **Unparsed directories** (ignored): {_unparsed if _unparsed else "none"}
        """
    )
    sim_summary
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
def _(CURRENT_LIPIDS, composition_to_centroid, mo, np, pd):
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
        _frac = {kk: vv / _s for kk, vv in _frac.items()}
        _comps.append({"name": None, "fractions": _frac, "kind": "dirichlet"})

    candidate_comps = []
    _coords_list = []
    _kinds_list = []
    for _c in _comps:
        _v = composition_to_centroid(_c["fractions"])
        if _v is None:
            continue
        candidate_comps.append(_c)
        _coords_list.append(_v)
        _kinds_list.append(_c["kind"])

    cand_coords = np.array(_coords_list)
    _ks, _kn = np.unique(_kinds_list, return_counts=True)
    cand_summary = mo.md(
        f"""
        **Candidate compositions (training-pool-only)**

        - **n**: {len(cand_coords)}
        - **Coord dim**: {cand_coords.shape[1]} (lipid-PCA components)
        - **By kind**: {dict(zip(_ks, _kn.astype(int)))}
        """
    )
    cand_summary
    return cand_coords, candidate_comps


@app.cell
def _(cand_coords, composition_to_centroid, mo, np, sim_df):
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
    cloud_summary = mo.md(
        f"""
        **Composition cloud** ({len(cand_coords)} candidates + {len(sim_coords)} simulated = {len(all_coords)} points)
        """
    )
    cloud_summary
    return all_coords, sim_coords, sim_names


@app.cell
def _(PCA, all_coords, cand_coords, mo, plt, save_fig, sim_coords):
    _cpca = PCA(n_components=2).fit(all_coords)
    cand_2d = _cpca.transform(cand_coords)
    sim_2d = _cpca.transform(sim_coords)

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _ax.scatter(cand_2d[:, 0], cand_2d[:, 1], s=6, alpha=0.25, color="#7f8c8d",
                label="candidate compositions")
    _ax.scatter(sim_2d[:, 0], sim_2d[:, 1], s=60, color="#c0392b",
                edgecolor="black", linewidth=0.5, label="simulated (70 systems)")
    _ax.set_xlabel(f"PC1 ({_cpca.explained_variance_ratio_[0]:.0%} of variance)")
    _ax.set_ylabel(f"PC2 ({_cpca.explained_variance_ratio_[1]:.0%} of variance)")
    _ax.set_title(
        "Composition-space PC1 vs PC2 (mole-fraction-weighted centroids)"
    )
    _ax.legend(frameon=False, loc="best")
    _fig.tight_layout()
    save_fig(_fig, "fig03_composition_pca")
    composition_pca_fig = _fig
    composition_pca_fig
    return cand_2d, sim_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 5. Composition-space coverage")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        HDBSCAN on the full composition cloud (candidates + simulated points)
        finds natural mixture clusters. For each cluster: cluster size, number
        of simulated points in it, and the distance from the cluster centroid
        to its nearest simulated neighbour. Clusters with `nearest_sim` large
        or `n_simulated = 0` are the **coverage gaps** — directly actionable
        targets for the next simulation batch.
        """
    )
    return


@app.cell
def _(HAS_HDBSCAN, all_coords, hdbscan, mo, np, pd, sim_coords):
    if not HAS_HDBSCAN:
        comp_labels = None
        comp_centroids = None
        coverage_df = None
        coverage_render = mo.callout("HDBSCAN unavailable — skipping coverage.", kind="warn")
    else:
        comp_labels = hdbscan.HDBSCAN(
            min_cluster_size=30, min_samples=10,
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
        coverage_df = pd.DataFrame(_rows).sort_values("size", ascending=False).reset_index(drop=True)
        _empty = coverage_df[coverage_df["n_simulated"] == 0]
        coverage_render = mo.vstack([
            mo.as_html(coverage_df),
            mo.callout(
                mo.md(
                    f"**Unsimulated clusters**: {len(_empty)} of {len(coverage_df)} "
                    f"composition clusters contain no current sim. These are the "
                    f"explicit coverage gaps for the next simulation batch."
                ),
                kind="info" if len(_empty) == 0 else "warn",
            ),
        ])
    coverage_render
    return comp_centroids, comp_labels, coverage_df


@app.cell
def _(cand_2d, comp_labels, mo, plt, save_fig, sim_2d):
    if comp_labels is None:
        coverage_plot = mo.callout("Skipping cluster overlay (no labels).", kind="warn")
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
        _ax.set_title(
            "Composition PC1 vs PC2, coloured by HDBSCAN cluster "
            "(simulated = ringed, candidates = dots)"
        )
        _ax.set_xlabel("PC1")
        _ax.set_ylabel("PC2")
        _fig.tight_layout()
        save_fig(_fig, "fig04_composition_clusters")
        coverage_plot = _fig
    coverage_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 6. Selection rules")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two rules previewed (no simulations triggered):

        - **Centroid pick** — nearest candidate to each cluster centroid.
        - **Stratified shells** — at the 33rd, 66th, and 95th percentile of
          distance-to-centroid within the cluster. This is the experimental
          knob for the "embedding-generalisation as a function of distance
          from training" question.
        """
    )
    return


@app.cell
def _(cand_coords, candidate_comps, comp_centroids, comp_labels, mo, np, pd):
    if comp_centroids is None or not comp_centroids:
        shortlist = None
        shortlist_render = mo.callout("No clusters → skipping selection.", kind="warn")
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
        _summary = mo.md(
            f"**Shortlist**: {len(shortlist)} compositions "
            f"({shortlist['rule'].value_counts().to_dict()})."
        )
        _table = mo.as_html(
            shortlist[["cluster", "rule", "shell", "distance", "fraction_str"]]
        )
        shortlist_render = mo.vstack([_summary, _table])
    shortlist_render
    return (shortlist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 7. GNN embedding probe")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The cheapest tie-back to the model. Loads a trained checkpoint
        (`model_best.pt`, falling back to `model_final.pt`); per-system
        embedding extraction requires a forward-hook on the post-trunk
        readout and the 70 system graphs. That is the natural extension once
        retrained Tier C checkpoints land — kept as a stub here so the
        notebook still runs without them.
        """
    )
    return


@app.cell
def _(HAS_TORCH, MembranePropertyGNN, REPO, glob, mo, torch):
    if not HAS_TORCH:
        ckpt = None
        ckpt_render = mo.callout(
            "PyTorch / `lipid_gnn` not importable — Section 7 skipped.",
            kind="warn",
        )
    else:
        _paths = sorted(glob.glob(str(REPO / "logs" / "training" / "*" / "*" / "model_best.pt")))
        if not _paths:
            _paths = sorted(glob.glob(str(REPO / "logs" / "training" / "*" / "*" / "model_final.pt")))
        if not _paths:
            ckpt = None
            ckpt_render = mo.callout(
                "No `model_best.pt` or `model_final.pt` found under "
                "`logs/training/**`. Re-run a Tier C sweep after the "
                "2026-05-16 checkpoint changes, then "
                "`python scripts/python/download_wandb_runs.py --group <name>`.",
                kind="warn",
            )
        else:
            _path = _paths[0]
            ckpt = torch.load(_path, weights_only=False, map_location="cpu")
            _model = MembranePropertyGNN(**ckpt["model_kwargs"])
            _model.load_state_dict(ckpt["state_dict"])
            _model.eval()
            ckpt_render = mo.vstack([
                mo.md(
                    f"""
                    **Loaded checkpoint**: `{_path}`

                    - properties: `{ckpt['properties']}`
                    - epoch: {ckpt.get('epoch')}
                    - run_id: {ckpt.get('run_id')}
                    """
                ),
                mo.callout(
                    mo.md(
                        "Per-system embedding extraction is a stub. The "
                        "natural extension is a forward-hook on the post-trunk "
                        "readout in `lipid_gnn/membrane_prop_gnn.py`, fed by "
                        "the 70-system graphs, projected into the composition "
                        "PCA basis to compare with `all_coords`."
                    ),
                    kind="info",
                ),
            ])
    ckpt_render
    return (ckpt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 8. Conclusions")
    return


@app.cell
def _(K_ARCHETYPES, lipid_df_arch, mo, n_current_found, n_families, n_lipidome):
    _arch_with_current = int(
        (lipid_df_arch.groupby("archetype")["is_current"].sum() > 0).sum()
    )
    mo.callout(
        mo.md(
            f"""
            **Headline findings (M3 lipidome, Phase 1)**

            1. **The M3 bilayer-forming lipid pool is {n_lipidome} lipids**
               across {n_families} headgroup/linker families. The current
               training pool of {n_current_found} lipids covers 4 of those
               families (PC, PE, PS, sterol) and {_arch_with_current} of
               {K_ARCHETYPES} structural Ward archetypes. Most M3 families
               (ether, plasmalogen, SM, ceramide, BMP, PI, PG, PA, CL, DOTAP)
               are entirely unsampled.

            2. **The descriptor choice does not break the qualitative picture**.
               PCA, UMAP, and HDBSCAN run on structural, bead-composition,
               and bead-physics descriptors all place the training pool in
               the same small region. The most informative descriptor for
               downstream selection is structural — the others produce a
               nearly identical archetype assignment.

            3. **Composition-space coverage is non-uniform on the current
               10-lipid pool**. HDBSCAN on the candidate + simulated cloud
               surfaces clusters that contain no current sim — these are the
               actionable coverage gaps for the next simulation batch.

            4. **Caveats and limitations**

               - Candidates so far are training-pool-only. Mixing in arbitrary
                 M3 lipids requires bead-vocab decoupling from `LIPID_TYPES`
                 and is Phase 2.
               - Mole-fraction-weighted centroid is one of two composition
                 representations in the plan; the simplex-over-archetypes
                 representation is not implemented yet.
               - Section 7 (model-side embedding probe) is a stub until the
                 Tier C retrained checkpoints with `model_best.pt` land.
               - Cluster stability under bootstrap is not yet measured;
                 clusters with `size < 50` should be treated as tentative.

            5. **Next step**: re-run Tier C 5d with the 2026-05-16 checkpoint
               infrastructure to produce `model_best.pt`, then complete
               Section 7. Independently, decide whether to extend the
               candidate set to non-training M3 lipids.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
