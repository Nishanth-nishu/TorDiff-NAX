# Flow Matching on General Geometries (Riemannian Flow Matching, RFM)
- **Citation:** R. T. Q. Chen, Y. Lipman. ICLR 2024. arXiv:2302.03660
- **URL:** https://arxiv.org/abs/2302.03660 · PDF: `2023_chen_riemannian_flow_matching.pdf` · text: `fulltext/2023_chen_riemannian_flow_matching.txt`
- **Tags:** technique, manifold generative modelling, product spaces

## Key idea
Trains continuous normalizing flows (CNFs) on manifolds using conditional vector fields built from a **premetric** d(x, y). When geodesics have a closed form — Euclidean space, sphere, hyperbolic space, **torus, or any product of these** — training is completely simulation-free (Sec. 1, Fig. 1). Spectral distances handle general manifolds.

## Relation to the three gaps
- Gaps 1–3: **not addressed directly.** RFM is the enabling technique for a flow-matching FlexiTors on **T^m × (bounded bond-angle box)**: geodesic interpolation per factor replaces TD's wrapped-normal score tables.

## Metrics
Protein and RNA torsion datasets, climate data and others; no GEOM.

## Reusable for FlexiTors
- Torus geodesic path: τ_t = τ_0 + t·wrap(τ_1 − τ_0), with target velocity wrap(τ_1 − τ_0). The product-manifold loss is a weighted sum of per-factor losses.
- Straight paths allow few sampling steps. Exact likelihood via the divergence stays available, which TD's Boltzmann generator needs.
