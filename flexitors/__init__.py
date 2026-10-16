"""FlexiTors: round-5 extensions of Torsional Diffusion (round5/PLAN.md "Code structure").

The original TD code in ``torsional-diffusion/`` is the frozen baseline. Every access to it (and to ``tools/``) goes
through :mod:`flexitors.td` (round5/DECISION.md graft 1). Importing this package imports nothing else, so it does not
pull in torch.
"""
