import re, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BS = chr(92)
Q = [('E-050', 'song_2021_score', "The key challenge is the fact that the estimated score functions are inaccurate in low density regions, where few data points are available for computing the score matching objective.", "Naive score-based generative modeling and its pitfalls"),
     ('E-051', 'song_2021_score', "Larger noise can obviously cover more low density regions for better score estimation, but it over-corrupts the data and alters it significantly from the original distribution.", "Score-based generative modeling with multiple noise perturbations"),
     ('E-052', 'weng_2019_domain_randomization', "Another is to avoid infeasible solutions that might arise from overly wide randomization distributions and thus might hinder successful policy learning.", "Guided Domain Randomization"),
     ('E-053', 'weng_2019_domain_randomization', "In the case of real-data-guided DR, we would like to learn the randomization parameters $" + BS + "xi$ that bring the state distribution in simulator close to the state distribution in the real world.", "Match Real Data Distribution"),
     ('E-054', 'dieleman_2022_guidance', "Clearly, guidance represents a trade-off: it dramatically improves adherence to the conditioning signal, as well as overall sample quality, but at great cost to diversity", "Classifier-free guidance"),
     ('E-055', 'dieleman_2022_guidance', "some percentage of the time, the conditioning information " + BS + "(y" + BS + ") is removed (10-20% tends to work well).", "Classifier-free guidance"),
     ('E-056', 'weng_2021_diffusion_models', "They found the most effective noise is to apply Gaussian noise at low resolution and Gaussian blur at high resolution.", "Scale up Generation Resolution and Quality")]
def n(s): return re.sub(r'\s+', ' ', s).strip()
root = sys.argv[1]
for e, f, q, sec in Q:
    t = open(f'{root}/{f}.txt', encoding='utf-8').read(); T = n(t)
    i = T.find(n(q))
    secs = [m.start() for m in re.finditer(re.escape(sec), T)]
    print(e, f, 'quote', 'FOUND at %d' % i if i >= 0 else 'NOT FOUND', '| section heading positions:', secs[:5])
    if i < 0:
        k = T.find(n(q)[:30]); print('   head30 at', k, T[k:k + 300] if k >= 0 else '')
    else:
        print('   ctx:', T[max(0, i - 500):i + len(n(q)) + 200])
    print()
