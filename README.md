
# QSTP–EWL Scientific Software

Research software for the doctoral-thesis analysis of the Quantum Sure Thing Principle in the quantum Prisoner's Dilemma.

## Included entanglers

- Original EWL operator $J(\gamma)$
- CNOT perfect entangler.
- dCNOT perfect entangler.
- B-gate perfect entangler.

## Article reproduction

```bash
pip install -e .
python experiments/reproduce_article.py
```

This generates:

- `Fig5_original_EWL`: standard and complementary original-EWL panels;
- `Fig6_perfect_standard`: CNOT and dCNOT standard-QSTP panels;
- `Fig7_perfect_complementary`: CNOT, dCNOT and B-gate complementary-QSTP panels.

The original-EWL reproduction uses $\Phi=\pi$. Theta-dependent curves are rendered in black and gray tones. Perfect-entangler presets follow the parameter pairs and phase labels reported in the article figures.

The article cited is **"Entanglement-Induced Violations of the Quantum Sure-Thing Principle in the Quantum Prisoners’ Dilemma"**, published in *Physica Scripta*. The paper demonstrates that quantum entanglement can induce violations of the *Quantum Sure-Thing Principle* in the quantum Prisoner's Dilemma, highlighting the fundamental role of quantum correlations in modifying decision-making processes. **DOI:** 10.1088/1402-4896/ae16e8.


## Thesis-impact analyses

```bash
python experiments/thesis_global.py
```

Generated results include:

1. global interaction between purity and entanglement;
2. phase–uncertainty maps for all entanglers;
3. quantitative ranking of entanglers by maximum violation;
4. reusable NPZ and CSV datasets.

Use `--full` for denser grids:

```bash
python experiments/thesis_global.py --full
```
The thesis cited is: **"Dinámica de decisiones y juegos en sistemas
cuánticos entrelazados"*

## Tests

```bash
pytest
```

## Scientific scope

The package is deliberately specific to the thesis question: identifying when quantum resources modify or violate the standard and complementary QSTP. It avoids unrelated payoff and tournament analyses.


## Impact metrics for the thesis

The extended program includes three global analyses that go beyond representative curves:

```bash
python experiments/impact_metrics.py
```

For denser Monte-Carlo and conditional-profile estimates:

```bash
python experiments/impact_metrics.py --full
```

### Normalized violation volume

The software estimates the fraction of a declared uniform parameter domain
that violates the standard or complementary QSTP. It reports a Wilson 95%
confidence interval, maximum violation, mean intensity conditioned on
violation, and a weighted volume combining extension and intensity.

Because a volume depends on the selected domain and measure, the program
stores those assumptions explicitly in `ParameterDomain`.

### Critical purity and entanglement

The program constructs conditional volume profiles:

- $\mathcal V(R)$ for every entangler;
- $\mathcal V(\gamma)$ for the original EWL operator.

The reported critical value is operational: the first sampled value at which
the estimated violating fraction reaches 0.5%. This threshold can be changed
in `experiments/impact_metrics.py`.

### Robustness to parameter perturbations

Around representative violating configurations, all applicable parameters are
simultaneously perturbed. The perturbation radius is expressed as a fraction
of each parameter's full physical range. The outputs are:

- probability that the violation survives;
- mean remaining violation intensity;
- standard deviation of the intensity.

### New output figures

- `T04_violation_volume`
- `T05_critical_purity_entanglement`
- `T06_parameter_robustness`

Numerical results are exported to CSV and compressed NPZ files.


## Generate all thesis figures

The dedicated thesis pipeline is:

```bash
python experiments/generate_thesis_figures.py
```

It generates, in `figures/thesis/`:

- `T04_violation_volume`
- `T05_critical_purity_entanglement`
- `T06_parameter_robustness`
- `T07_phase_theta_comparison`
- `T08_strategy_plane_global`

Use denser grids and larger Monte-Carlo samples with:

```bash
python experiments/generate_thesis_figures.py --full
```

The article figures remain available through:

```bash
python experiments/reproduce_article.py
```

## Standalone original-EWL article panels

The article reproduction now also exports both original-EWL panels separately:

```text
figures/article/Fig5a_EWL_standard.png
figures/article/Fig5b_EWL_complementary.png
```

The complementary panel reproduces the supplied EWL $(- , +)$ image with
$t_A=-0.25$, $t_B=0.3$, $R=1$, $\Phi=\pi$, and
$\Theta\in\{\pi/2,17\pi/30,19\pi/30\}$. The three $P_D$ curves use
black and gray tones and the region above $P_D=1/2$ is shaded green.


## Phase convention used to reproduce Figure 5

The two panels require different phase values:

- Standard QSTP panel: $\Phi=\pi$.
- Complementary QSTP panel: $\Phi=0$.

Using $\Phi=\pi$ in the complementary panel incorrectly places the
Theta-dependent $P_D$ curves near zero and does not reproduce the article.


## Final Figure 5 phase and curve-order convention

To reproduce the supplied article panels exactly:

- Standard panel: $\Phi=\pi$, curves ordered
  $\Theta=\pi/2,\,2\pi/5,\,3\pi/10$.
- Complementary panel: $\Phi=0$, curves ordered
  $\Theta=19\pi/30,\,17\pi/30,\,\pi/2$.

The order is important because the article uses black for the upper curve,
dark gray for the middle curve, and light gray for the lower curve.

## Citation

If you use this repository in your research, please cite the corresponding Ph.D. thesis and any related publications.

## License

This project is released for academic and research purposes.
