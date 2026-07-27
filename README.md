# QSTP–EWL Scientific Software

Research software for reproducing and extending the analysis of the **Quantum Sure Thing Principle (QSTP)** in the **Eisert–Wilkens–Lewenstein (EWL) quantum Prisoner's Dilemma**.

The package reproduces the published results of our *Physica Scripta* article and provides additional analyses developed for the associated Ph.D. thesis, including global violation metrics, robustness studies, parameter-space exploration, and comparisons between perfect entanglers.

---

# Features

- Reproduces all published figures from the associated journal article.
- Implements the original EWL entangler and three perfect entanglers.
- Evaluates standard and complementary QSTP violations.
- Generates publication-quality figures.
- Performs global parameter-space analyses.
- Computes normalized violation volumes and robustness metrics.
- Estimates critical purity and entanglement thresholds.
- Exports reproducible CSV and NPZ datasets.

---

# Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/your_username/QSTP_EWL_Scientific_Software.git
cd QSTP_EWL_Scientific_Software
pip install -e .
```

---

# Included entanglers

The software supports the following entangling operators:

- Original EWL operator $J(\gamma)$
- CNOT perfect entangler
- dCNOT perfect entangler
- B-gate perfect entangler

---

# Article reproduction

Run

```bash
python experiments/reproduce_article.py
```

This generates:

- `Fig5_original_EWL`
- `Fig6_perfect_standard`
- `Fig7_perfect_complementary`

The implementation reproduces the results reported in:

> **Entanglement-Induced Violations of the Quantum Sure-Thing Principle in the Quantum Prisoners’ Dilemma**  
> *Physica Scripta*  
> DOI: **10.1088/1402-4896/ae16e8**

The software reproduces both the original EWL panels and the perfect-entangler analyses using the same parameter conventions reported in the publication.

---

# Figure 5 reproduction

The original article uses different phase conventions for the two EWL panels.

### Standard panel

- $\Phi=\pi$
- $\Theta=\{\pi/2,\;2\pi/5,\;3\pi/10\}$

### Complementary panel

- $\Phi=0$
- $\Theta=\{19\pi/30,\;17\pi/30,\;\pi/2\}$

The implementation follows the same ordering and grayscale convention used in the published figures.

Standalone exports are also generated:

```text
figures/article/Fig5a_EWL_standard.png
figures/article/Fig5b_EWL_complementary.png
```

---

# Extended thesis analyses

The software includes analyses that extend the published article.

Run

```bash
python experiments/thesis_global.py
```

Generated outputs include

- global interaction between purity and entanglement;
- phase–uncertainty maps;
- quantitative comparison of entanglers;
- reusable CSV and NPZ datasets.

For denser parameter grids:

```bash
python experiments/thesis_global.py --full
```

---

# Impact metrics

Global quantitative analyses can be generated with

```bash
python experiments/impact_metrics.py
```

or

```bash
python experiments/impact_metrics.py --full
```

The implemented metrics include:

## Normalized violation volume

The software estimates the fraction of a declared parameter domain that violates the standard or complementary QSTP.

Reported quantities include

- violation fraction;
- Wilson 95% confidence interval;
- weighted violation volume;
- mean violation intensity;
- maximum violation point.

The assumed sampling domain is explicitly stored in `ParameterDomain`.

---

## Critical purity and entanglement

Conditional violation profiles are constructed for

- $\mathcal{V}(R)$ for every entangler;
- $\mathcal{V}(\gamma)$ for the original EWL operator.

The reported critical value is the first sampled point whose violating fraction exceeds the selected operational threshold.

---

## Robustness analysis

Representative violating configurations are perturbed simultaneously in every physical parameter.

The software reports

- survival probability;
- remaining mean violation intensity;
- standard deviation of the violation.

---

# Generate all thesis figures

The complete pipeline is

```bash
python experiments/generate_thesis_figures.py
```

This produces

- `T04_violation_volume`
- `T05_critical_purity_entanglement`
- `T06_parameter_robustness`
- `T07_phase_theta_comparison`
- `T08_strategy_plane_global`

For higher-resolution figures:

```bash
python experiments/generate_thesis_figures.py --full
```

---

# Tests

Run

```bash
pytest
```

---

# Scientific scope

This software focuses specifically on identifying when quantum resources modify or violate the standard and complementary **Quantum Sure Thing Principle (QSTP)** within the EWL quantum Prisoner's Dilemma.

It is intentionally dedicated to this research problem and does not include unrelated payoff optimization or tournament analyses.

---

# Citation

If this software contributes to your research, please cite the associated Ph.D. thesis and the following publication:

> **Entanglement-Induced Violations of the Quantum Sure-Thing Principle in the Quantum Prisoners’ Dilemma**  
> *Physica Scripta*  
> DOI: **10.1088/1402-4896/ae16e8**

---

# License

This project is released for academic and research purposes.
