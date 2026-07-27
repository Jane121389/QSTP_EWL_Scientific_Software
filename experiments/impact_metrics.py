# Este script calcula métricas globales de impacto para el estudio del
# Quantum Sure Thing Principle (QSTP) dentro del Dilema del Prisionero
# cuántico de Eisert–Wilkens–Lewenstein (EWL).
#
# Análisis incluidos:
#
#   1. Volumen normalizado de violación.
#   2. Volumen ponderado por intensidad.
#   3. Intervalos de confianza de Wilson.
#   4. Pureza crítica y entrelazamiento crítico.
#   5. Perfiles condicionados de volumen de violación.
#   6. Robustez frente a perturbaciones simultáneas de parámetros.
#
# Operadores considerados:
#
#   • EWL original;
#   • CNOT;
#   • dCNOT;
#   • B-gate.
#
# El modo rápido usa muestras moderadas. La opción --full incrementa
# considerablemente el número de muestras para obtener estimaciones
# numéricas más densas.
#
# La lógica científica original se conserva. Únicamente se añadieron
# comentarios y documentación explicativa.
# =============================================================================


from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from qstp_ewl.core import PI, ExperimentPoint
from qstp_ewl.metrics import (
    estimate_violation_volume,
    estimate_volume_profile,
    critical_threshold,
    robustness_curve,
)
from qstp_ewl.plotting import save


# Directorio raíz del proyecto.
ROOT = Path(__file__).resolve().parents[1]
# Carpeta de salida para las figuras de la tesis.
OUT = ROOT / "figures" / "thesis"
# Carpeta de salida para los archivos numéricos.
DATA = ROOT / "data"

# Operadores de entrelazamiento incluidos en la comparación.
OPERATORS = ["original", "cnot", "dcnot", "bgate"]
# Tipos de violación analizados.
KINDS = ["standard", "complementary"]

# Representative article configurations used only as centers for the local
# perturbation analysis. Global volume estimates do not depend on these.
# Configuraciones centrales empleadas exclusivamente en el análisis local
# de robustez. No afectan las estimaciones globales de volumen.
ROBUSTNESS_REFERENCES = {
    ("original", "standard"): ExperimentPoint(-0.75, 0.30, 1.0, PI/2, PI, PI/2, "original"),
    ("original", "complementary"): ExperimentPoint(-0.25, 0.30, 1.0, 19*PI/30, PI, PI/2, "original"),
    ("cnot", "standard"): ExperimentPoint(0.8, 0.565, 1.0, PI/2, 3*PI/2, PI/2, "cnot"),
    ("cnot", "complementary"): ExperimentPoint(0.8, 0.435, 1.0, PI/2, PI/2, PI/2, "cnot"),
    ("dcnot", "standard"): ExperimentPoint(1.0, 0.455, 1.0, PI/2, 0.0, PI/2, "dcnot"),
    ("dcnot", "complementary"): ExperimentPoint(0.0, 0.544, 1.0, PI/2, 0.0, PI/2, "dcnot"),
    ("bgate", "complementary"): ExperimentPoint(0.72, -0.34, 1.0, PI/2, PI/2, PI/2, "bgate"),
}



# Estima el volumen global de violación para cada operador y para
# las versiones estándar y complementaria del QSTP.
def volume_analysis(samples: int, seed: int) -> list:
    # Lista donde se almacenan las estimaciones de volumen.
    results = []
    # Recorrer todos los operadores de entrelazamiento.
    for i, operator in enumerate(OPERATORS):
        # Evaluar las versiones estándar y complementaria.
        for j, kind in enumerate(KINDS):
            results.append(
                # Ejecutar una estimación Monte Carlo sobre el dominio físico.
                estimate_violation_volume(
                    operator,
                    kind,
                    samples=samples,
                    # Usar semillas distintas pero reproducibles.
                    seed=seed + 100*i + j,
                )
            )

    # Exportar las métricas de volumen a un archivo CSV.
    with (DATA / "violation_volume.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Escribir encabezados descriptivos.
        writer.writerow([
            "operator", "kind", "samples",
            "volume_fraction", "ci_low", "ci_high",
            "weighted_volume", "mean_intensity_given_violation",
            "maximum_intensity", "maximum_point",
        ])
        # Escribir una fila por combinación operador–tipo.
        for r in results:
        # Escribir encabezados descriptivos.
            writer.writerow([
                r.operator, r.kind, r.samples,
                r.violation_fraction, r.ci_low, r.ci_high,
                r.weighted_volume, r.mean_intensity_given_violation,
                r.maximum_intensity, repr(r.maximum_point),
            ])

    # Crear dos paneles: volumen geométrico y volumen ponderado.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), constrained_layout=True)
    # Posiciones horizontales de los operadores.
    x = np.arange(len(OPERATORS))
    # Ancho de las barras agrupadas.
    width = 0.36

    # Dibujar barras separadas para estándar y complementario.
    for offset, kind in [(-width/2, "standard"), (width/2, "complementary")]:
        rows = [r for r in results if r.kind == kind]
        # Fracción estimada del dominio que presenta violación.
        y = np.array([r.violation_fraction for r in rows])
        # Límite inferior del intervalo de confianza.
        low = np.array([r.ci_low for r in rows])
        # Límite superior del intervalo de confianza.
        high = np.array([r.ci_high for r in rows])
        axes[0].bar(x + offset, y, width, label=kind)
        # Añadir intervalos de confianza a la fracción estimada.
        axes[0].errorbar(
            x + offset, y,
            yerr=np.vstack([y-low, high-y]),
            fmt="none", capsize=3,
        )
        axes[1].bar(
            x + offset,
            # Volumen ponderado: extensión multiplicada por intensidad.
            [r.weighted_volume for r in rows],
            width,
            label=kind,
        )

    for ax in axes:
        ax.set_xticks(x, ["EWL", "CNOT", "dCNOT", "B-gate"])
        ax.grid(axis="y", alpha=0.2)
        ax.legend()

    axes[0].set_ylabel("Volumen normalizado de violación")
    axes[0].set_title("Extensión del dominio violatorio")
    axes[1].set_ylabel("Volumen ponderado")
    axes[1].set_title("Extensión × intensidad")
    save(fig, OUT / "T04_violation_volume")
    return results



# Construye perfiles condicionados respecto de R y gamma y calcula
# los valores críticos operacionales asociados.
def critical_analysis(samples_per_value: int, seed: int) -> None:
    # Malla de pureza para los perfiles condicionados.
    R_values = np.linspace(0.0, 1.0, 31)
    # Malla de entrelazamiento para el operador EWL original.
    gamma_values = np.linspace(0.0, PI/2, 31)

    # Diccionario con todos los perfiles estimados.
    profiles = {}
    # Tabla resumida de valores críticos.
    rows = []

    # Recorrer todos los operadores de entrelazamiento.
    for i, operator in enumerate(OPERATORS):
        # Evaluar las versiones estándar y complementaria.
        for j, kind in enumerate(KINDS):
        # Estimar el perfil condicionado por gamma para EWL.
            profile = estimate_volume_profile(
                operator,
                kind,
                "R",
                R_values,
                samples_per_value,
                seed=seed + 1000 + 100*i + j,
            )
            # Guardar el perfil condicionado por pureza.
            profiles[(operator, kind, "R")] = profile
            # Primer valor de R cuya fracción violatoria alcanza 0.5%.
            rc = critical_threshold(R_values, profile["fraction"], minimum_fraction=0.005)
            rows.append((operator, kind, "R", rc))

        # Evaluar las versiones estándar y complementaria.
    for j, kind in enumerate(KINDS):
        # Estimar el perfil condicionado por gamma para EWL.
        profile = estimate_volume_profile(
            "original",
            kind,
            "gamma",
            gamma_values,
            samples_per_value,
            seed=seed + 2000 + j,
        )
        profiles[("original", kind, "gamma")] = profile
        # Primer valor de gamma cuya fracción violatoria alcanza 0.5%.
        gc = critical_threshold(gamma_values, profile["fraction"], minimum_fraction=0.005)
        rows.append(("original", kind, "gamma", gc))

    # Exportar los valores críticos estimados.
    with (DATA / "critical_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["operator", "kind", "parameter", "critical_value"])
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), constrained_layout=True)

    # Graficar los perfiles de pureza de todos los operadores.
    for operator, label in zip(OPERATORS, ["EWL", "CNOT", "dCNOT", "B-gate"]):
        p = profiles[(operator, "standard", "R")]
        axes[0].plot(p["values"], p["fraction"], label=f"{label}, estándar")
        p = profiles[(operator, "complementary", "R")]
        axes[0].plot(p["values"], p["fraction"], linestyle="--", label=f"{label}, comp.")

    axes[0].set(
        xlabel=r"Pureza $R$",
        ylabel="Volumen de violación condicionado",
        title="Pureza crítica y crecimiento de las violaciones",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)

    # Comparar el perfil de gamma estándar y complementario para EWL.
    for kind, style in [("standard", "-"), ("complementary", "--")]:
        p = profiles[("original", kind, "gamma")]
        axes[1].plot(p["values"]/PI, p["fraction"], style, linewidth=2, label=kind)

    axes[1].set(
        xlabel=r"$\gamma/\pi$",
        ylabel="Volumen de violación condicionado",
        title="Entrelazamiento crítico del operador EWL original",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    save(fig, OUT / "T05_critical_purity_entanglement")

    # Guardar perfiles completos en formato NPZ comprimido.
    np.savez_compressed(
        DATA / "critical_profiles.npz",
        R=R_values,
        gamma=gamma_values,
        **{
            f"{op}_{kind}_{parameter}_{metric}": values[metric]
            for (op, kind, parameter), values in profiles.items()
            for metric in ("fraction", "weighted", "maximum")
        },
    )



# Evalúa la estabilidad de configuraciones violatorias representativas
# frente a perturbaciones simultáneas de todos los parámetros físicos.
def robustness_analysis(samples_per_radius: int, seed: int) -> None:
    # Radios relativos de perturbación, del 0% al 15% del rango físico.
    radii = np.linspace(0.0, 0.15, 16)
    # Diccionario de curvas de robustez.
    curves = {}

    # Evaluar cada configuración de referencia.
    for index, ((operator, kind), point) in enumerate(ROBUSTNESS_REFERENCES.items()):
        # Perturbar simultáneamente los parámetros alrededor del punto central.
        curves[(operator, kind)] = robustness_curve(
            point,
            kind,
            radii,
            samples_per_radius,
            seed=seed + 3000 + index,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), constrained_layout=True)

    for (operator, kind), curve in curves.items():
        # Etiqueta identificadora de la curva.
        label = f"{operator}: {kind}"
        # Probabilidad de que la violación sobreviva a la perturbación.
        axes[0].plot(curve["radius"], curve["survival"], label=label)
        # Intensidad media remanente después de la perturbación.
        axes[1].plot(curve["radius"], curve["mean_intensity"], label=label)

    axes[0].set(
        xlabel="Radio relativo de perturbación",
        ylabel="Probabilidad de conservar la violación",
        title="Robustez topológica local",
        ylim=(-0.02, 1.02),
    )
    axes[1].set(
        xlabel="Radio relativo de perturbación",
        ylabel="Intensidad media de violación",
        title="Degradación de la intensidad",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

    save(fig, OUT / "T06_parameter_robustness")

    # Guardar perfiles completos en formato NPZ comprimido.
    np.savez_compressed(
        DATA / "robustness_curves.npz",
        radius=radii,
        **{
            f"{operator}_{kind}_{metric}": curve[metric]
            for (operator, kind), curve in curves.items()
            for metric in ("survival", "mean_intensity", "std_intensity")
        },
    )



# Procesa los argumentos de línea de comandos, crea directorios de salida
# y ejecuta los tres análisis principales.
def main() -> None:
    # Crear el analizador de argumentos del programa.
    parser = argparse.ArgumentParser(
        description="Impact metrics: violation volume, critical parameters and robustness."
    )
    # --full activa estimaciones con más muestras.
    parser.add_argument("--full", action="store_true")
    # Semilla reproducible para todos los experimentos Monte Carlo.
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Crear la carpeta de datos si no existe.
    DATA.mkdir(parents=True, exist_ok=True)
    # Crear la carpeta de figuras si no existe.
    OUT.mkdir(parents=True, exist_ok=True)

    # Configuración de alta resolución.
    if args.full:
        # Número de muestras para el volumen global.
        volume_samples = 60000
        # Muestras por valor fijo en los perfiles condicionados.
        profile_samples = 4000
        # Muestras por radio en el análisis de robustez.
        robustness_samples = 8000
    else:
        # Configuración rápida para el volumen global.
        volume_samples = 8000
        # Configuración rápida para los perfiles.
        profile_samples = 600
        # Configuración rápida para la robustez.
        robustness_samples = 1500

    # Ejecutar el análisis global de volumen.
    volume_analysis(volume_samples, args.seed)
    # Ejecutar el análisis de parámetros críticos.
    critical_analysis(profile_samples, args.seed)
    # Ejecutar el análisis local de robustez.
    robustness_analysis(robustness_samples, args.seed)


if __name__ == "__main__":
    main()
