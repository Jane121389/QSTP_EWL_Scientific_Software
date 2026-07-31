"""Genera mapas de violación del QSTP para entrelazadores perfectos.

El programa realiza dos etapas:

1. Busca, dentro de cada cuadrante estratégico (t_A, t_B), un punto
   representativo que produzca una región de violación amplia en el plano
   fase--incertidumbre.
2. Evalúa ese punto con una malla más fina y exporta las figuras y los datos.

Los entrelazadores estudiados son CNOT, dCNOT y B-gate. Para cada uno se
analizan la condición estándar (superior) y la complementaria (inferior) del
Quantum Sure Thing Principle (QSTP).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# LOCALIZACIÓN E IMPORTACIÓN DEL PAQUETE CIENTÍFICO
# -----------------------------------------------------------------------------

# Directorio en el que se encuentra este script.
HERE = Path(__file__).resolve().parent

# Ruta esperada del paquete que contiene la implementación matemática del QSTP.
PROJECT = HERE / "unpacked" / "QSTP_EWL_Scientific_Software_v4"

# Se agrega PROJECT al inicio de sys.path para que Python pueda importar
# qstp_ewl aunque el paquete no esté instalado globalmente.
sys.path.insert(0, str(PROJECT))

# ExperimentPoint encapsula los parámetros de un experimento.
# evaluate ejecuta el modelo EWL y obtiene las probabilidades/resultados.
# violation_magnitude calcula cuánto se viola la condición indicada del QSTP.
from qstp_ewl.core import ExperimentPoint, evaluate, violation_magnitude


# -----------------------------------------------------------------------------
# CONSTANTES Y CONFIGURACIÓN DEL BARRIDO
# -----------------------------------------------------------------------------

PI = np.pi

# Cada tupla contiene:
#   1) nombre interno utilizado por el paquete;
#   2) etiqueta que aparecerá en figuras y archivos.
OPERATORS = [
    ("cnot", "CNOT"),
    ("dcnot", "dCNOT"),
    ("bgate", "B-gate"),
]

# Intervalos de los parámetros estratégicos t_A y t_B para cada cuadrante.
# Se usa -1e-6 en vez de 0 en los intervalos negativos para evitar incluir
# accidentalmente el origen en dos cuadrantes distintos.
QUADRANTS = {
    "++": ((0.0, 1.0), (0.0, 1.0)),
    "+-": ((0.0, 1.0), (-1.0, -1.0e-6)),
    "-+": ((-1.0, -1.0e-6), (0.0, 1.0)),
    "--": ((-1.0, -1.0e-6), (-1.0, -1.0e-6)),
}

# Las dos desigualdades del QSTP que se evaluarán en cada figura.
KINDS = [
    ("standard", "Condición superior"),
    ("complementary", "Condición inferior"),
]


def phase_theta_map(
    entangler: str,
    kind: str,
    t_a: float,
    t_b: float,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
) -> np.ndarray:
    """Calcula el mapa de violación en el plano (Theta, Phi).

    Cada elemento ``out[i, j]`` contiene la magnitud de violación para
    ``Theta = theta_values[i]`` y ``Phi = phi_values[j]``, manteniendo fijos
    el entrelazador y los parámetros estratégicos t_A y t_B.
    """

    # Matriz con filas asociadas a Theta y columnas asociadas a Phi.
    out = np.zeros((len(theta_values), len(phi_values)), dtype=float)

    # Barrido completo de la malla fase--incertidumbre.
    for i, theta in enumerate(theta_values):
        for j, phi in enumerate(phi_values):
            # Punto experimental del modelo EWL.
            # R=1 fija un estado completamente puro.
            # gamma=pi/2 fija entrelazamiento máximo.
            p = evaluate(
                ExperimentPoint(
                    t_a=t_a,
                    t_b=t_b,
                    R=1.0,
                    theta=float(theta),
                    phi=float(phi),
                    gamma=PI / 2,
                    entangler=entangler,
                )
            )

            # Magnitud positiva de la violación. Un valor cero indica que la
            # condición del QSTP no se viola en este punto.
            out[i, j] = violation_magnitude(p, kind)

    return out


def strategic_values(bounds: tuple[float, float], n: int) -> np.ndarray:
    """Construye una malla uniforme de valores estratégicos en un intervalo."""

    lo, hi = bounds
    return np.linspace(lo, hi, n)


def select_representative_point(
    entangler: str,
    kind: str,
    quadrant: str,
    n_t: int,
    n_theta: int,
    n_phi: int,
):
    """Selecciona el punto (t_A, t_B) más representativo de un cuadrante.

    Para cada pareja estratégica se calcula primero un mapa grueso en
    (Theta, Phi). Los candidatos se comparan lexicográficamente mediante:

    1. fracción del mapa donde existe violación;
    2. magnitud media de la violación;
    3. magnitud máxima de la violación.

    Esta regla favorece regiones amplias y persistentes de violación en lugar
    de máximos muy altos pero aislados.
    """

    # Recupera los intervalos de t_A y t_B correspondientes al cuadrante.
    a_bounds, b_bounds = QUADRANTS[quadrant]

    # Mallas estratégicas utilizadas durante la búsqueda gruesa.
    a_values = strategic_values(a_bounds, n_t)
    b_values = strategic_values(b_bounds, n_t)

    # Malla reducida para abaratar la selección del punto representativo.
    theta_values = np.linspace(0.0, PI, n_theta)
    phi_values = np.linspace(0.0, 2.0 * PI, n_phi)

    best = None

    # Se prueban todas las combinaciones estratégicas del cuadrante.
    for t_a in a_values:
        for t_b in b_values:
            matrix = phase_theta_map(
                entangler,
                kind,
                float(t_a),
                float(t_b),
                theta_values,
                phi_values,
            )

            # Proporción de puntos de la malla con violación numéricamente
            # distinta de cero. El umbral evita contar ruido de redondeo.
            area = float(np.count_nonzero(matrix > 1e-10) / matrix.size)

            # Magnitud promedio en toda la malla, incluidos los ceros.
            mean = float(matrix.mean())

            # Mayor violación observada para el candidato actual.
            vmax = float(matrix.max())

            # Python compara tuplas lexicográficamente: primero area, luego
            # mean y finalmente vmax en caso de empate.
            score = (area, mean, vmax)

            if best is None or score > best[0]:
                best = (score, float(t_a), float(t_b))

    # Garantiza que la malla no haya estado vacía.
    assert best is not None

    # Devuelve t_A, t_B y la puntuación del punto seleccionado.
    return best[1], best[2], best[0]


def render_combination(
    entangler: str,
    display: str,
    quadrant: str,
    selected: dict[str, tuple[float, float, tuple[float, float, float]]],
    n_theta: int,
    n_phi: int,
    out_dir: Path,
    data_dir: Path,
):
    """Genera y exporta los dos mapas finos de una combinación.

    La figura contiene dos paneles: condición estándar y condición
    complementaria. Cada panel puede utilizar un punto estratégico distinto,
    porque el punto representativo se selecciona independientemente para cada
    condición del QSTP.
    """

    # Malla fina utilizada para las figuras definitivas.
    theta_values = np.linspace(0.0, PI, n_theta)
    phi_values = np.linspace(0.0, 2.0 * PI, n_phi)

    # Dos paneles con ejes compartidos para facilitar la comparación visual.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        sharex=True,
        sharey=True,
    )

    # Diccionario que será almacenado en el archivo NPZ.
    exported = {"theta": theta_values, "phi": phi_values}
    image = None

    # Recorre simultáneamente los ejes y las dos condiciones del QSTP.
    for idx, (ax, (kind, condition)) in enumerate(zip(axes, KINDS), start=1):
        # Punto estratégico previamente seleccionado con la malla gruesa.
        t_a, t_b, coarse = selected[kind]

        # Recalcula el mapa con la resolución final.
        matrix = phase_theta_map(
            entangler,
            kind,
            t_a,
            t_b,
            theta_values,
            phi_values,
        )

        # Conserva tanto la matriz como sus parámetros estratégicos.
        exported[f"panel_{idx}"] = matrix
        exported[f"panel_{idx}_metadata"] = np.array(
            [kind, t_a, t_b], dtype=object
        )

        # imshow representa Phi/pi en [0, 2] y Theta/pi en [0, 1].
        # Se fija la escala de color entre 0 y 0.5 para que todas las figuras
        # sean directamente comparables.
        image = ax.imshow(
            matrix,
            origin="lower",
            extent=[0, 2, 0, 1],
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=0.5,
            cmap="viridis",
        )

        # Dibuja en blanco la frontera aproximada entre la región sin
        # violación y la región con violación.
        if np.any(matrix > 1e-10):
            ax.contour(
                phi_values / PI,
                theta_values / PI,
                matrix,
                levels=[1e-8],
                colors="white",
                linewidths=0.8,
            )

        ax.set_title(
            f"{condition} del QSTP, {display} "
            f"({quadrant[0]}, {quadrant[1]})",
            fontsize=12,
        )
        ax.set_xlabel(r"Fase $\Phi/\pi$")
        ax.set_ylabel(r"Incertidumbre $\Theta/\pi$")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        # Anota dentro del panel el punto estratégico que generó el mapa.
        ax.text(
            0.02,
            0.98,
            rf"$t_A={t_a:.3f},\;t_B={t_b:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )

    assert image is not None

    fig.suptitle(
        f"{display} ({quadrant[0]}, {quadrant[1]}): regiones de violación "
        "en el plano fase–incertidumbre",
        fontsize=15,
    )

    # Reserva espacio a la derecha para una única barra de color compartida.
    fig.subplots_adjust(right=0.88, top=0.87, wspace=0.16)
    cax = fig.add_axes([0.90, 0.16, 0.022, 0.68])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("Magnitud de la violación")

    # Nombre base común para las figuras y el archivo de datos.
    stem = (
        f"{display.replace('-', '').replace(' ', '')}_"
        f"{quadrant}_phase_theta_maps"
    )

    # Crea los directorios si todavía no existen.
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Exporta una imagen rasterizada, una versión vectorial y los datos
    # numéricos necesarios para reproducir o analizar los mapas.
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    np.savez_compressed(data_dir / f"{stem}.npz", **exported)

    # Libera la memoria de Matplotlib, importante al producir muchas figuras.
    plt.close(fig)


def main():
    """Procesa argumentos y ejecuta el barrido completo."""

    parser = argparse.ArgumentParser(
        description="Generate all quadrant combinations for perfect entanglers."
    )

    # Resolución de la malla estratégica usada para seleccionar (t_A, t_B).
    parser.add_argument(
        "--nt",
        type=int,
        default=13,
        help="Strategic grid points per axis during point selection.",
    )

    # Resolución del mapa grueso usado en la etapa de selección.
    parser.add_argument("--coarse-theta", type=int, default=11)
    parser.add_argument("--coarse-phi", type=int, default=15)

    # Resolución del mapa fino utilizado en las figuras definitivas.
    parser.add_argument("--n-theta", type=int, default=91)
    parser.add_argument("--n-phi", type=int, default=121)

    # Directorio raíz para todas las salidas.
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "all_perfect_entangler_combinations",
    )

    # Permiten ejecutar solamente operadores o cuadrantes específicos.
    parser.add_argument(
        "--operators",
        nargs="*",
        default=["cnot", "dcnot", "bgate"],
    )
    parser.add_argument(
        "--quadrants",
        nargs="*",
        default=["++", "+-", "-+", "--"],
    )

    args = parser.parse_args()

    out_dir = args.output / "figures"
    data_dir = args.output / "data"

    # Aquí se acumula una fila por operador, cuadrante y condición del QSTP.
    summary_rows = []

    # Recorre los tres operadores entrelazadores.
    for entangler, display in OPERATORS:
        if entangler not in args.operators:
            continue

        # Recorre los cuatro cuadrantes estratégicos solicitados.
        for quadrant in args.quadrants:
            selected = {}

            # La condición estándar y la complementaria seleccionan sus propios
            # puntos representativos de manera independiente.
            for kind, condition in KINDS:
                t_a, t_b, score = select_representative_point(
                    entangler,
                    kind,
                    quadrant,
                    args.nt,
                    args.coarse_theta,
                    args.coarse_phi,
                )

                selected[kind] = (t_a, t_b, score)

                # Registra los resultados de la búsqueda gruesa para análisis
                # posterior y trazabilidad de las figuras.
                summary_rows.append(
                    {
                        "operator": display,
                        "quadrant": quadrant,
                        "condition": condition,
                        "kind": kind,
                        "t_A": t_a,
                        "t_B": t_b,
                        "coarse_nonzero_fraction": score[0],
                        "coarse_mean_magnitude": score[1],
                        "coarse_max_magnitude": score[2],
                    }
                )

                # Muestra el progreso en tiempo real; flush=True evita que la
                # salida permanezca retenida en el búfer.
                print(
                    display,
                    quadrant,
                    kind,
                    t_a,
                    t_b,
                    score,
                    flush=True,
                )

            # Una vez elegidos ambos puntos, genera la figura comparativa.
            render_combination(
                entangler,
                display,
                quadrant,
                selected,
                args.n_theta,
                args.n_phi,
                out_dir,
                data_dir,
            )

    args.output.mkdir(parents=True, exist_ok=True)

    # El nombre del resumen refleja los operadores y cuadrantes solicitados.
    summary_path = args.output / (
        f"summary_{'_'.join(args.operators)}_"
        f"{'_'.join(args.quadrants)}.csv"
    )

    # Exporta los puntos representativos y sus métricas de selección.
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # Genera una breve descripción metodológica dentro del directorio de salida.
    readme = args.output / "README.txt"
    readme.write_text(
        "Barrido sistemático de CNOT, dCNOT y B-gate en los cuatro cuadrantes "
        "estratégicos (t_A,t_B): (++), (+-), (-+), (--).\n\n"
        "Cada figura contiene la condición superior e inferior del QSTP. "
        "El punto representativo de cada panel se selecciona maximizando primero "
        "la fracción no nula del mapa fase-incertidumbre y después su magnitud "
        "media y máxima.\n"
        "Los signos se refieren al signo de t_A y t_B, no a signos arbitrarios "
        "en la matriz SU(2).\n",
        encoding="utf-8",
    )


# Este bloque evita que main() se ejecute cuando el archivo es importado como
# módulo desde otro script. Solo se ejecuta al invocarlo directamente.
if __name__ == "__main__":
    main()
