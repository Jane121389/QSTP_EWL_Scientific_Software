
from __future__ import annotations

# =============================================================================
# IMPORTACIONES
# =============================================================================

# `dataclass` permite definir estructuras de datos compactas e inmutables.
from dataclasses import dataclass

# `Literal` restringe ciertos argumentos a opciones válidas previamente
# definidas, lo cual evita errores al indicar el tipo de QSTP.
from typing import Literal

# NumPy se utiliza para generación aleatoria, manejo de arreglos, estadística
# básica y operaciones numéricas sobre los perfiles de violación.
import numpy as np

# Se importan los objetos fundamentales del núcleo matemático:
#
# PI:
#     Constante π.
#
# ExperimentPoint:
#     Estructura que representa un punto completo del espacio de parámetros.
#
# evaluate:
#     Función que calcula las probabilidades relevantes del QSTP.
#
# violation_magnitude:
#     Función que determina la intensidad de una violación estándar o
#     complementaria.
from .core import (
    PI,
    ExperimentPoint,
    evaluate,
    violation_magnitude,
)


# =============================================================================
# TIPOS AUXILIARES
# =============================================================================

# Solo se permiten dos clases de violación:
#
#   "standard"       -> QSTP estándar.
#   "complementary"  -> QSTP complementario.
QSTPKind = Literal["standard", "complementary"]


# =============================================================================
# DOMINIO DE PARÁMETROS
# =============================================================================

@dataclass(frozen=True)
class ParameterDomain:
    """
    Define el dominio uniforme de muestreo para estimar el volumen de violación.

    Cada parámetro se considera independiente y se muestrea uniformemente
    dentro de su intervalo físico.

    Es importante señalar que el volumen numérico obtenido depende de:

      1. Los límites del dominio.
      2. La medida de probabilidad adoptada.
      3. La resolución o número de muestras utilizado.

    Por tanto, el volumen calculado no es una propiedad absoluta del operador,
    sino una propiedad relativa al dominio y a la medida especificados.

    Attributes
    ----------
    t_a:
        Intervalo del parámetro estratégico del jugador A.

    t_b:
        Intervalo del parámetro estratégico del jugador B.

    R:
        Intervalo de pureza del estado mixto.

    theta:
        Intervalo del ángulo polar de la esfera de Bloch.

    phi:
        Intervalo de fase relativa.

    gamma:
        Intervalo del grado de entrelazamiento del operador EWL original.
    """

    # Parámetros estratégicos de ambos jugadores.
    t_a: tuple[float, float] = (-1.0, 1.0)
    t_b: tuple[float, float] = (-1.0, 1.0)

    # Pureza del estado mixto.
    R: tuple[float, float] = (0.0, 1.0)

    # Ángulos de la esfera de Bloch.
    theta: tuple[float, float] = (0.0, PI)
    phi: tuple[float, float] = (0.0, 2.0 * PI)

    # Entrelazamiento variable del operador EWL original.
    gamma: tuple[float, float] = (0.0, PI / 2.0)


# =============================================================================
# RESULTADOS DE VOLUMEN DE VIOLACIÓN
# =============================================================================

@dataclass(frozen=True)
class VolumeEstimate:
    """
    Almacena el resultado de una estimación Monte Carlo del volumen de violación.

    Attributes
    ----------
    operator:
        Nombre del operador de entrelazamiento evaluado.

    kind:
        Tipo de QSTP: estándar o complementario.

    samples:
        Número total de muestras utilizadas.

    violation_fraction:
        Fracción del dominio que produce violaciones.

    ci_low, ci_high:
        Límites inferior y superior del intervalo de confianza de Wilson.

    weighted_volume:
        Promedio de la magnitud de violación sobre todo el dominio.

    mean_intensity_given_violation:
        Intensidad promedio condicionada a que exista una violación.

    maximum_intensity:
        Mayor violación encontrada durante el muestreo.

    maximum_point:
        Punto del espacio de parámetros donde se encontró la máxima violación.
    """

    operator: str
    kind: str
    samples: int
    violation_fraction: float
    ci_low: float
    ci_high: float
    weighted_volume: float
    mean_intensity_given_violation: float
    maximum_intensity: float
    maximum_point: ExperimentPoint | None


# =============================================================================
# RESULTADOS DE ROBUSTEZ
# =============================================================================

@dataclass(frozen=True)
class RobustnessEstimate:
    """
    Almacena el resultado de una prueba local de robustez paramétrica.

    Attributes
    ----------
    operator:
        Operador de entrelazamiento evaluado.

    kind:
        Tipo de QSTP.

    radius:
        Radio relativo de perturbación.

    samples:
        Número de perturbaciones aleatorias aplicadas.

    survival_probability:
        Probabilidad de que la violación se conserve después de perturbar
        simultáneamente los parámetros.

    mean_intensity:
        Intensidad media de la violación después de las perturbaciones.

    intensity_std:
        Desviación estándar de la intensidad.

    reference_intensity:
        Intensidad de la configuración original sin perturbación.
    """

    operator: str
    kind: str
    radius: float
    samples: int
    survival_probability: float
    mean_intensity: float
    intensity_std: float
    reference_intensity: float


# =============================================================================
# FUNCIONES AUXILIARES DE MUESTREO
# =============================================================================

def _uniform(
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> float:
    """
    Genera un número aleatorio uniforme dentro de un intervalo.

    Parameters
    ----------
    rng:
        Generador pseudoaleatorio de NumPy.

    bounds:
        Tupla (mínimo, máximo).

    Returns
    -------
    float
        Valor aleatorio en el intervalo indicado.
    """

    return float(rng.uniform(bounds[0], bounds[1]))


def sample_point(
    rng: np.random.Generator,
    operator: str,
    domain: ParameterDomain,
) -> ExperimentPoint:
    """
    Genera un punto aleatorio del espacio de parámetros.

    Para el operador EWL original, gamma se muestrea en su intervalo físico.

    Para CNOT, dCNOT y B-gate, el operador es fijo y se usa gamma=pi/2 como
    valor convencional para mantener una interfaz uniforme.

    Parameters
    ----------
    rng:
        Generador aleatorio.

    operator:
        Nombre del operador.

    domain:
        Dominio de muestreo.

    Returns
    -------
    ExperimentPoint
        Punto aleatorio completo.
    """

    # Solo el entrelazador original tiene un grado de entrelazamiento continuo.
    gamma = (
        _uniform(rng, domain.gamma)
        if operator == "original"
        else PI / 2.0
    )

    # Construcción del punto experimental.
    return ExperimentPoint(
        t_a=_uniform(rng, domain.t_a),
        t_b=_uniform(rng, domain.t_b),
        R=_uniform(rng, domain.R),
        theta=_uniform(rng, domain.theta),
        phi=_uniform(rng, domain.phi),
        gamma=gamma,
        entangler=operator,
    )


# =============================================================================
# INTERVALO DE CONFIANZA DE WILSON
# =============================================================================

def wilson_interval(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """
    Calcula el intervalo de confianza de Wilson para una proporción binomial.

    Se utiliza para estimar la incertidumbre estadística de la fracción de
    puntos que violan el QSTP.

    El valor por defecto de z corresponde aproximadamente a un intervalo de
    confianza del 95 %.

    Parameters
    ----------
    successes:
        Número de puntos que producen violación.

    n:
        Número total de muestras.

    z:
        Cuantil de la distribución normal estándar.

    Returns
    -------
    tuple[float, float]
        Límites inferior y superior del intervalo de confianza.
    """

    # Si no hay muestras, el intervalo no está definido.
    if n <= 0:
        return np.nan, np.nan

    # Proporción observada.
    p = successes / n

    # Denominador de la corrección de Wilson.
    denominator = 1.0 + z * z / n

    # Centro corregido del intervalo.
    center = (
        p + z * z / (2.0 * n)
    ) / denominator

    # Radio corregido del intervalo.
    radius = (
        z
        * np.sqrt(
            (p * (1.0 - p) / n)
            + z * z / (4.0 * n * n)
        )
        / denominator
    )

    # Límites preliminares.
    low = center - radius
    high = center + radius

    # Correcciones exactas para los casos extremos.
    if successes == 0:
        low = 0.0

    if successes == n:
        high = 1.0

    # Restricción final al intervalo físico [0,1].
    return (
        max(0.0, float(low)),
        min(1.0, float(high)),
    )


# =============================================================================
# ESTIMACIÓN GLOBAL DEL VOLUMEN DE VIOLACIÓN
# =============================================================================

def estimate_violation_volume(
    operator: str,
    kind: QSTPKind,
    samples: int,
    seed: int = 2026,
    domain: ParameterDomain | None = None,
) -> VolumeEstimate:
    """
    Estima mediante Monte Carlo el volumen global de violación.

    Se calculan tres métricas complementarias:

    1. violation_fraction

       Fracción del dominio que viola el QSTP:

           V = N_viol / N_total.

    2. weighted_volume

       Promedio de la intensidad sobre todo el dominio:

           W = (1/N) sum_i v_i,

       donde los puntos sin violación contribuyen con cero.

    3. mean_intensity_given_violation

       Intensidad media condicionada a la existencia de violación:

           I = (1/N_viol) sum_{i en violaciones} v_i.

    Parameters
    ----------
    operator:
        Operador de entrelazamiento.

    kind:
        Tipo de QSTP.

    samples:
        Número de muestras Monte Carlo.

    seed:
        Semilla del generador aleatorio.

    domain:
        Dominio de muestreo. Si se omite, se usa ParameterDomain().

    Returns
    -------
    VolumeEstimate
        Resultado completo de la estimación.
    """

    # La estimación requiere al menos una muestra.
    if samples < 1:
        raise ValueError("samples must be positive")

    # Se utiliza el dominio predeterminado si no se proporciona uno.
    domain = domain or ParameterDomain()

    # Generador aleatorio reproducible.
    rng = np.random.default_rng(seed)

    # Número de puntos que violan el QSTP.
    count = 0

    # Suma de todas las intensidades; los puntos no violatorios aportan cero.
    total_intensity = 0.0

    # Mayor intensidad observada.
    max_intensity = 0.0

    # Punto donde se encuentra la máxima intensidad.
    max_point = None

    # Muestreo Monte Carlo.
    for _ in range(samples):

        # Generar un punto aleatorio del dominio.
        point = sample_point(
            rng,
            operator,
            domain,
        )

        # Evaluar probabilidades y magnitud de violación.
        intensity = float(
            violation_magnitude(
                evaluate(point),
                kind,
            )
        )

        # Acumular la intensidad global.
        total_intensity += intensity

        # Una intensidad positiva indica una violación.
        if intensity > 0.0:
            count += 1

            # Actualizar el máximo si corresponde.
            if intensity > max_intensity:
                max_intensity = intensity
                max_point = point

    # Fracción normalizada del dominio violatorio.
    fraction = count / samples

    # Intervalo estadístico de confianza.
    low, high = wilson_interval(
        count,
        samples,
    )

    # Intensidad promedio condicionada a la existencia de violaciones.
    conditional_mean = (
        total_intensity / count
        if count
        else 0.0
    )

    # Empaquetado del resultado.
    return VolumeEstimate(
        operator=operator,
        kind=kind,
        samples=samples,
        violation_fraction=fraction,
        ci_low=low,
        ci_high=high,
        weighted_volume=total_intensity / samples,
        mean_intensity_given_violation=conditional_mean,
        maximum_intensity=max_intensity,
        maximum_point=max_point,
    )


# =============================================================================
# PERFILES CONDICIONADOS V(R) Y V(GAMMA)
# =============================================================================

def estimate_volume_profile(
    operator: str,
    kind: QSTPKind,
    parameter: Literal["R", "gamma"],
    values: np.ndarray,
    samples_per_value: int,
    seed: int = 2026,
    domain: ParameterDomain | None = None,
) -> dict[str, np.ndarray]:
    """
    Construye un perfil condicionado del volumen de violación.

    Puede calcular:

        V(R)

    o, exclusivamente para el operador EWL original,

        V(gamma).

    Para cada valor fijo del parámetro se muestrean aleatoriamente los demás
    parámetros del dominio.

    Returns
    -------
    dict[str, np.ndarray]
        Diccionario con:

        values:
            Valores del parámetro fijo.

        fraction:
            Volumen condicionado de violación.

        weighted:
            Volumen ponderado condicionado.

        maximum:
            Máxima intensidad encontrada para cada valor.
    """

    # Los entrelazadores perfectos no poseen un gamma variable en este modelo.
    if parameter == "gamma" and operator != "original":
        raise ValueError(
            "gamma is variable only for the original EWL entangler"
        )

    # Dominio y generador aleatorio.
    domain = domain or ParameterDomain()
    rng = np.random.default_rng(seed)

    # Arreglos para almacenar las métricas de cada valor.
    fractions = np.zeros(len(values))
    weighted = np.zeros(len(values))
    maximum = np.zeros(len(values))

    # Barrido del parámetro condicionado.
    for i, value in enumerate(values):

        # Contador de violaciones.
        count = 0

        # Suma de intensidades.
        total = 0.0

        # Máxima intensidad observada.
        max_v = 0.0

        # Muestreo de los parámetros restantes.
        for _ in range(samples_per_value):

            # Punto aleatorio inicial.
            p = sample_point(
                rng,
                operator,
                domain,
            )

            # Sustituir únicamente el parámetro que se desea fijar.
            if parameter == "R":
                p = ExperimentPoint(
                    p.t_a,
                    p.t_b,
                    float(value),
                    p.theta,
                    p.phi,
                    p.gamma,
                    p.entangler,
                )

            else:
                p = ExperimentPoint(
                    p.t_a,
                    p.t_b,
                    p.R,
                    p.theta,
                    p.phi,
                    float(value),
                    p.entangler,
                )

            # Evaluar la magnitud de la violación.
            v = float(
                violation_magnitude(
                    evaluate(p),
                    kind,
                )
            )

            # Acumular estadísticas.
            total += v
            count += v > 0.0
            max_v = max(max_v, v)

        # Fracción de puntos violatorios.
        fractions[i] = count / samples_per_value

        # Intensidad media sobre todo el subdominio.
        weighted[i] = total / samples_per_value

        # Máxima intensidad para el valor fijado.
        maximum[i] = max_v

    # Resultado estructurado.
    return {
        "values": np.asarray(values, dtype=float),
        "fraction": fractions,
        "weighted": weighted,
        "maximum": maximum,
    }


# =============================================================================
# UMBRAL CRÍTICO OPERACIONAL
# =============================================================================

def critical_threshold(
    values: np.ndarray,
    fraction: np.ndarray,
    minimum_fraction: float = 1.0e-3,
) -> float:
    """
    Determina el primer valor del parámetro donde aparece una región violatoria
    suficientemente grande.

    Se define operacionalmente como

        x_c = min{x : V(x) >= minimum_fraction}.

    Esta cantidad es un umbral numérico dependiente de:

      • la malla de valores;
      • el número de muestras;
      • el criterio minimum_fraction.

    No debe interpretarse automáticamente como un punto crítico analítico.

    Returns
    -------
    float
        Primer valor que supera el umbral o NaN si ninguno lo alcanza.
    """

    # Localizar los índices que satisfacen el criterio.
    indices = np.flatnonzero(
        np.asarray(fraction) >= minimum_fraction
    )

    # Devolver el primer valor válido.
    return (
        float(values[indices[0]])
        if len(indices)
        else np.nan
    )


# =============================================================================
# MANEJO DE VARIABLES ANGULARES PERIÓDICAS
# =============================================================================

def _wrap(
    value: float,
    low: float,
    high: float,
) -> float:
    """
    Reintroduce una variable periódica en el intervalo [low, high).

    Es especialmente útil para la fase phi, ya que valores mayores que 2π o
    menores que cero representan físicamente la misma orientación angular.
    """

    # Longitud del intervalo periódico.
    width = high - low

    # Operación módulo desplazada al intervalo deseado.
    return float(
        ((value - low) % width) + low
    )


# =============================================================================
# PERTURBACIONES PARAMÉTRICAS
# =============================================================================

def perturb_point(
    point: ExperimentPoint,
    rng: np.random.Generator,
    radius: float,
    include_gamma: bool = True,
) -> ExperimentPoint:
    """
    Perturba simultáneamente todos los parámetros de un punto experimental.

    El radio se expresa como fracción del intervalo físico completo de cada
    parámetro.

    Por ejemplo, radius=0.05 permite perturbaciones máximas del 5 % de cada
    rango físico.

    Las variables limitadas se recortan mediante clip y la fase phi se trata
    como variable periódica.

    Parameters
    ----------
    point:
        Configuración de referencia.

    rng:
        Generador aleatorio.

    radius:
        Radio relativo de perturbación.

    include_gamma:
        Indica si gamma debe perturbarse para el entrelazador original.

    Returns
    -------
    ExperimentPoint
        Punto perturbado.
    """

    # Un radio negativo no tiene significado físico.
    if radius < 0:
        raise ValueError("radius must be nonnegative")

    def delta(width: float) -> float:
        """
        Genera una perturbación uniforme en:

            [-radius*width, radius*width].
        """

        return float(
            rng.uniform(
                -radius * width,
                radius * width,
            )
        )

    # t_A tiene rango total de longitud 2.
    t_a = float(
        np.clip(
            point.t_a + delta(2.0),
            -1.0,
            1.0,
        )
    )

    # t_B tiene rango total de longitud 2.
    t_b = float(
        np.clip(
            point.t_b + delta(2.0),
            -1.0,
            1.0,
        )
    )

    # R tiene rango total de longitud 1.
    R = float(
        np.clip(
            point.R + delta(1.0),
            0.0,
            1.0,
        )
    )

    # theta tiene rango total pi.
    theta = float(
        np.clip(
            point.theta + delta(PI),
            0.0,
            PI,
        )
    )

    # phi es una variable periódica con rango 2pi.
    phi = _wrap(
        point.phi + delta(2.0 * PI),
        0.0,
        2.0 * PI,
    )

    # Inicialmente se conserva gamma.
    gamma = point.gamma

    # Solo se perturba gamma para el entrelazador EWL original.
    if include_gamma and point.entangler == "original":
        gamma = float(
            np.clip(
                point.gamma + delta(PI / 2.0),
                0.0,
                PI / 2.0,
            )
        )

    # Construcción del nuevo punto perturbado.
    return ExperimentPoint(
        t_a,
        t_b,
        R,
        theta,
        phi,
        gamma,
        point.entangler,
    )


# =============================================================================
# ROBUSTEZ LOCAL
# =============================================================================

def estimate_local_robustness(
    reference: ExperimentPoint,
    kind: QSTPKind,
    radius: float,
    samples: int,
    seed: int = 2026,
) -> RobustnessEstimate:
    """
    Evalúa la robustez local de una configuración violatoria.

    La métrica principal es

        survival_probability
        = N_perturbaciones_violatorias / N_total.

    Esta cantidad mide qué tan estable es la violación frente a errores o
    fluctuaciones simultáneas de los parámetros.

    También se calcula la intensidad media y su desviación estándar.

    Parameters
    ----------
    reference:
        Punto de referencia.

    kind:
        Tipo de QSTP.

    radius:
        Radio relativo de perturbación.

    samples:
        Número de perturbaciones.

    seed:
        Semilla aleatoria.

    Returns
    -------
    RobustnessEstimate
        Resultado completo de robustez.
    """

    # Generador reproducible.
    rng = np.random.default_rng(seed)

    # Intensidad de la configuración sin perturbar.
    reference_intensity = float(
        violation_magnitude(
            evaluate(reference),
            kind,
        )
    )

    # Arreglo para almacenar la intensidad de cada perturbación.
    intensities = np.zeros(samples)

    # Aplicación de perturbaciones independientes.
    for i in range(samples):

        # Generar punto perturbado.
        p = perturb_point(
            reference,
            rng,
            radius,
            include_gamma=True,
        )

        # Calcular intensidad después de la perturbación.
        intensities[i] = float(
            violation_magnitude(
                evaluate(p),
                kind,
            )
        )

    # Construcción del resultado estadístico.
    return RobustnessEstimate(
        operator=reference.entangler,
        kind=kind,
        radius=radius,
        samples=samples,

        # Fracción de perturbaciones que conservan una violación positiva.
        survival_probability=float(
            np.mean(intensities > 0.0)
        ),

        # Intensidad promedio después de perturbar.
        mean_intensity=float(
            np.mean(intensities)
        ),

        # Dispersión de la intensidad.
        intensity_std=float(
            np.std(intensities)
        ),

        # Intensidad original.
        reference_intensity=reference_intensity,
    )


# =============================================================================
# CURVAS DE ROBUSTEZ
# =============================================================================

def robustness_curve(
    reference: ExperimentPoint,
    kind: QSTPKind,
    radii: np.ndarray,
    samples_per_radius: int,
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    """
    Construye una curva de robustez para diferentes radios de perturbación.

    Para cada radio se calculan:

      • probabilidad de supervivencia;
      • intensidad media;
      • desviación estándar de la intensidad.

    Returns
    -------
    dict[str, np.ndarray]
        Diccionario con las curvas estadísticas.
    """

    # Arreglos de salida.
    survival = np.zeros(len(radii))
    mean = np.zeros(len(radii))
    std = np.zeros(len(radii))

    # Evaluación independiente para cada radio.
    for i, radius in enumerate(radii):

        # Cambiar ligeramente la semilla evita reutilizar exactamente las
        # mismas perturbaciones en todos los radios.
        estimate = estimate_local_robustness(
            reference,
            kind,
            float(radius),
            samples_per_radius,
            seed + i,
        )

        # Almacenar las métricas.
        survival[i] = estimate.survival_probability
        mean[i] = estimate.mean_intensity
        std[i] = estimate.intensity_std

    # Empaquetado final.
    return {
        "radius": np.asarray(radii, dtype=float),
        "survival": survival,
        "mean_intensity": mean,
        "std_intensity": std,
    }
