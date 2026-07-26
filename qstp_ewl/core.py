# =============================================================================
#
# Este archivo implementa el núcleo matemático del software científico para el
# estudio del Quantum Sure Thing Principle (QSTP).
#
# Se documentan los siguientes bloques:
#   • Estrategias cuánticas.
#   • Operadores de entrelazamiento EWL y operadores perfectos.
#   • Construcción del estado final.
#   • Estados condicionales cuánticos.
#   • Actualización bayesiana cuántica.
#   • Evaluación del QSTP estándar y complementario.
#
# El código fuente NO fue modificado; únicamente se añadió esta documentación.
# =============================================================================
from __future__ import annotations

# =============================================================================
# IMPORTACIONES
# =============================================================================

# `dataclass` permite definir estructuras de datos compactas e inmutables.
from dataclasses import dataclass

# `Literal` restringe algunos argumentos a valores de texto específicos,
# mientras que `NamedTuple` facilita devolver varias probabilidades con nombres.
from typing import Literal, NamedTuple

# NumPy se utiliza para álgebra lineal compleja, productos de Kronecker,
# exponentiales de fase y manipulación de matrices de densidad.
import numpy as np


# =============================================================================
# CONSTANTES NUMÉRICAS Y OBJETOS BÁSICOS
# =============================================================================

# Constante π usada en la parametrización de estrategias y entrelazadores.
PI = np.pi

# Tolerancia numérica para detectar valores prácticamente nulos.
EPS = 1e-12

# Tolerancia empleada al evaluar desigualdades del QSTP alrededor de 1/2.
TOL = 1e-10

# Matrices identidad de uno y dos qubits.
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

# Operador clásico de "defección" usado en el esquema EWL original:
#
#       D = [[ 0,  1],
#            [-1,  0]]
#
# Este operador es unitario y antihermítico. Su producto tensorial D⊗D
# aparece en el generador del operador de entrelazamiento original.
D = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)

# Producto de Kronecker D⊗D.
DD = np.kron(D, D)

# Estado inicial |00> en la base computacional ordenada como
# |00>, |01>, |10>, |11>.
KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

# Proyectores computacionales de un qubit:
#
# P0 = |0><0|
# P1 = |1><1|
P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

# Proyector que selecciona el evento "A elige |1>" sin medir explícitamente B:
#
#     P_A1 = |1><1|_A ⊗ I_B
P_A1 = np.kron(P1, I2)


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

class Probabilities(NamedTuple):
    """
    Agrupa las probabilidades relevantes para evaluar el QSTP.

    Attributes
    ----------
    p_d:
        Probabilidad actualizada de que A elija la estrategia de delatar
        cuando B está descrito por el estado mixto varrho(R, theta, phi).

    p_a1_b0:
        Probabilidad condicional P(A=1 | B=0).

    p_a1_b1:
        Probabilidad condicional P(A=1 | B=1).

    p_a1_unconditional:
        Probabilidad marginal P(A=1) obtenida directamente del estado conjunto.
    """

    p_d: float
    p_a1_b0: float
    p_a1_b1: float
    p_a1_unconditional: float


@dataclass(frozen=True)
class ExperimentPoint:
    """
    Define un punto completo del espacio de parámetros del experimento.

    Parameters
    ----------
    t_a, t_b:
        Parámetros de estrategia de los jugadores A y B en el intervalo [-1,1].

    R:
        Pureza/radio del vector de Bloch del estado mixto asociado a B.

    theta:
        Ángulo polar del estado de B en la esfera de Bloch.

    phi:
        Fase relativa del estado de B.

    gamma:
        Grado de entrelazamiento del operador EWL original.
        Para entrelazadores perfectos se conserva por compatibilidad,
        aunque el operador correspondiente es fijo.

    entangler:
        Nombre del operador de entrelazamiento:
        "original", "cnot", "dcnot" o "bgate".
    """

    t_a: float
    t_b: float
    R: float
    theta: float
    phi: float
    gamma: float = np.pi / 2
    entangler: str = "original"


# =============================================================================
# ESTRATEGIAS CUÁNTICAS
# =============================================================================

def strategy(t: float) -> np.ndarray:
    """
    Construye la estrategia unitaria U(t) utilizada en el artículo.

    El espacio uniparamétrico se divide en dos ramas:

    1. Si t >= 0:
           theta = t*pi
           phi   = 0

       Esta rama interpola entre cooperación y defección mediante el
       ángulo theta.

    2. Si t < 0:
           theta = 0
           phi   = -t*pi/2

       Esta rama introduce una fase cuántica manteniendo theta=0.

    La forma general de la estrategia es

        U(theta,phi) =
        [[ exp(i phi) cos(theta/2),   sin(theta/2)],
         [ -sin(theta/2),             exp(-i phi) cos(theta/2)]]

    Returns
    -------
    np.ndarray
        Matriz unitaria 2x2 de la estrategia.
    """

    # Verificación del dominio físico del parámetro estratégico.
    if not -1 - EPS <= t <= 1 + EPS:
        raise ValueError("t must be in [-1,1]")

    # Se recorta el valor para evitar pequeñas desviaciones numéricas fuera
    # del intervalo permitido.
    t = float(np.clip(t, -1, 1))

    # Rama positiva del espacio de estrategias.
    if t >= 0:
        theta, phi = t * PI, 0.0

    # Rama negativa del espacio de estrategias.
    else:
        theta, phi = 0.0, -t * PI / 2

    # Abreviaturas trigonométricas.
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)

    # Construcción de la matriz unitaria.
    return np.array(
        [
            [np.exp(1j * phi) * c, s],
            [-s, np.exp(-1j * phi) * c],
        ],
        dtype=complex,
    )


# =============================================================================
# OPERADORES DE ENTRELAZAMIENTO
# =============================================================================

def original_entangler(gamma: float) -> np.ndarray:
    """
    Construye el operador de entrelazamiento original de EWL.

    Se define como

        J(gamma) = exp[-i gamma/2 (D⊗D)].

    Como (D⊗D)^2 = I, la exponencial puede escribirse analíticamente como

        J(gamma)
        = cos(gamma/2) I
          - i sin(gamma/2) (D⊗D).

    Parameters
    ----------
    gamma:
        Parámetro de entrelazamiento en [0, pi/2].

    Returns
    -------
    np.ndarray
        Matriz unitaria 4x4.
    """

    # Validación del intervalo de entrelazamiento permitido.
    if not -EPS <= gamma <= PI / 2 + EPS:
        raise ValueError("gamma must be in [0,pi/2]")

    # Protección frente a errores de punto flotante.
    gamma = float(np.clip(gamma, 0, PI / 2))

    # Evaluación cerrada de la exponencial matricial.
    return (
        np.cos(gamma / 2) * I4
        - 1j * np.sin(gamma / 2) * DD
    )


def weyl_entangler(
    eta1: float,
    eta2: float,
    eta3: float,
) -> np.ndarray:
    """
    Construye el operador no local general parametrizado en la cámara de Weyl.

    Los parámetros (eta1, eta2, eta3) determinan una clase de equivalencia
    local de compuertas de dos qubits.

    Esta forma permite representar, entre otros, los entrelazadores perfectos:

        CNOT  = J(pi/2, 0, 0)
        dCNOT = J(pi/4, pi/4, 0)
        B     = J(pi/2, pi/4, 0)

    Returns
    -------
    np.ndarray
        Matriz unitaria 4x4.
    """

    # Combinaciones angulares que aparecen en los bloques de la matriz.
    a = (eta1 - eta2) / 2
    b = (eta1 + eta2) / 2

    # Fases globales de los subespacios pares e impares.
    em = np.exp(-1j * eta3 / 2)
    ep = np.exp(1j * eta3 / 2)

    # Construcción explícita de la matriz de dos qubits.
    return np.array(
        [
            [
                em * np.cos(a),
                0,
                0,
                -1j * em * np.sin(a),
            ],
            [
                0,
                ep * np.cos(b),
                -1j * ep * np.sin(b),
                0,
            ],
            [
                0,
                -1j * ep * np.sin(b),
                ep * np.cos(b),
                0,
            ],
            [
                -1j * em * np.sin(a),
                0,
                0,
                em * np.cos(a),
            ],
        ],
        dtype=complex,
    )


def entangler(
    name: str,
    gamma: float = PI / 2,
) -> np.ndarray:
    """
    Selecciona el operador de entrelazamiento solicitado.

    Parameters
    ----------
    name:
        Identificador textual del operador.

    gamma:
        Se utiliza únicamente para el operador EWL original.

    Returns
    -------
    np.ndarray
        Matriz unitaria 4x4 correspondiente al operador seleccionado.
    """

    # Normalización del nombre para aceptar variantes como
    # "B-gate", "b_gate", "dcnot", etc.
    key = name.lower().replace("-", "").replace("_", "")

    # Operador original EWL con entrelazamiento variable.
    if key in {"original", "ewl"}:
        return original_entangler(gamma)

    # Entrelazador perfecto CNOT.
    if key == "cnot":
        return weyl_entangler(PI / 2, 0, 0)

    # Entrelazador perfecto double-CNOT.
    if key == "dcnot":
        return weyl_entangler(PI / 4, PI / 4, 0)

    # Entrelazador perfecto B-gate.
    if key in {"bgate", "b"}:
        return weyl_entangler(PI / 2, PI / 4, 0)

    # Error explícito para evitar usar silenciosamente un operador incorrecto.
    raise ValueError(f"Unknown entangler: {name}")


# =============================================================================
# ESTADO FINAL DEL ESQUEMA EWL
# =============================================================================

def final_state(
    t_a: float,
    t_b: float,
    J: np.ndarray,
) -> np.ndarray:
    """
    Calcula el estado final puro del esquema EWL.

    La secuencia es

        |psi_f>
        = J^dagger (U_A ⊗ U_B) J |00>.

    Parameters
    ----------
    t_a, t_b:
        Parámetros estratégicos de A y B.

    J:
        Operador de entrelazamiento 4x4.

    Returns
    -------
    np.ndarray
        Vector de estado normalizado de dimensión 4.
    """

    # Aplicación de la secuencia completa del circuito EWL.
    psi = (
        J.conj().T
        @ np.kron(strategy(t_a), strategy(t_b))
        @ J
        @ KET00
    )

    # Normalización explícita para eliminar errores acumulados de redondeo.
    return psi / np.linalg.norm(psi)


# =============================================================================
# OPERACIONES SOBRE MATRICES DE DENSIDAD
# =============================================================================

def partial_trace_b(rho: np.ndarray) -> np.ndarray:
    """
    Calcula la traza parcial sobre el subsistema B.

    Si rho actúa sobre H_A ⊗ H_B, esta operación devuelve

        rho_A = Tr_B(rho).

    El reordenamiento reshape(2,2,2,2) interpreta los índices como

        rho[a, b, a', b'].

    Después se contraen los índices b y b'.
    """

    return np.trace(
        rho.reshape(2, 2, 2, 2),
        axis1=1,
        axis2=3,
    )


def projectors(
    theta: float,
    phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye los proyectores de los eigenestados de un qubit general.

    Los estados son

        |psi>
        = cos(theta/2)|0>
          + exp(i phi) sin(theta/2)|1>,

    y

        |psi_perp>
        = sin(theta/2)|0>
          - exp(i phi) cos(theta/2)|1>.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Proyectores |psi><psi| y |psi_perp><psi_perp|.
    """

    # Eigenestado principal del estado mixto varrho.
    psi = np.array(
        [
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2),
        ],
        dtype=complex,
    )

    # Estado ortogonal complementario.
    perp = np.array(
        [
            np.sin(theta / 2),
            -np.exp(1j * phi) * np.cos(theta / 2),
        ],
        dtype=complex,
    )

    # Construcción de los proyectores de rango uno.
    return (
        np.outer(psi, psi.conj()),
        np.outer(perp, perp.conj()),
    )


def conditional_a(
    rho_ab: np.ndarray,
    proj_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Calcula el estado condicional de A después de proyectar B.

    El operador de medición conjunto es

        M = I_A ⊗ Pi_B.

    La probabilidad del resultado es

        p = Tr[M rho_AB].

    El estado posterior normalizado es

        rho'_AB = M rho_AB M / p,

    y finalmente

        rho_A = Tr_B(rho'_AB).

    Returns
    -------
    tuple[np.ndarray, float]
        Estado condicional de A y probabilidad del resultado proyectado.
    """

    # Operador que actúa solo sobre el qubit B.
    M = np.kron(I2, proj_b)

    # Probabilidad de obtener el resultado asociado al proyector.
    prob = float(np.trace(M @ rho_ab).real)

    # Si el resultado tiene probabilidad prácticamente nula, el estado
    # condicional no está definido. Se devuelve NaN para evitar interpretaciones
    # físicas incorrectas.
    if prob <= EPS:
        return np.full((2, 2), np.nan + 0j), prob

    # Estado posterior conjunto después de la proyección.
    post = M @ rho_ab @ M / prob

    # Reducción al subsistema A.
    rho_a = partial_trace_b(post)

    # Simetrización hermítica para compensar pequeñas desviaciones numéricas.
    return (rho_a + rho_a.conj().T) / 2, prob


def updated_a(
    rho_ab: np.ndarray,
    R: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    """
    Actualiza el estado de A cuando B está descrito por un estado mixto general.

    El estado de B tiene eigenvalores

        w1 = (1+R)/2,
        w2 = (1-R)/2,

    y eigenproyectores asociados a |psi> y |psi_perp>.

    La actualización se construye como la mezcla convexa

        rho_A,varrho
        = w1 rho_A|psi
          + w2 rho_A|psi_perp.

    Parameters
    ----------
    rho_ab:
        Estado conjunto de A y B.

    R:
        Pureza/radio de Bloch del estado de B.

    theta, phi:
        Ángulos que determinan la orientación de los eigenestados de B.

    Returns
    -------
    np.ndarray
        Matriz de densidad actualizada de A.
    """

    # Proyectores de los dos eigenestados ortogonales.
    q, qp = projectors(theta, phi)

    # Estados condicionales de A asociados a cada proyección.
    r1, p1 = conditional_a(rho_ab, q)
    r2, p2 = conditional_a(rho_ab, qp)

    # Eigenvalores del estado mixto varrho.
    w1, w2 = (1 + R) / 2, (1 - R) / 2

    # Inicialización de la matriz actualizada.
    out = np.zeros((2, 2), dtype=complex)

    # Primer término de la mezcla.
    if w1 > EPS:
        if p1 <= EPS:
            return np.full((2, 2), np.nan + 0j)
        out += w1 * r1

    # Segundo término de la mezcla.
    if w2 > EPS:
        if p2 <= EPS:
            return np.full((2, 2), np.nan + 0j)
        out += w2 * r2

    # Renormalización final para garantizar Tr(rho_A)=1.
    tr = np.trace(out).real
    return out / tr


# =============================================================================
# PROBABILIDADES DE DECISIÓN
# =============================================================================

def p_defect(rho_a: np.ndarray) -> float:
    """
    Extrae la probabilidad de que A elija la estrategia de delatar.

    En la convención usada:

        |0> = cooperar,
        |1> = delatar.

    Por tanto,

        P_D = <1|rho_A|1> = rho_A[1,1].
    """

    # Elemento diagonal correspondiente al estado |1>.
    x = float(rho_a[1, 1].real)

    # Se recorta al intervalo físico [0,1] para absorber errores numéricos
    # del orden de la precisión de punto flotante.
    return float(np.clip(x, 0, 1)) if np.isfinite(x) else np.nan


def evaluate(point: ExperimentPoint) -> Probabilities:
    """
    Evalúa todas las probabilidades necesarias en un punto del experimento.

    Secuencia:
      1. Construir el entrelazador.
      2. Obtener el estado final puro.
      3. Construir rho_AB = |psi_f><psi_f|.
      4. Calcular los estados condicionales para B=0 y B=1.
      5. Calcular el estado actualizado para varrho(R,theta,phi).
      6. Extraer las probabilidades relevantes del QSTP.

    Returns
    -------
    Probabilities
        Tupla nombrada con P_D, las probabilidades condicionales y la marginal.
    """

    # Selección del operador de entrelazamiento.
    J = entangler(point.entangler, point.gamma)

    # Estado final del juego.
    psi = final_state(point.t_a, point.t_b, J)

    # Matriz de densidad conjunta pura.
    rho = np.outer(psi, psi.conj())

    # Estados de A condicionados a que B se proyecte en |0> o |1>.
    a0, pb0 = conditional_a(rho, P0)
    a1, pb1 = conditional_a(rho, P1)

    # Estado de A condicionado al estado mixto general de B.
    av = updated_a(
        rho,
        point.R,
        point.theta,
        point.phi,
    )

    # Empaquetado de las cuatro probabilidades relevantes.
    return Probabilities(
        # Probabilidad actualizada P_D.
        p_defect(av),

        # P(A=1 | B=0), si el condicionamiento tiene probabilidad no nula.
        p_defect(a0) if pb0 > EPS else np.nan,

        # P(A=1 | B=1), si el condicionamiento tiene probabilidad no nula.
        p_defect(a1) if pb1 > EPS else np.nan,

        # Probabilidad marginal P(A=1).
        float(np.trace(P_A1 @ rho).real),
    )


# =============================================================================
# EVALUACIÓN DEL QUANTUM SURE THING PRINCIPLE
# =============================================================================

def standard_violation(
    p: Probabilities,
    tol: float = TOL,
) -> bool:
    """
    Detecta una violación del QSTP estándar.

    La violación ocurre cuando ambas probabilidades condicionadas favorecen
    la defección,

        P(A=1|B=0) > 1/2,
        P(A=1|B=1) > 1/2,

    pero bajo incertidumbre sobre B,

        P_D < 1/2.
    """

    return (
        p.p_a1_b0 > 0.5 + tol
        and p.p_a1_b1 > 0.5 + tol
        and p.p_d < 0.5 - tol
    )


def complementary_violation(
    p: Probabilities,
    tol: float = TOL,
) -> bool:
    """
    Detecta una violación del QSTP complementario.

    La violación ocurre cuando ambas probabilidades condicionadas rechazan
    la defección,

        P(A=1|B=0) < 1/2,
        P(A=1|B=1) < 1/2,

    pero bajo incertidumbre sobre B,

        P_D > 1/2.
    """

    return (
        p.p_a1_b0 < 0.5 - tol
        and p.p_a1_b1 < 0.5 - tol
        and p.p_d > 0.5 + tol
    )


def violation_magnitude(
    p: Probabilities,
    kind: Literal["standard", "complementary"],
) -> float:
    """
    Cuantifica la intensidad de una violación del QSTP.

    Para la versión estándar:

        V_std = 1/2 - P_D,

    siempre que se cumplan los antecedentes condicionales.

    Para la versión complementaria:

        V_comp = P_D - 1/2.

    Si no existe violación, la función devuelve cero.
    """

    # Magnitud de la violación estándar.
    if kind == "standard" and standard_violation(p):
        return 0.5 - p.p_d

    # Magnitud de la violación complementaria.
    if kind == "complementary" and complementary_violation(p):
        return p.p_d - 0.5

    # No hay violación.
    return 0.0
