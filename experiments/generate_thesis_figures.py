# Este script genera las figuras globales de la tesis relacionadas con el
# Quantum Sure Thing Principle (QSTP) en el Dilema del Prisionero cuántico
# de Eisert–Wilkens–Lewenstein (EWL).
#
# Figuras generadas:
#
#   T04_violation_volume
#       Comparación del volumen normalizado, volumen ponderado y máxima
#       intensidad de violación para todos los operadores.
#
#   T05_critical_purity_entanglement
#       Perfiles condicionados por pureza R y entrelazamiento gamma.
#
#   T06_parameter_robustness
#       Robustez de configuraciones violatorias frente a perturbaciones.
#
#   T07_phase_theta_comparison
#       Mapas de violación en el plano fase–incertidumbre (Phi,Theta).
#
#   T08_strategy_plane_global
#       Mapas globales en el espacio estratégico (t_A,t_B), maximizados
#       sobre Theta y Phi.
#
# Operadores considerados:
#   • EWL original
#   • CNOT
#   • dCNOT
#   • B-gate
#
# La opción --full aumenta el número de muestras y la resolución espacial.
# La lógica científica original se conserva; únicamente se añadieron
# comentarios y documentación explicativa.
# =============================================================================


from __future__ import annotations

from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from qstp_ewl.core import PI, ExperimentPoint, evaluate, violation_magnitude
from qstp_ewl.metrics import (
    estimate_violation_volume,
    estimate_volume_profile,
    critical_threshold,
    robustness_curve,
)
from qstp_ewl.plotting import save

# Directorio raíz del repositorio.
ROOT = Path(__file__).resolve().parents[1]
# Carpeta donde se guardan las figuras de la tesis.
OUT = ROOT / "figures" / "thesis"
# Carpeta donde se guardan los resultados numéricos.
DATA = ROOT / "data"

# Identificadores internos de los operadores de entrelazamiento.
OPERATORS = ["original", "cnot", "dcnot", "bgate"]
# Etiquetas legibles utilizadas en las figuras.
DISPLAY = {"original":"EWL", "cnot":"CNOT", "dcnot":"dCNOT", "bgate":"B-gate"}
# Dos versiones del Quantum Sure Thing Principle.
KINDS = ["standard", "complementary"]

# Configuraciones representativas tomadas del análisis del artículo.
# Se usan como puntos centrales en estudios locales de robustez y fase.
ARTICLE_CONFIGS = {
    ("original","standard"): ExperimentPoint(-0.75,0.30,1.0,PI/2,PI,PI/2,"original"),
    ("original","complementary"): ExperimentPoint(-0.25,0.30,1.0,19*PI/30,PI,PI/2,"original"),
    ("cnot","standard"): ExperimentPoint(0.8,0.565,1.0,PI/2,3*PI/2,PI/2,"cnot"),
    ("cnot","complementary"): ExperimentPoint(0.8,0.435,1.0,PI/2,PI/2,PI/2,"cnot"),
    ("dcnot","standard"): ExperimentPoint(1.0,0.455,1.0,PI/2,0.0,PI/2,"dcnot"),
    ("dcnot","complementary"): ExperimentPoint(0.0,0.544,1.0,PI/2,0.0,PI/2,"dcnot"),
    ("bgate","complementary"): ExperimentPoint(0.72,-0.34,1.0,PI/2,PI/2,PI/2,"bgate"),
}


# Genera la Figura T04 mediante estimaciones globales del volumen
# de violación estándar y complementaria para cada operador.
def volume_figure(samples:int, seed:int):
    # Lista con los resultados de volumen de todos los operadores.
    rows=[]
    # Recorrer cada operador de entrelazamiento.
    for oi,op in enumerate(OPERATORS):
        # Evaluar las versiones estándar y complementaria.
        for ki,kind in enumerate(KINDS):
            # Estimar por Monte Carlo la fracción violatoria del dominio.
            r=estimate_violation_volume(op,kind,samples=samples,seed=seed+100*oi+ki)
            rows.append(r)
    # Exportar los resultados completos de volumen a CSV.
    with (DATA/"T04_violation_volume.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["operator","kind","samples","fraction","ci_low","ci_high","weighted","mean_given_violation","maximum","argmax"])
        for r in rows:
            w.writerow([r.operator,r.kind,r.samples,r.violation_fraction,r.ci_low,r.ci_high,
                        r.weighted_volume,r.mean_intensity_given_violation,r.maximum_intensity,repr(r.maximum_point)])

    # Posiciones horizontales y ancho de las barras agrupadas.
    x=np.arange(len(OPERATORS)); width=.36
    # Crear tres paneles: volumen, volumen ponderado y máximo.
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    # Desplazar las barras para comparar ambos tipos de QSTP.
    for offset,kind in [(-width/2,"standard"),(width/2,"complementary")]:
        rr=[r for r in rows if r.kind==kind]
        # Fracción estimada del dominio que viola el principio.
        frac=np.array([r.violation_fraction for r in rr])
        # Límites inferior y superior del intervalo de confianza.
        lo=np.array([r.ci_low for r in rr]); hi=np.array([r.ci_high for r in rr])
        axes[0].bar(x+offset,frac,width,label=kind)
        # Añadir barras de error asimétricas.
        axes[0].errorbar(x+offset,frac,yerr=np.vstack([frac-lo,hi-frac]),fmt="none",capsize=3)
        axes[1].bar(x+offset,[r.weighted_volume for r in rr],width,label=kind)
        axes[2].bar(x+offset,[r.maximum_intensity for r in rr],width,label=kind)
    for ax in axes:
        ax.set_xticks(x,[DISPLAY[o] for o in OPERATORS])
        ax.grid(axis="y",alpha=.2); ax.legend(fontsize=8)
    axes[0].set_title("Volumen normalizado"); axes[0].set_ylabel("Fracción del dominio")
    axes[1].set_title("Volumen ponderado"); axes[1].set_ylabel("Extensión × intensidad")
    axes[2].set_title("Violación máxima"); axes[2].set_ylabel("Magnitud")
    fig.suptitle("Comparación global de operadores de entrelazamiento")
    save(fig,OUT/"T04_violation_volume")


# Genera la Figura T05 a partir de perfiles condicionados por pureza R
# y, para el operador EWL original, por entrelazamiento gamma.
def critical_figure(samples_per_value:int, seed:int):
    # Malla de pureza R.
    Rvals=np.linspace(0,1,21)
    # Malla del parámetro de entrelazamiento gamma.
    Gvals=np.linspace(0,PI/2,21)
    # Diccionario con los perfiles condicionados calculados.
    profiles={}
    # Tabla resumida con los valores críticos.
    summary=[]
    # Recorrer cada operador de entrelazamiento.
    for oi,op in enumerate(OPERATORS):
        # Evaluar las versiones estándar y complementaria.
        for ki,kind in enumerate(KINDS):
            # Estimar el volumen de violación condicionado a cada valor de R.
            p=estimate_volume_profile(op,kind,"R",Rvals,samples_per_value,seed+1000+100*oi+ki)
            profiles[(op,kind,"R")]=p
            # Registrar el primer R cuya fracción alcanza el umbral de 0.5%.
            summary.append((op,kind,"R",critical_threshold(Rvals,p["fraction"],.005)))
        # Evaluar las versiones estándar y complementaria.
    for ki,kind in enumerate(KINDS):
        # Estimar el perfil en gamma para el operador EWL original.
        p=estimate_volume_profile("original",kind,"gamma",Gvals,samples_per_value,seed+2000+ki)
        profiles[("original",kind,"gamma")]=p
        # Registrar el valor crítico operativo de gamma.
        summary.append(("original",kind,"gamma",critical_threshold(Gvals,p["fraction"],.005)))

    # Exportar los valores críticos a CSV.
    with (DATA/"T05_critical_parameters.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["operator","kind","parameter","critical_value"]); w.writerows(summary)

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for op in OPERATORS:
        for kind,ls in [("standard","-"),("complementary","--")]:
            p=profiles[(op,kind,"R")]
            axes[0].plot(p["values"],p["fraction"],ls,label=f"{DISPLAY[op]} {kind}")
    axes[0].set(xlabel=r"Pureza $R$",ylabel="Volumen condicionado",title="Pureza crítica")
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=7,ncol=2)
    for kind,ls in [("standard","-"),("complementary","--")]:
        p=profiles[("original",kind,"gamma")]
        axes[1].plot(p["values"]/PI,p["fraction"],ls,lw=2,label=kind)
    axes[1].set(xlabel=r"$\gamma/\pi$",ylabel="Volumen condicionado",title="Entrelazamiento crítico (EWL original)")
    axes[1].grid(alpha=.2); axes[1].legend()
    save(fig,OUT/"T05_critical_purity_entanglement")


# Genera la Figura T06 evaluando la persistencia de las violaciones
# cuando se perturban simultáneamente los parámetros del experimento.
def robustness_figure(samples:int,seed:int):
    # Radios relativos de perturbación entre 0% y 15%.
    radii=np.linspace(0,.15,13)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    # Diccionario con los datos numéricos que se exportarán.
    saved={"radius":radii}
    # Evaluar la robustez alrededor de cada configuración representativa.
    for idx,((op,kind),point) in enumerate(ARTICLE_CONFIGS.items()):
        # Calcular supervivencia e intensidad media bajo perturbaciones.
        c=robustness_curve(point,kind,radii,samples,seed+3000+idx)
        label=f"{DISPLAY[op]} {kind}"
        # Probabilidad de conservar la violación.
        axes[0].plot(radii,c["survival"],label=label)
        # Intensidad media de la violación restante.
        axes[1].plot(radii,c["mean_intensity"],label=label)
        saved[f"{op}_{kind}_survival"]=c["survival"]
        saved[f"{op}_{kind}_mean"]=c["mean_intensity"]
    axes[0].set(xlabel="Radio relativo de perturbación",ylabel="Probabilidad de conservar la violación",
                title="Robustez de la región violatoria",ylim=(-.02,1.02))
    axes[1].set(xlabel="Radio relativo de perturbación",ylabel="Intensidad media",
                title="Degradación de la intensidad")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    # Guardar todas las curvas de robustez en formato NPZ comprimido.
    np.savez_compressed(DATA/"T06_robustness_curves.npz",**saved)
    save(fig,OUT/"T06_parameter_robustness")


# Genera la Figura T07 en el plano bidimensional (Phi,Theta) para
# configuraciones representativas de cada operador.
def phase_theta_figure(n:int):
    # Configuración representativa elegida para cada operador.
    configs={
        "EWL":ARTICLE_CONFIGS[("original","standard")],
        "CNOT":ARTICLE_CONFIGS[("cnot","standard")],
        "dCNOT":ARTICLE_CONFIGS[("dcnot","standard")],
        "B-gate":ARTICLE_CONFIGS[("bgate","complementary")],
    }
    # Tipo de QSTP mostrado para cada operador.
    kinds={"EWL":"standard","CNOT":"standard","dCNOT":"standard","B-gate":"complementary"}
    # Construir las mallas uniformes de Theta y Phi.
    th=np.linspace(0,PI,n); ph=np.linspace(0,2*PI,n)
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for ax,(name,ref) in zip(axes.flat,configs.items()):
        # Matriz de magnitud de violación.
        M=np.zeros((n,n))
        for i,t in enumerate(th):
            for j,p in enumerate(ph):
                # Sustituir únicamente Theta y Phi en la configuración.
                q=ExperimentPoint(ref.t_a,ref.t_b,ref.R,t,p,ref.gamma,ref.entangler)
                # Evaluar y almacenar la magnitud de violación.
                M[i,j]=violation_magnitude(evaluate(q),kinds[name])
        # Mostrar Phi/pi en x y Theta/pi en y.
        im=ax.imshow(M,origin="lower",extent=[0,2,0,1],aspect="auto")
        ax.set_title(name); ax.set_xlabel(r"$\Phi/\pi$"); ax.set_ylabel(r"$\Theta/\pi$")
        fig.colorbar(im,ax=ax,label="Magnitud de violación")
    fig.suptitle("Estructura de las violaciones en el plano fase–incertidumbre")
    save(fig,OUT/"T07_phase_theta_comparison")


# Genera la Figura T08 en el plano estratégico (t_A,t_B), conservando
# para cada punto la mayor violación hallada sobre Theta y Phi.
def strategy_plane_figure(n:int):
    # Malla de estrategias locales t_A y t_B.
    ts=np.linspace(-1,1,n)
    # Orden de los operadores en la cuadrícula de paneles.
    operators=["original","cnot","dcnot","bgate"]
    fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    # Malla interna sobre la que se maximiza la violación.
    ths=np.linspace(0,PI,13); phs=np.linspace(0,2*PI,12,endpoint=False)
    for ax,op in zip(axes.flat,operators):
        # Matriz de magnitud de violación.
        M=np.zeros((n,n))
        # Todos los operadores se evalúan con entrelazamiento máximo.
        gamma=PI/2
        for iy,tb in enumerate(ts):
            for ix,ta in enumerate(ts):
                # Máxima violación hallada para el par estratégico actual.
                best=0
                for th in ths:
                    for ph in phs:
                        p=evaluate(ExperimentPoint(ta,tb,1,th,ph,gamma,op))
                        # Comparar simultáneamente ambas versiones del QSTP.
                        best=max(best,violation_magnitude(p,"standard"),violation_magnitude(p,"complementary"))
                # Guardar el máximo para el punto (t_A,t_B).
                M[iy,ix]=best
        im=ax.imshow(M,origin="lower",extent=[-1,1,-1,1],aspect="equal")
        # Marcar los ejes que separan los cuatro cuadrantes estratégicos.
        ax.axvline(0,color="white",ls="--",lw=.8); ax.axhline(0,color="white",ls="--",lw=.8)
        ax.set_title(DISPLAY[op]); ax.set_xlabel(r"$t_A$"); ax.set_ylabel(r"$t_B$")
        fig.colorbar(im,ax=ax,label="Máxima violación")
    fig.suptitle(r"Mapa estratégico global, maximizado sobre $(\Theta,\Phi)$")
    save(fig,OUT/"T08_strategy_plane_global")


# Procesa los argumentos de línea de comandos y ejecuta toda la
# tubería de generación de figuras de la tesis.
def main():
    # Crear el analizador de argumentos.
    ap=argparse.ArgumentParser()
    # --full activa el modo de alta resolución.
    ap.add_argument("--full",action="store_true")
    # Semilla reproducible para los cálculos Monte Carlo.
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    # Crear automáticamente los directorios de salida.
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    # Parámetros numéricos para el modo de alta resolución.
    if args.full:
        # Mayor número de muestras y mallas más densas.
        volume_samples=30000; profile_samples=2500; robustness_samples=5000; nmap=121; nstrategy=41
    else:
        # Configuración rápida para pruebas y reproducción preliminar.
        volume_samples=5000; profile_samples=350; robustness_samples=800; nmap=61; nstrategy=25
    # Generar la Figura T04.
    volume_figure(volume_samples,args.seed)
    # Generar la Figura T05.
    critical_figure(profile_samples,args.seed)
    # Generar la Figura T06.
    robustness_figure(robustness_samples,args.seed)
    # Generar la Figura T07.
    phase_theta_figure(nmap)
    # Generar la Figura T08.
    strategy_plane_figure(nstrategy)

if __name__=="__main__":
    main()
