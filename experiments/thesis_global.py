# Este script genera análisis globales para la tesis sobre el Quantum
# Sure Thing Principle (QSTP) en el Dilema del Prisionero cuántico de
# Eisert–Wilkens–Lewenstein (EWL).
#
# Estudios incluidos:
#
#   1. Interacción global entre pureza R y entrelazamiento gamma.
#   2. Mapas bidimensionales fase–incertidumbre (Phi, Theta).
#   3. Comparación cuantitativa de operadores de entrelazamiento.
#
# Operadores considerados:
#
#   • operador EWL original;
#   • CNOT;
#   • dCNOT;
#   • B-gate.
#
# El modo rápido emplea mallas moderadas. La opción --full incrementa
# la resolución para producir resultados más densos.
#
# La implementación científica original se conserva; únicamente se
# añadieron comentarios y documentación explicativa.
# =============================================================================


from pathlib import Path
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from qstp_ewl.core import ExperimentPoint,evaluate,violation_magnitude
from qstp_ewl.plotting import save

# Directorio raíz del proyecto.
ROOT=Path(__file__).resolve().parents[1]
# Carpeta donde se guardan las figuras de la tesis.
OUT=ROOT/"figures"/"thesis"
# Carpeta donde se guardan los archivos numéricos.
DATA=ROOT/"data"

# Identificadores internos de los operadores de entrelazamiento.
ENTANGLERS=["original","cnot","dcnot","bgate"]


# Calcula, para cada operador, la máxima violación encontrada sobre
# las variables internas Theta y Phi para cada par (gamma,R).
def maxima_gamma_purity(quick=True):
    # Seleccionar la resolución de gamma, R, Theta y Phi.
    ng,nr,nt,np_= (21,15,19,16) if quick else (51,41,41,40)
    # Dominio físico del parámetro de entrelazamiento.
    gammas=np.linspace(0,np.pi/2,ng)
    # Dominio físico de la pureza del estado.
    Rs=np.linspace(0,1,nr)
    # Ángulo polar de la observación o incertidumbre.
    thetas=np.linspace(0,np.pi,nt)
    # Fase azimutal; se evita duplicar 0 y 2pi.
    phis=np.linspace(0,2*np.pi,np_,endpoint=False)
    # Crear un panel para cada operador de entrelazamiento.
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    # Diccionario que almacenará las matrices calculadas.
    arrays={}
    # Recorrer simultáneamente los ejes y los operadores.
    for ax,name in zip(axes.flat,ENTANGLERS):
        # Matriz de máxima violación para cada par (R,gamma).
        M=np.zeros((nr,ng))
        # Original uses known (-,+) configuration; perfect operators are globally
        # evaluated over representative article configurations in comparison script.
        # Estrategias representativas: una para EWL y otra para
        # los operadores perfectos.
        ta,tb=(-.75,.3) if name=="original" else (.8,.565)
        # Recorrer todos los valores de pureza.
        for ir,R in enumerate(Rs):
            # Recorrer todos los valores de entrelazamiento.
            for ig,g in enumerate(gammas):
                # En EWL se barre gamma; los perfect entanglers se
                # evalúan en su configuración de entrelazamiento máximo.
                gg=g if name=="original" else np.pi/2
                # Mejor magnitud de violación hallada en Theta y Phi.
                best=0
                for th in thetas:
                    for ph in phis:
                        # Evaluar el punto completo del experimento.
                        p=evaluate(ExperimentPoint(ta,tb,R,th,ph,gg,name))
                        # Conservar la mayor violación estándar o complementaria.
                        best=max(best,violation_magnitude(p,"standard"),violation_magnitude(p,"complementary"))
                # Registrar el máximo correspondiente al par (R,gamma).
                M[ir,ig]=best
        # Guardar la matriz para exportarla posteriormente.
        arrays[name]=M
        # Representar gamma/pi en el eje horizontal y R en el vertical.
        im=ax.imshow(M,origin="lower",extent=[0,.5,0,1],aspect="auto")
        ax.set_title(name); ax.set_xlabel(r"$\gamma/\pi$"); ax.set_ylabel(r"$R$")
        fig.colorbar(im,ax=ax,label="Máxima violación")
    fig.suptitle("Interacción global entre pureza y entrelazamiento")
    fig.tight_layout(); save(fig,OUT/"T01_gamma_purity_comparison")
    # Exportar todas las matrices en formato NPZ comprimido.
    np.savez_compressed(DATA/"gamma_purity_comparison.npz",gamma=gammas,R=Rs,**arrays)


# Construye mapas de calor de la magnitud de violación en el plano
# de fase Phi e incertidumbre Theta para configuraciones representativas.
def phase_theta_maps(quick=True):
    # Resolución del mapa (Theta,Phi).
    nt,np_=(91,121) if quick else (181,241)
    # Construir las mallas angulares completas.
    ths=np.linspace(0,np.pi,nt); phs=np.linspace(0,2*np.pi,np_)
    # Configuraciones representativas y tipo de QSTP evaluado.
    configs={
      "original":(-.75,.3,"standard"),
      "cnot":(.8,.565,"standard"),
      "dcnot":(1,.455,"standard"),
      "bgate":(.72,-.34,"complementary")
    }
    # Crear un panel para cada operador de entrelazamiento.
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    for ax,(name,(ta,tb,kind)) in zip(axes.flat,configs.items()):
        # Matriz de magnitud de violación en el plano (Theta,Phi).
        M=np.zeros((nt,np_))
        for i,t in enumerate(ths):
            for j,p in enumerate(phs):
                # Evaluar un estado puro y entrelazamiento máximo.
                q=evaluate(ExperimentPoint(ta,tb,1,t,p,np.pi/2,name))
                # Guardar la magnitud correspondiente al tipo de QSTP.
                M[i,j]=violation_magnitude(q,kind)
        # Mostrar Phi/pi y Theta/pi como variables adimensionales.
        im=ax.imshow(M,origin="lower",extent=[0,2,0,1],aspect="auto")
        ax.set_title(name); ax.set_xlabel(r"$\Phi/\pi$"); ax.set_ylabel(r"$\Theta/\pi$")
        fig.colorbar(im,ax=ax,label="Magnitud")
    fig.suptitle("Papel de la fase y la incertidumbre para cada entrelazador")
    fig.tight_layout(); save(fig,OUT/"T02_phase_theta_comparison")


# Compara los operadores mediante su máxima violación estándar y
# complementaria, además del número de puntos violatorios encontrados.
def operator_ranking(quick=True):
    # Resolución usada para la comparación de operadores.
    nt,np_=(41,32) if quick else (81,64)
    ths=np.linspace(0,np.pi,nt); phs=np.linspace(0,2*np.pi,np_,endpoint=False)
    # Configuraciones representativas y tipo de QSTP evaluado.
    configs={
      "original":(-.75,.3,np.pi/2),
      "CNOT":(.8,.565,np.pi/2),
      "dCNOT":(1,.455,np.pi/2),
      "B-gate":(.72,-.34,np.pi/2)
    }
    # Cada fila contendrá las métricas de un operador.
    rows=[]
    # Evaluar cada operador en su configuración representativa.
    for name,(ta,tb,g) in configs.items():
        # Convertir el nombre mostrado al identificador interno.
        ent=name.lower().replace("-","")
        if ent=="bgate": ent="bgate"
        # Máximos estándar y complementario.
        bests=bestc=0
        # Conteo de puntos violatorios de cada tipo.
        count_s=count_c=0
        # Barrer once niveles de pureza.
        for R in np.linspace(0,1,11):
            for th in ths:
                for ph in phs:
                    p=evaluate(ExperimentPoint(ta,tb,R,th,ph,g,ent))
                    # Magnitud de la violación estándar.
                    vs=violation_magnitude(p,"standard")
                    # Magnitud de la violación complementaria.
                    vc=violation_magnitude(p,"complementary")
                    # Actualizar los máximos globales.
                    bests=max(bests,vs); bestc=max(bestc,vc)
                    # Contar los puntos con violación positiva.
                    count_s+=vs>0; count_c+=vc>0
        # Almacenar los resultados resumidos del operador.
        rows.append((name,bests,bestc,count_s,count_c))
    # Exportar la clasificación numérica a CSV.
    with open(DATA/"operator_ranking.csv","w",newline="",encoding="utf8") as f:
        w=csv.writer(f);w.writerow(["operator","max_standard","max_complementary","count_standard","count_complementary"]);w.writerows(rows)
    # Posiciones y ancho de las barras agrupadas.
    x=np.arange(len(rows)); width=.35
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(x-width/2,[r[1] for r in rows],width,label="Estándar")
    ax.bar(x+width/2,[r[2] for r in rows],width,label="Complementario")
    ax.set_xticks(x,[r[0] for r in rows]);ax.set_ylabel("Máxima magnitud")
    ax.set_title("Comparación de impacto entre operadores de entrelazamiento")
    ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout();save(fig,OUT/"T03_operator_ranking")


# Procesa los argumentos de línea de comandos y ejecuta los tres análisis.
def main():
    # Crear el analizador de argumentos.
    ap=argparse.ArgumentParser()
    # --full activa las mallas de mayor resolución.
    ap.add_argument("--full",action="store_true")
    args=ap.parse_args()
    # quick=True salvo que se solicite explícitamente --full.
    q=not args.full
    maxima_gamma_purity(q)
    phase_theta_maps(q)
    operator_ranking(q)

if __name__=="__main__": main()
