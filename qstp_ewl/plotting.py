
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .core import ExperimentPoint, evaluate, standard_violation, complementary_violation
from .configs import Panel

PI=np.pi

def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)

def references(panel: Panel, x: np.ndarray, variable: str):
    vals=[]
    for z in x:
        gamma=z if variable=="gamma" else np.pi/2
        theta=z if variable=="theta" else np.pi/2
        p=evaluate(ExperimentPoint(panel.t_a,panel.t_b,1,theta,np.pi,gamma,panel.entangler))
        vals.append(p)
    return vals

def article_original_panel(ax, panel: Panel, thetas, phi=np.pi, n=501):
    gammas=np.linspace(0,PI/2,n)
    ref=[evaluate(ExperimentPoint(panel.t_a,panel.t_b,1,thetas[0],phi,g,panel.entangler)) for g in gammas]
    ax.plot(gammas,[p.p_a1_b1 for p in ref],color="#6064ff",lw=2.3)
    ax.plot(gammas,[p.p_a1_b0 for p in ref],color="#ff6666",lw=2.3)
    ax.plot(gammas,[p.p_a1_unconditional for p in ref],color="#ff7600",lw=3,ls=":")
    grays=["#000000","#555555","#999999"]
    allpd=[]; masks=[]
    for th,gc in zip(thetas,grays):
        probs=[evaluate(ExperimentPoint(panel.t_a,panel.t_b,1,th,phi,g,panel.entangler)) for g in gammas]
        pd=np.array([p.p_d for p in probs]); allpd.append(pd)
        mask=np.array([(standard_violation(p) if panel.kind=="standard" else complementary_violation(p)) for p in probs])
        masks.append(mask)
        ax.plot(gammas,pd,color=gc,lw=2)
    union=np.logical_or.reduce(masks)
    if panel.kind=="standard":
        ax.fill_between(gammas,np.minimum.reduce(allpd),.5,where=union,color="#c8f7c5",alpha=.75)
    else:
        ax.fill_between(gammas,.5,np.maximum.reduce(allpd),where=union,color="#c8f7c5",alpha=.75)
    ax.set(xlim=(0,PI/2),ylim=(-.02,1.02),xlabel=r"$\gamma$",ylabel=r"$P_D$",title=panel.title)
    ax.minorticks_on()

def perfect_panel(ax, panel: Panel, n=501):
    theta=np.linspace(0,PI,n)
    ref=[evaluate(ExperimentPoint(panel.t_a,panel.t_b,1,t,0,PI/2,panel.entangler)) for t in theta]
    ax.plot(theta,[p.p_a1_b1 for p in ref],color="#6064ff",lw=2)
    ax.plot(theta,[p.p_a1_b0 for p in ref],color="#ff6666",lw=2)
    ax.plot(theta,[p.p_a1_unconditional for p in ref],color="#ff7600",lw=2.6,ls=":")
    curves=[]; masks=[]
    for c in panel.curves:
        probs=[evaluate(ExperimentPoint(panel.t_a,panel.t_b,c.R,t,c.phi,PI/2,panel.entangler)) for t in theta]
        pd=np.array([p.p_d for p in probs]); curves.append((c,pd))
        mask=np.array([(standard_violation(p) if panel.kind=="standard" else complementary_violation(p)) for p in probs])
        masks.append(mask)
        ax.plot(theta,pd,color=c.gray,ls=c.style,lw=2)
    union=np.logical_or.reduce(masks)
    if panel.kind=="standard":
        ax.fill_between(theta,np.minimum.reduce([x[1] for x in curves]),.5,where=union,color="#c8f7c5",alpha=.75)
    else:
        ax.fill_between(theta,.5,np.maximum.reduce([x[1] for x in curves]),where=union,color="#c8f7c5",alpha=.75)
    # label curves near representative positions
    xs=np.linspace(.22*PI,.78*PI,len(curves))
    for (c,pd),xp in zip(curves,xs):
        yp=np.interp(xp,theta,pd)
        ax.text(xp,yp+.035,c.label,fontsize=8)
    ax.set(xlim=(0,PI),ylim=(-.02,1.02),xlabel=r"$\Theta$",ylabel=r"$P_D$",title=panel.title)
    ax.minorticks_on()
