
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PI=np.pi

@dataclass(frozen=True)
class Curve:
    R: float
    phi: float
    label: str
    style: str = "-"
    gray: str = "black"

@dataclass(frozen=True)
class Panel:
    title: str
    entangler: str
    kind: str
    t_a: float
    t_b: float
    curves: tuple[Curve,...]


ORIGINAL_STANDARD = Panel(
    "EWL (-+): QSTP estándar","original","standard",-0.75,0.30,
    tuple()
)
ORIGINAL_COMPLEMENTARY = Panel(
    "EWL (-+): QSTP complementario","original","complementary",-0.25,0.30,
    tuple()
)

FIG6_STANDARD = (
    Panel("CNOT (++)","cnot","standard",0.8,0.565,(
        Curve(0,3*PI/2,r"$\Phi_0=3\pi/2$","-","#777777"),
        Curve(0,PI/2,r"$\Phi_0=\pi/2$","-","#000000"),
        Curve(1,3*PI/2,r"$\Phi_1=3\pi/2$","--","#000000"),
    )),
    Panel("CNOT (+−)","cnot","standard",0.355,-0.7,(
        Curve(0,PI/2,r"$\Phi_0=\pi/2$","-","#000000"),
        Curve(0,3*PI/2,r"$\Phi_0=3\pi/2$","-","#888888"),
        Curve(1,PI/2,r"$\Phi_1=\pi/2$","--","#000000"),
    )),
    Panel("dCNOT (++)","dcnot","standard",1.0,0.455,(
        Curve(0,0,r"$\Phi_0=0$","-","#000000"),
        Curve(0,PI/2,r"$\Phi_0=\pi/2$","-","#888888"),
        Curve(1,0,r"$\Phi_1=0$","--","#000000"),
    )),
)

FIG7_COMPLEMENTARY = (
    Panel("CNOT (++)","cnot","complementary",0.8,0.435,(
        Curve(0,PI/2,r"$\Phi_0=\pi/2$","-","#000000"),
        Curve(0,3*PI/2,r"$\Phi_0=3\pi/2$","-","#888888"),
        Curve(1,PI/2,r"$\Phi_1=\pi/2$","--","#000000"),
    )),
    Panel("CNOT (+−)","cnot","complementary",0.355,-0.3,(
        Curve(0,3*PI/2,r"$\Phi_0=3\pi/2$","-","#000000"),
        Curve(0,PI/2,r"$\Phi_0=\pi/2$","-","#888888"),
        Curve(1,PI/2,r"$\Phi_1=\pi/2$","--","#000000"),
    )),
    Panel("dCNOT (++,+−,−+)","dcnot","complementary",0.0,0.544,(
        Curve(0,0,r"$\Phi_0=0$","-","#000000"),
        Curve(0,PI,r"$\widetilde{\Phi}_0=\pi$","-","#888888"),
        Curve(1,0,r"$\Phi_1=0$","--","#000000"),
    )),
    Panel("B-gate (+−)","bgate","complementary",0.72,-0.34,(
        Curve(0,PI/2,r"$\Phi_0=\pi/2$","-","#000000"),
        Curve(0,3*PI/2,r"$\Phi_0=3\pi/2$","-","#888888"),
        Curve(1,PI/2,r"$\Phi_1=\pi/2$","--","#000000"),
    )),
)
