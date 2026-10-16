"""Figure 6. Step 5: mean nesting suitability E of the wetted reach through both seasons, and E against stage (the
ecological rating curve).

Inputs:  outputs/intermediate/suitability_2h.csv, suitability_curve.csv
Output:  outputs/figures/fig6_step5.png (and .pdf)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.patches as mp
from matplotlib.lines import Line2D
from riverml.config import WORK, FIGURES, SURVEYED
SURV=SURVEYED
INK,INK2,MUTED,AXIS,GRID="#1f2937","#4b5563","#6b7280","#b9b8ae","#e7e6e0"; C5="#16a34a"; CE="#15803d"; CS="#2563eb"; OUTHI,OUTLO="#1f2937","#9ca3af"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],"font.size":7.5,"axes.labelsize":7.5,
 "xtick.labelsize":7,"ytick.labelsize":7,"axes.edgecolor":AXIS,"axes.linewidth":0.5,"xtick.major.size":1.8,"ytick.major.size":1.8,
 "xtick.major.pad":1.5,"ytick.major.pad":1.5,"axes.labelpad":2,"xtick.color":INK,"ytick.color":INK})
E_=pd.read_csv(WORK/"suitability_2h.csv"); E_["time_utc"]=pd.to_datetime(E_.time_utc,utc=True); E_["t"]=mdates.date2num(E_.time_utc.dt.tz_convert(None)); R=E_
curve=pd.read_csv(WORK/"suitability_curve.csv")
W,H=7.2,3.0; fig=plt.figure(figsize=(W,H))
L,R_=0.09,0.635; pan=[(0.555,0.285),(0.135,0.285)]
def spines(a):
    for sp in a.spines.values(): sp.set_edgecolor(AXIS)
def seg(a,m,x,dt,y,h,col):
    idx=np.flatnonzero(m)
    if len(idx):
        br=np.flatnonzero(np.diff(idx)>1); s0=np.r_[idx[0],idx[br+1]]; s1=np.r_[idx[br],idx[-1]]
        a.broken_barh([(x[i],x[j]-x[i]+dt) for i,j in zip(s0,s1)],(y,h),facecolors=col,lw=0)
for k,yr in enumerate([2025,2026]):
    y0,h=pan[k]; ax=fig.add_axes([L,y0,R_-L,h]); e=E_[E_.time_utc.dt.year==yr]; r=R[R.time_utc.dt.year==yr]
    ax.fill_between(e.t,e.E_lo*100,e.E_hi*100,color=CE,alpha=0.22,lw=0)
    ax.plot(e.t,e.E_med*100,color=CE,lw=0.9)
    dt=np.median(np.diff(r.t)); st=r.stage_m.values
    seg(ax,st>SURV[1],r.t.values,dt,-11,6,OUTHI); seg(ax,st<SURV[0],r.t.values,dt,-11,6,OUTLO)
    ax.axhline(0,color=AXIS,lw=0.4)
    ax.set_ylim(-13,100); ax.set_yticks([0,50,100]); ax.set_ylabel("")
    ax.grid(True,axis="y",color=GRID,lw=0.5); ax.set_axisbelow(True)
    import matplotlib.patheffects as pe
    ax2=ax.twinx(); ax2.plot(r.t,st,color=INK,lw=0.6,path_effects=[pe.Stroke(linewidth=1.5,foreground='white',alpha=0.8),pe.Normal()])
    [ax2.axhline(v,color=INK2,lw=0.5,ls=(0,(3,2)),zorder=1) for v in SURV]
    ax2.set_ylim(-0.45,2.0); ax2.set_yticks([0.5,1.0,1.5]); ax2.set_ylabel("stage (m)",labelpad=2)
    spines(ax); spines(ax2)
    ax.set_xlim(r.t.min()-1.5,r.t.max()+1.5); ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b 1"))
    fig.text(L-0.055,y0+h+0.02,"ab"[k],fontweight="bold",fontsize=9.5,va="bottom",color=INK)
    fig.text(L-0.03,y0+h+0.02,f"{yr} Season",fontweight="bold",fontsize=8,va="bottom",color=C5)
# panel c
cx0,cw=0.755,0.225; ax=fig.add_axes([cx0,0.135,cw,0.705])
ax.axvspan(*SURV,color="#eceae3",lw=0,zorder=0)
ax.text(np.mean(SURV),3,"surveyed\nrange",ha="center",va="bottom",fontsize=6.3,color=MUTED,linespacing=0.95)
ax.fill_between(curve.stage_m,curve.E_lo*100,curve.E_hi*100,color=CE,alpha=0.22,lw=0)
ax.plot(curve.stage_m,curve.E_med*100,color=CE,lw=1.4)
ax.set_xlim(0.65,1.35); ax.set_ylim(0,100); ax.set_yticks([0,25,50,75,100]); ax.set_xticks([0.7,0.9,1.1,1.3])
ax.set_xlabel("stage (m)"); ax.set_ylabel("Mean Nesting Suitability E (%)",labelpad=1); ax.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); spines(ax)
fig.text(cx0-0.06,0.86,"c",fontweight="bold",fontsize=9.5,va="bottom",color=INK)
fig.text(cx0-0.035,0.86,"Ecological Rating Curve",fontweight="bold",fontsize=8,va="bottom",color=C5)
hs=[(Line2D([],[],color=CE,lw=1.2),mp.Patch(color=CE,alpha=0.22)),Line2D([],[],color=INK,lw=0.8),Line2D([],[],color=INK2,lw=0.6,ls=(0,(3,2))),mp.Patch(color=OUTHI),mp.Patch(color=OUTLO)]
from matplotlib.legend_handler import HandlerTuple
fig.legend(hs,["mean nesting suitability E (90% band)","stage",f"surveyed stage range ({SURV[0]:.2f}–{SURV[1]:.2f} m)","stage above range","stage below range"],
 handler_map={tuple:HandlerTuple(ndivide=None,pad=0)},loc="lower center",bbox_to_anchor=(0.5,-0.015),ncol=5,fontsize=6.3,frameon=False,handlelength=1.6,columnspacing=1.2,handletextpad=0.4)
fig.text(0.006,(0.135+0.84)/2,"Mean Nesting Suitability E (%)",rotation=90,va="center",ha="left",fontsize=7.5,color=INK)
fig.patches.append(mp.FancyBboxPatch((0.012,0.915),0.976,0.07,boxstyle="round,pad=0,rounding_size=0.006",fc=C5,ec=C5,transform=fig.transFigure,mutation_aspect=W/H))
fig.text(0.024,0.95,"Step 5  ·  Transform downstream",color="white",fontweight="bold",fontsize=8,va="center")
fig.savefig(FIGURES/"fig6_step5.png",dpi=600,facecolor="white"); fig.savefig(FIGURES/"fig6_step5.pdf",dpi=600,facecolor="white")
