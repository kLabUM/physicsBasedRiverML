"""Figure 3. Steps 1 and 2: the roughness length, example log-law fits, and the survey-day amplitudes before and after
the stage law.

Inputs:  outputs/intermediate/readings.csv, locations.csv, log_law.json, z0_search.csv, stage_law.json
Output:  outputs/figures/fig3_steps1_2.png (and .pdf)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from riverml.config import WORK, FIGURES, Z_MIN as ZMIN, V_MIN as VMIN
H0=json.load(open(WORK/"stage_law.json"))["selected"]["h0"]
INK,INK2,MUTED,AXIS,GRID="#1f2937","#4b5563","#6b7280","#b9b8ae","#e7e6e0"; C1,C2="#2563eb","#0d9488"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],"font.size":6.5,"axes.labelsize":6.5,
 "xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":5.8,"legend.frameon":False,"axes.edgecolor":AXIS,"axes.linewidth":0.5,"grid.color":GRID,"grid.linewidth":0.35,
 "xtick.major.size":1.8,"ytick.major.size":1.8,"xtick.minor.size":1,"ytick.minor.size":1,"xtick.major.pad":1.5,"ytick.major.pad":1.5,"axes.labelpad":2})
tr=pd.read_csv(WORK/"readings.csv"); loc=pd.read_csv(WORK/"locations.csv"); Z0=json.load(open(WORK/"log_law.json"))["z0_m"]
use=tr[(tr.height_m>=ZMIN)&(tr.velocity_ms>VMIN)]
zs=pd.read_csv(WORK/"z0_search.csv"); zg=zs.z0_m.values; sse=zs.sse_m2_s2.values; zb=zg[sse.argmin()]; flat=zg[sse<=sse.min()*1.05]
locp=loc[loc.a>0].copy(); locp["A"]=locp.a/np.clip(locp.stage_m-H0,0.02,None)
W,H=7.2,2.2; fig=plt.figure(figsize=(W,H))
L,R,gap=0.06,0.99,0.075; w=(R-L-3*gap)/4; B,Hh=0.17,0.63
axs=[fig.add_axes([L+i*(w+gap),B,w,Hh]) for i in range(4)]
titles=["Roughness length z₀","Log-law fits","Before: amplitude a","After: rating coefficient A"]
cols=[C1,C1,C2,C2]
# a
ax=axs[0]; ax.axvspan(flat.min()*1e3,flat.max()*1e3,color=C1,alpha=0.13,lw=0); ax.plot(zg*1e3,sse,color=INK,lw=1); ax.axvline(zb*1e3,color=C1,lw=0.9)
ax.set_xscale("log"); ax.set_xlabel("z₀ (mm)"); ax.set_ylabel("SSR (m²/s²)")
ax.text(zb*1e3*0.55,sse.max()*0.93,f"{zb*1e3:.1f} mm",color=C1,fontsize=6,va="top",ha="right")
# b
ax=axs[1]; ex=loc[loc.n_used>=6].sample(12,random_state=3).location_id
for lid in ex:
    p=use[use.location_id==lid].sort_values("height_m"); a=loc.set_index("location_id").a[lid]; z=np.linspace(Z0*1.01,max(p.height_m.max(),0.3)*1.05,60)
    ax.plot(a*np.log(z/Z0),z,color=C1,lw=0.6,alpha=0.75); ax.scatter(p.velocity_ms,p.height_m,s=3,color=INK,alpha=0.65,lw=0,zorder=3)
ax.set_xlim(0,0.55); ax.set_ylim(0,0.8); ax.set_xlabel("velocity (m/s)"); ax.set_ylabel("height above bed (m)")
ax.legend(handles=[Line2D([],[],marker="o",ls="",ms=2.2,color=INK,label="readings"),Line2D([],[],color=C1,lw=0.8,label="v = a ln(z/z₀)")],loc="upper right",handletextpad=0.3,borderaxespad=0.2)
# c, d
dm=locp.groupby("date").agg(st=("stage_m","median"),yr=("year","first"),
    la=("a",lambda v:np.log(v).median()),la_lo=("a",lambda v:np.log(v).quantile(.25)),la_hi=("a",lambda v:np.log(v).quantile(.75)),
    lA=("A",lambda v:np.log(v).median()),lA_lo=("A",lambda v:np.log(v).quantile(.25)),lA_hi=("A",lambda v:np.log(v).quantile(.75)))
for ax,(m,lo,hi) in zip(axs[2:],[("la","la_lo","la_hi"),("lA","lA_lo","lA_hi")]):
    for yr,mk,fc in [(2025,"o",C2),(2026,"s","white")]:
        d=dm[dm.yr==yr]; ax.errorbar(d.st,d[m],yerr=[d[m]-d[lo],d[hi]-d[m]],fmt=mk,ms=3.2,mfc=fc,mec=C2,mew=0.8,ecolor=C2,elinewidth=0.6,capsize=0,label=f"{yr}")
    ax.set_xlabel("stage on survey day (m)"); ax.set_xlim(0.7,1.13); ax.set_ylim(-4.3,-0.8)
xx=np.linspace(0.71,1.12,40); axs[2].plot(xx,np.log(xx-H0)+dm.la.mean()-np.mean(np.log(dm.st-H0)),color=INK2,lw=0.7,ls="--",label="stage law")
axs[2].set_ylabel("ln a"); axs[3].set_ylabel("ln A")
axs[2].legend(loc="lower right",handletextpad=0.3,borderaxespad=0.2); axs[3].legend(loc="lower right",handletextpad=0.3,borderaxespad=0.2)
sd_b=dm.la.std(); sd_a=dm.lA.std()
for i,ax in enumerate(axs):
    ax.grid(True,which="major"); ax.set_axisbelow(True)
    ax.text(0,1.04,"abcd"[i],transform=ax.transAxes,fontweight="bold",fontsize=8.5,va="bottom",ha="left",color=INK)
    ax.text(0.09,1.04,titles[i],transform=ax.transAxes,fontweight="bold",fontsize=7,va="bottom",ha="left",color=cols[i])
# step bands
for k,(i0,i1,c,t) in enumerate([(0,1,C1,"Step 1  ·  Map each profile to one number"),(2,3,C2,"Step 2  ·  Remove the stage")]):
    x0=axs[i0].get_position().x0-0.045; x1=axs[i1].get_position().x1
    fig.patches.append(matplotlib.patches.FancyBboxPatch((x0,0.925),x1-x0,0.062,boxstyle="round,pad=0,rounding_size=0.006",fc=c,ec=c,transform=fig.transFigure,mutation_aspect=W/H))
    fig.text(x0+0.012,0.955,t,color="white",fontweight="bold",fontsize=7,va="center")
fig.savefig(FIGURES/"fig3_steps1_2.png",dpi=600,facecolor="white"); fig.savefig(FIGURES/"fig3_steps1_2.pdf",facecolor="white")
