"""Figure 5. Step 4: the median velocity of every section, every 2 h through both seasons, driven by the stage record.

Inputs:  outputs/intermediate/section_velocity_2h.csv, locations.csv
Output:  outputs/figures/fig5_step4.png (and .pdf)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.patches as mp
from riverml.config import WORK, FIGURES, SURVEYED
SURV=SURVEYED; OUTHI,OUTLO="#1f2937","#9ca3af"; INK,INK2,MUTED,AXIS="#1f2937","#4b5563","#6b7280","#b9b8ae"; C4="#ea580c"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],"font.size":7.5,"axes.labelsize":7.5,
 "xtick.labelsize":7,"ytick.labelsize":7,"axes.edgecolor":AXIS,"axes.linewidth":0.5,"xtick.major.size":1.8,"ytick.major.size":1.8,"xtick.major.pad":1.5,"ytick.major.pad":1.5,"axes.labelpad":2})
loc=pd.read_csv(WORK/"locations.csv"); loc=loc[loc.a>0]
order=list(loc.groupby("transect").y_m.median().sort_values(ascending=False).index)
ss=pd.read_csv(WORK/"section_velocity_2h.csv",parse_dates=["time_utc"]); ss["time_utc"]=pd.to_datetime(ss.time_utc,utc=True)
W,H=7.2,3.0; fig=plt.figure(figsize=(W,H))
L,R=0.075,0.875; pan=[(0.555,0.285),(0.135,0.285)]
for k,yr in enumerate([2025,2026]):
    y0,h=pan[k]; ax=fig.add_axes([L,y0,R-L,h]); s=ss[ss.time_utc.dt.year==yr]
    piv=s.pivot(index="transect",columns="time_utc",values="mean_v_med").reindex(order); tt=piv.columns
    xe=mdates.date2num(tt.tz_convert(None)); xe=np.r_[xe,xe[-1]+(xe[-1]-xe[-2])]
    im=ax.pcolormesh(xe,np.arange(len(order)+1),piv.values,cmap="Oranges",vmin=0,vmax=0.6,shading="flat",rasterized=True)
    ax.set_yticks([0.5,len(order)-0.5]); ax.set_yticklabels(["T1\nnorth","T20\nsouth"],linespacing=0.95); ax.set_ylim(len(order)+1.5,0); ax.set_ylabel("River Section",labelpad=1)
    st=s[s.transect==order[0]]; xt=mdates.date2num(st.time_utc.dt.tz_convert(None))
    dt=np.median(np.diff(xt))
    for m,col in [(st.stage_m.values>SURV[1],OUTHI),(st.stage_m.values<SURV[0],OUTLO)]:
        idx=np.flatnonzero(m)
        if len(idx):
            br=np.flatnonzero(np.diff(idx)>1); starts=np.r_[idx[0],idx[br+1]]; ends=np.r_[idx[br],idx[-1]]
            ax.broken_barh([(xt[a],xt[b]-xt[a]+dt) for a,b in zip(starts,ends)],(len(order)+0.35,0.9),facecolors=col,lw=0)
    ax2=ax.twinx(); import matplotlib.patheffects as pe; ax2.plot(xt,st.stage_m,color=INK,lw=0.6,path_effects=[pe.Stroke(linewidth=1.5,foreground='white',alpha=0.8),pe.Normal()]); ax2.set_ylim(0.3,1.9); [ax2.axhline(v,color=INK2,lw=0.5,ls=(0,(3,2)),zorder=1) for v in SURV]; ax2.set_yticks([0.5,1.0,1.5]); ax2.set_ylabel("stage (m)",labelpad=2)
    for a in (ax,ax2):
        for sp in a.spines.values(): sp.set_edgecolor(AXIS)
    pad=1.5; x0,x1=xe[0]-pad,xe[-1]+pad
    ax.set_xlim(x0,x1); ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b 1"))
    ax.set_facecolor("#f3f4f6")
    fig.text(L-0.06,y0+h+0.02,"ab"[k],fontweight="bold",fontsize=9.5,va="bottom",color=INK)
    fig.text(L-0.035,y0+h+0.02,f"{yr} Season",fontweight="bold",fontsize=8,va="bottom",color=C4)
cax=fig.add_axes([0.935,0.25,0.012,0.5]); cb=fig.colorbar(im,cax=cax,ticks=[0,0.2,0.4,0.6]); cb.ax.tick_params(labelsize=6.5,length=1.5,pad=1); cb.set_label("section mean velocity (m/s)",fontsize=7,labelpad=2); cb.outline.set_linewidth(0.3)
# legend for stage line and out-of-range ticks
from matplotlib.lines import Line2D
h_=[Line2D([],[],color=INK,lw=0.8),Line2D([],[],color=INK2,lw=0.6,ls=(0,(3,2))),mp.Patch(color=OUTHI),mp.Patch(color=OUTLO),mp.Patch(fc="#f3f4f6",ec=AXIS,lw=0.3)]
fig.legend(h_,["stage",f"surveyed stage range ({SURV[0]:.2f}–{SURV[1]:.2f} m)","stage above range","stage below range","no stage record"],loc="lower center",bbox_to_anchor=(0.475,-0.015),ncol=5,fontsize=6.5,frameon=False,handlelength=1.4,columnspacing=1.2,handletextpad=0.4)
fig.patches.append(mp.FancyBboxPatch((0.012,0.915),0.976,0.07,boxstyle="round,pad=0,rounding_size=0.006",fc=C4,ec=C4,transform=fig.transFigure,mutation_aspect=W/H))
fig.text(0.024,0.95,"Step 4  ·  Extend through time",color="white",fontweight="bold",fontsize=8,va="center")
fig.savefig(FIGURES/"fig5_step4.png",dpi=600,facecolor="white"); fig.savefig(FIGURES/"fig5_step4.pdf",facecolor="white",dpi=600)
