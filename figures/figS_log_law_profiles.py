"""Supplementary Figures S1-S6. The log-law fit at every surveyed location, sections T1 (north) to T20 (south),
48 locations per page.

Inputs:  outputs/intermediate/readings.csv, locations.csv, log_law.json, sections.csv
Output:  outputs/figures/figS1_log_law_profiles.png ... figS6_log_law_profiles.png (and .pdf)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mp
from riverml.config import WORK, FIGURES, Z_MIN, V_MIN, Z_REF
z0=json.load(open(WORK/"log_law.json"))["z0_m"]; ZMIN,VMIN,ZREF=Z_MIN,V_MIN,Z_REF
INK,MUTED,AXIS,GRID="#1f2937","#6b7280","#b9b8ae","#e7e6e0"; BLUE="#2563eb"; ORANGE="#ea580c"; RED="#dc2626"; WATER="#eef4fb"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],"font.size":6,"axes.edgecolor":AXIS,"axes.linewidth":0.4,
 "xtick.labelsize":5.5,"ytick.labelsize":5.5,"xtick.major.size":1.5,"ytick.major.size":1.5,"xtick.major.pad":1,"ytick.major.pad":1,"xtick.major.width":0.4,"ytick.major.width":0.4})
loc=pd.read_csv(WORK/"locations.csv"); tr=pd.read_csv(WORK/"readings.csv")
tm=pd.read_csv(WORK/"sections.csv").set_index("transect")
loc["T"]=loc.transect.map(tm.section); loc["Tn"]=loc["T"].str[1:].astype(int)
loc=loc.sort_values(["Tn","date","across_m"]).reset_index(drop=True); dup=loc.duplicated(["T","point"],keep=False)
loc["lab"]=loc["T"]+" · "+loc.point.astype(str)+np.where(dup," ("+pd.to_datetime(loc.date).map(lambda d: f"{d:%b} {d.day}")+")","")
per=48; pages=[loc.iloc[i:i+per] for i in range(0,len(loc),per)]
XL,YL=(-0.02,0.9),(0,1.1)
for pi,pg in enumerate(pages):
    nc=6; nr=-(-len(pg)//nc); H=8.3*0.13+nr*8.3*0.87/8; fig=plt.figure(figsize=(6.5,H))
    gs=fig.add_gridspec(nr,nc,left=0.075,right=0.99,top=1-8.3*0.075/H,bottom=8.3*0.055/H,hspace=0.62,wspace=0.12)
    for k,(_,r) in enumerate(pg.iterrows()):
        i,j=divmod(k,nc); ax=fig.add_subplot(gs[i,j])
        p=tr[tr.location_id==r.location_id]; ok=(p.height_m>=ZMIN)&(p.velocity_ms>VMIN)
        ax.axhspan(0,r.water_depth_m,color=WATER,lw=0,zorder=0); ax.axhline(r.water_depth_m,color="#60a5fa",lw=0.6,zorder=1)
        ax.axhline(ZREF,color=MUTED,lw=0.4,ls=(0,(2,1.5)),zorder=1)
        single=r.n_used==1
        if r.a>0 and r.n_used>0:
            z=np.linspace(z0*1.02,max(r.water_depth_m,p.height_m.max())*1.0,60)
            ax.fill_betweenx(z,(r.a-1.96*r.a_sd)*np.log(z/z0),(r.a+1.96*r.a_sd)*np.log(z/z0),color=BLUE,alpha=0.16,lw=0,zorder=2)
            ax.plot(r.a*np.log(z/z0),z,color=BLUE,lw=0.9,zorder=3)
        ax.scatter(p.velocity_ms[ok],p.height_m[ok],s=5,color=ORANGE if single else INK,zorder=4,lw=0)
        ax.scatter(p.velocity_ms[~ok],p.height_m[~ok],s=9,facecolor="none",edgecolor=RED,lw=0.6,zorder=5)
        ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_xticks([0,0.4,0.8]); ax.set_yticks([0,0.5,1.0])
        ax.grid(True,color=GRID,lw=0.35); ax.set_axisbelow(True)
        if j>0: ax.tick_params(labelleft=False)
        if i<nr-1 and k+nc<len(pg): ax.tick_params(labelbottom=False)
        tags=[]
        if r.n_used==0: tags.append("no usable reading")
        elif single: tags.append("1 reading")
        if r.fit_r2<0: tags.append("R² < 0")
        if r.shallower_than_1ft: tags.append("< 1 ft deep")
        line2=(f"R² {r.fit_r2:.2f} · " if pd.notna(r.fit_r2) else "")+rf"$\bar{{v}}$ {r.v_mean:.2f} m/s"
        ax.set_title(r.lab,fontsize=6,color=INK,pad=7.5,fontweight="bold")
        ax.text(0.5,1.015,line2,transform=ax.transAxes,ha="center",va="bottom",fontsize=5.3,color=MUTED)
        if tags: ax.text(0.97,0.96,", ".join(tags),transform=ax.transAxes,ha="right",va="top",fontsize=5,color=ORANGE if single else RED if r.n_used==0 else INK,
                        bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.85),zorder=6)
        for sp in ax.spines.values(): sp.set_edgecolor(AXIS)
    fig.text(0.53,8.3*0.018/H,"velocity (m/s)",ha="center",fontsize=7); fig.text(0.012,(8.3*0.055/H+1-8.3*0.075/H)/2,"height above bed (m)",rotation=90,va="center",fontsize=7)
    h=[Line2D([],[],marker="o",ls="",color=INK,ms=2.8),Line2D([],[],marker="o",ls="",color=ORANGE,ms=2.8),Line2D([],[],marker="o",ls="",mfc="none",mec=RED,ms=3.4,mew=0.6),
       (Line2D([],[],color=BLUE,lw=1),mp.Patch(color=BLUE,alpha=0.16)),mp.Patch(fc=WATER,ec="#60a5fa",lw=0.6),Line2D([],[],color=MUTED,lw=0.5,ls=(0,(2,1.5)))]
    from matplotlib.legend_handler import HandlerTuple
    fig.legend(h,["reading","single reading","not used in fit","log-law fit, 95% band","water column","1 ft above bed"],handler_map={tuple:HandlerTuple(ndivide=None,pad=0)},
        loc="upper center",bbox_to_anchor=(0.53,1-8.3*0.005/H),ncol=6,fontsize=6,frameon=False,handlelength=1.4,columnspacing=1.1,handletextpad=0.4)
    T0,T1=pg["T"].iloc[0],pg["T"].iloc[-1]
    fig.text(0.53,1-8.3*0.045/H,f"Sections {T0}–{T1} (page {pi+1} of {len(pages)})",ha="center",fontsize=6.5,color=MUTED)
    fig.savefig(FIGURES/f"figS{pi+1}_log_law_profiles.png",dpi=300,facecolor="white"); fig.savefig(FIGURES/f"figS{pi+1}_log_law_profiles.pdf",facecolor="white"); plt.close(fig)

