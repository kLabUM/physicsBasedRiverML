"""Figure 4. Step 3: the Gaussian process with whole sections held out, at the point and section level, and the map of
the rating coefficient A with its posterior uncertainty.

Inputs:  outputs/intermediate/cv_scores.csv, cv_points.csv, cv_sections.csv, features.csv, locations.csv, grid.csv,
         bank_lines.geojson, stage_law.json
Output:  outputs/figures/fig4_step3.png (and .pdf)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mp
from riverml.config import WORK, FIGURES
from riverml.config import FT, ADCP_E, ADCP_N
H0=json.load(open(WORK/"stage_law.json"))["selected"]["h0"]
INK,INK2,MUTED,AXIS,GRID="#1f2937","#4b5563","#6b7280","#b9b8ae","#e7e6e0"; C3="#7c3aed"; C3L="#c4b5fd"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],"font.size":7.5,"axes.labelsize":7.5,
 "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,"legend.frameon":False,"axes.edgecolor":AXIS,"axes.linewidth":0.5,"grid.color":GRID,"grid.linewidth":0.35,
 "xtick.major.size":1.8,"ytick.major.size":1.8,"xtick.minor.size":1,"ytick.minor.size":1,"xtick.major.pad":1.5,"ytick.major.pad":1.5,"axes.labelpad":2})
loc=pd.read_csv(WORK/"locations.csv"); loc=loc[loc.a>0].copy(); loc["A"]=loc.a/np.clip(loc.stage_m-H0,0.02,None)
fe=pd.read_csv(WORK/"features.csv")
grid=pd.read_csv(WORK/"grid.csv"); gj=json.load(open(WORK/"bank_lines.geojson"))
banks=[np.array(c) for f in gj["features"] for c in ([f["geometry"]["coordinates"]] if f["geometry"]["type"]=="LineString" else f["geometry"]["coordinates"])]
M=pd.read_csv(WORK/"cv_scores.csv"); P=pd.read_csv(WORK/"cv_points.csv")
W,H=7.2,2.55; fig=plt.figure(figsize=(W,H))
L,R,gap=0.065,0.985,0.08; WT=[1,0.95,1,1.4]; unit=(R-L-3*gap)/sum(WT); Bt,Hh=0.175,0.6
xs0=[L+sum(WT[:i])*unit+i*gap for i in range(4)]
axs=[fig.add_axes([xs0[i],Bt,WT[i]*unit,Hh]) for i in range(4)]
titles=["Points","Point score","Section means","Reach: A and its uncertainty"]
S=pd.read_csv(WORK/"cv_sections.csv")
# a: point level, held-out predictions vs measured, coloured by bank distance
ax=axs[0]; pf=P[(P.scheme=="section_out")&(P.model=="six features")].merge(fe[["location_id","dist_bank_m"]],on="location_id")
lim=(-6.6,-0.2); ax.plot(lim,lim,color=AXIS,lw=0.7,zorder=1)
sc=ax.scatter(pf.mu,pf.y,c=pf.dist_bank_m,cmap="Purples",vmin=-3,vmax=25,s=6,edgecolor=INK,lw=0.15,zorder=3)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xticks([-6,-4,-2]); ax.set_yticks([-6,-4,-2]); ax.set_xlabel("predicted ln A"); ax.set_ylabel("measured ln A")
cov=float(M[(M.scheme=="section_out")&(M.model=="six features")].coverage95.iloc[0])
ax.text(0.04,0.96,f"95% coverage {cov:.2f}",transform=ax.transAxes,fontsize=6.5,va="top",ha="left")
cax=ax.inset_axes([0.6,0.2,0.36,0.035]); cb=fig.colorbar(sc,cax=cax,orientation="horizontal",ticks=[0,10,20]); cb.ax.tick_params(labelsize=6,length=1.2,pad=0.8); cb.outline.set_linewidth(0.3); cax.set_xlabel("bank distance (m)",fontsize=6.2,labelpad=1)
# b: point-level scores
ax=axs[1]; mods=["reach mean","six features"]; lab=["reach\nmean","six\nfeatures"]; xs=np.arange(2)
so=M[M.scheme=="section_out"].set_index("model").CRPS; rf=M[M.scheme=="random_10fold"].set_index("model").CRPS
ax.bar(xs-0.19,[so[m] for m in mods],0.36,color=C3,label="section held out"); ax.bar(xs+0.19,[rf[m] for m in mods],0.36,color=C3L,label="random folds")
for i,m in enumerate(mods): ax.text(xs[i]-0.19,so[m]+0.008,f"{so[m]:.2f}",ha="center",fontsize=6,color=INK); ax.text(xs[i]+0.19,rf[m]+0.008,f"{rf[m]:.2f}",ha="center",fontsize=6,color=INK2)
ax.set_xticks(xs); ax.set_xticklabels(lab,linespacing=0.95); ax.set_xlim(-0.6,1.6); ax.set_ylabel("CRPS of ln A (lower is better)"); ax.set_ylim(0,0.68); ax.set_yticks([0,0.2,0.4,0.6]); ax.tick_params(axis="x",length=0)
ax.legend(loc="upper right",handlelength=1,handletextpad=0.3,borderaxespad=0.2)
# c: section level, held-out section means with 90% intervals from the joint GP covariance
ax=axs[2]; lim=(-2.8,-0.1); ax.plot(lim,lim,color=AXIS,lw=0.7,zorder=1)
for mname,col,off,mk,fc in [("bank distance",MUTED,-0.018,"o","white"),("six features",C3,0.018,"o",C3)]:
    g=S[S.model==mname]; k=int((abs(g.obs-g.pred)<=1.645*g.sd).sum())
    ax.errorbar(g.pred+off,g.obs,xerr=1.645*g.sd,fmt=mk,ms=2.8,mfc=fc,mec=col,mew=0.7,ecolor=col,elinewidth=0.6,capsize=0,zorder=3,label=f"{mname}: {k}/{len(g)}")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xticks([-2.5,-1.5,-0.5]); ax.set_yticks([-2.5,-1.5,-0.5]); ax.set_xlabel("predicted section mean ln A"); ax.set_ylabel("measured section mean ln A")
ax.legend(loc="upper left",handletextpad=0.2,borderaxespad=0.2,labelspacing=0.3,handlelength=1.2,title="in 90% band",title_fontsize=6.5,alignment="left",frameon=True,facecolor="white",edgecolor="none",framealpha=1)
# d: reach level, posterior mean of A and posterior sd of ln A on the 2 m grid
axs[3].axis("off"); p=axs[3].get_position(); half=p.width/2
for k,(col,cmap,vmin,vmax,ticks,lab,sub) in enumerate([(np.exp(grid.logA),"Purples",0,np.nanquantile(np.exp(grid.logA),.99),[0,0.1,0.2,0.3],"A (m/s per m)","mean"),
                                                      (grid.logA_sd,"Greys",0.15,0.45,[0.2,0.3,0.4],"sd of ln A","sd")]):
    ax=fig.add_axes([p.x0+k*half,p.y0,half,p.height])
    sc=ax.scatter(grid.x_m,grid.y_m,c=col,cmap=cmap,s=0.3,lw=0,vmin=vmin,vmax=vmax,rasterized=True)
    for b in banks: ax.plot((b[:,0]-ADCP_E)*FT,(b[:,1]-ADCP_N)*FT,color="#8a877f",lw=0.35)
    for tt,g in loc.groupby("transect"): g=g.sort_values("across_m"); ax.plot(g.x_m,g.y_m,color=INK,lw=0.35,alpha=0.75)
    ax.set_xlim(-100,230); ax.set_ylim(-470,245); ax.set_aspect("equal"); ax.set_anchor("C"); ax.set_xticks([0,100])
    ax.set_xlabel("east (m)",labelpad=1)
    if k==0: ax.set_ylabel("north (m)",labelpad=1)
    else: ax.set_yticklabels([])
    ax.tick_params(labelsize=6.3)
    ax.text(0.05,0.03,sub,transform=ax.transAxes,ha="left",va="bottom",fontsize=6.8,color=INK2,fontweight="bold")
    cax=ax.inset_axes([0.62,0.5,0.07,0.44]); cb=fig.colorbar(sc,cax=cax,ticks=ticks); cb.ax.tick_params(labelsize=5.8,length=1.2,pad=0.8); cb.outline.set_linewidth(0.3)
    cb.set_label(lab,fontsize=6,labelpad=1)
for i,ax in enumerate(axs):
    if i<3: ax.grid(True,which="major"); ax.set_axisbelow(True)
    fig.canvas.draw(); p=ax.get_position(); cx=(p.x0+p.x1)/2; y_=Bt+Hh+0.035
    t=fig.text(cx,y_,titles[i],fontweight="bold",fontsize=8,va="bottom",ha="center",color=C3)
    bb=t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x0-0.008,y_,"abcd"[i],fontweight="bold",fontsize=9.5,va="bottom",ha="right",color=INK)
axs[1].grid(False,axis="x")
x0=axs[0].get_position().x0-0.05; x1=0.995
fig.patches.append(mp.FancyBboxPatch((x0,0.925),x1-x0,0.062,boxstyle="round,pad=0,rounding_size=0.006",fc=C3,ec=C3,transform=fig.transFigure,mutation_aspect=W/H))
fig.text(x0+0.012,0.956,"Step 3  ·  Learn the place",color="white",fontweight="bold",fontsize=8,va="center")
fig.savefig(FIGURES/"fig4_step3.png",dpi=600,facecolor="white"); fig.savefig(FIGURES/"fig4_step3.pdf",facecolor="white")
