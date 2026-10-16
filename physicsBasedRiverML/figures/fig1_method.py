"""Figure 1. The five-step method (top) and what each step produces for the Huron River reach (bottom).

Inputs:  outputs/intermediate/readings.csv, locations.csv, log_law.json, stage_law.json, grid.csv, bank_lines.geojson,
         section_velocity_2h.csv, suitability_curve.csv
Output:  outputs/figures/fig1_method.png (and .pdf)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.legend, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from riverml.config import WORK, FIGURES, FT, ADCP_E, ADCP_N, SURVEYED as SURV
H0 = json.load(open(WORK / "stage_law.json"))["selected"]["h0"]
INK,INK2,MUTED,AXIS,GRID="#000000","#000000","#000000","#b9b8ae","#e7e6e0"   # all text black
COL=['#2563eb','#0d9488','#7c3aed','#ea580c','#16a34a']
plt.rcParams.update({"text.color":"#000000","axes.labelcolor":"#000000","xtick.color":"#000000","ytick.color":"#000000","font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],"font.size":6.5,"axes.titlesize":6.8,"axes.labelsize":6.3,
 "xtick.labelsize":5.8,"ytick.labelsize":5.8,"legend.fontsize":5.6,"legend.frameon":False,"axes.edgecolor":AXIS,"axes.linewidth":0.5,"grid.color":GRID,"grid.linewidth":0.35,
 "xtick.major.size":1.8,"ytick.major.size":1.8,"xtick.major.pad":1.5,"ytick.major.pad":1.5,"axes.labelpad":1.5,"axes.titlepad":3,"axes.titlelocation":"left","mathtext.fontset":"custom","mathtext.rm":"Liberation Sans","mathtext.it":"Liberation Sans:italic","mathtext.bf":"Liberation Sans:bold:italic"})
# ---------- data
tr=pd.read_csv(WORK/"readings.csv"); loc=pd.read_csv(WORK/"locations.csv"); Z0=json.load(open(WORK/"log_law.json"))["z0_m"]
locp=loc[loc.a>0].copy(); locp["A"]=locp.a/np.clip(locp.stage_m-H0,0.02,None)
grid=pd.read_csv(WORK/"grid.csv"); gj=json.load(open(WORK/"bank_lines.geojson"))
banks=[np.array(c) for f in gj["features"] for c in ([f["geometry"]["coordinates"]] if f["geometry"]["type"]=="LineString" else f["geometry"]["coordinates"])]
ss=pd.read_csv(WORK/"section_velocity_2h.csv",parse_dates=["time_utc"]); ss["time_utc"]=pd.to_datetime(ss.time_utc,utc=True)
order=list(locp.groupby("transect").y_m.median().sort_values(ascending=False).index)
cur=pd.read_csv(WORK/"suitability_curve.csv")
# ---------- figure
W,H=7.2,4.45; fig=plt.figure(figsize=(W,H))
x0,gap=0.025,0.032; cw=(1-2*x0-4*gap)/5          # column left, width, gap (figure fraction)
cols=[x0+i*(cw+gap) for i in range(5)]
py0,ph=0.165,0.36                     # plot row
PANELS={}
TITLES=["Log-law fit per location","De-staging law",r"Rating coefficient $\mathbf{A}$","Section velocity","Decision variable vs stage"]
def pax(i,pad_l=0.035,pad_r=0.0):
    ax=fig.add_axes([cols[i]+pad_l,py0,cw-pad_l-pad_r,ph]); PANELS[i]=ax; return ax
def letter(ax,i,dx=-0.035):   # clear the axes title (titles are drawn above each column) and keep legends out of the layout
    ax.set_title("",loc="left")
    for l in ax.get_children():
        if isinstance(l,matplotlib.legend.Legend): l.set_in_layout(False)

# a: profiles
ax=pax(0)
picks=[("2025-07-21|AT10|AT1014","#93c5fd","slow"),("2025-07-14|AT07|AT0710",COL[0],"typical"),("2026-06-25|0625-T03|0625-30","#1e3a8a","fast")]
for lid,c,lab in picks:
    p=tr[tr.location_id==lid].sort_values("height_m"); a=float(loc.set_index("location_id").a[lid]); z=np.linspace(Z0*1.05,p.height_m.max()*1.1,60)
    ax.plot(a*np.log(z/Z0),z,color=c,lw=1); ax.scatter(p.velocity_ms,p.height_m,s=7,color=c,edgecolor="white",lw=0.3,zorder=3,label=f"$a$ = {a:.3f} m/s")
ax.set_xlim(0,0.65); ax.set_ylim(0,0.65); ax.set_xlabel("velocity (m/s)"); ax.set_ylabel("height above bed (m)")
ax.legend(loc="upper left",handletextpad=0.1,borderaxespad=0.2,labelspacing=0.25); ax.grid(True); ax.set_axisbelow(True)
ax.set_title("Log-law fit per location"); letter(ax,0)
# b: before/after the stage law
ax=pax(1)
dm=locp.groupby("date").agg(st=("stage_m","median"),la=("a",lambda v:np.log(v).median()),lA=("A",lambda v:np.log(v).median()))
xx=np.linspace(0.72,1.12,30); ax.plot(xx,np.log(xx-H0)+dm.la.mean()-np.mean(np.log(dm.st-H0)),color=INK2,lw=0.6,ls="--")
ax.scatter(dm.st,dm.la,s=10,color=COL[1],zorder=3,label=r"$\ln\,a$")
ax.scatter(dm.st,dm.lA,s=10,facecolor="white",edgecolor=COL[1],lw=0.8,zorder=3,label=r"$\ln\,A$ (de-staged)")
ax.set_ylim(-3.45,-0.8); ax.set_xlim(0.71,1.13); ax.set_xlabel("stage on survey day (m)"); ax.set_ylabel(r"survey-day median ($\ln$ m/s)")
ax.legend(loc="upper left",handletextpad=0.1,borderaxespad=0.2,labelspacing=0.25); ax.grid(True); ax.set_axisbelow(True)
ax.text(0.73,-3.38,"dashed: stage law",fontsize=5.4,color=INK2,va="bottom")
ax.set_title("De-staging law"); letter(ax,1)
# c: map of A
ax=pax(2,pad_l=0.028,pad_r=0.035)
vmax=np.nanquantile(np.exp(grid.logA),.99)
sc=ax.scatter(grid.x_m,grid.y_m,c=np.exp(grid.logA),cmap="Purples",s=0.35,lw=0,vmin=0,vmax=vmax,rasterized=True)
for b in banks: ax.plot((b[:,0]-ADCP_E)*FT,(b[:,1]-ADCP_N)*FT,color="#8a877f",lw=0.4)
for t,g in locp.groupby("transect"): g=g.sort_values("across_m"); ax.plot(g.x_m,g.y_m,color=INK,lw=0.35,alpha=0.55)
ax.set_aspect("equal",adjustable="datalim"); ax.set_xlim(-110,170); ax.set_xticks([0,100]); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)",labelpad=0.5)
cax=ax.inset_axes([1.06,0.05,0.07,0.5]); cb=fig.colorbar(sc,cax=cax); cb.ax.tick_params(labelsize=5.2,length=1.5,pad=1); cb.set_label("A",fontsize=5.8,labelpad=1,fontstyle="italic"); cb.outline.set_linewidth(0.3)
ax.set_title("Rating coefficient A"); letter(ax,2)
# d: section velocity through 2026
ax=pax(3,pad_l=0.028,pad_r=0.03)
s=ss[(ss.time_utc>="2026-06-01")&(ss.time_utc<"2026-08-01")]; piv=s.pivot(index="transect",columns="time_utc",values="mean_v_med").reindex(order); tt=piv.columns
xe=mdates.date2num(tt.tz_convert(None)); xe=np.r_[xe,xe[-1]+(xe[-1]-xe[-2])]
im=ax.pcolormesh(xe,np.arange(len(order)+1),piv.values,cmap="Oranges",vmin=0,vmax=0.6,shading="flat",rasterized=True)
ax.set_yticks([0.5,len(order)-0.5]); ax.set_yticklabels(["T1\nnorth","T20\nsouth"],linespacing=0.95); ax.invert_yaxis(); ax.set_ylabel("River Section"); ax.yaxis.set_label_coords(-0.035,0.5); ax.tick_params(axis="y",pad=0.8)
ax.xaxis_date(); import datetime as _dt; D_=lambda m,d: mdates.date2num(_dt.datetime(2026,m,d)); ax.set_xlim(D_(6,1),D_(8,1)); ax.set_xticks([D_(6,1),D_(7,1),D_(8,1)]); ax.set_xticklabels([]); ax.set_xticks([D_(6,16),D_(7,16)],minor=True); ax.set_xticklabels(["June","July"],minor=True); ax.tick_params(axis="x",which="minor",length=0); ax.set_xlabel("2026")
st=s[s.transect==order[0]]; ax2=ax.twinx(); ax2.plot(mdates.date2num(st.time_utc.dt.tz_convert(None)),st.stage_m,color=INK,lw=0.45)
ax2.set_ylim(0.45,1.25); ax2.set_yticks([]); ax2.spines[:].set_visible(False)
ax.text(0.97,0.95,"stage",transform=ax.transAxes,fontsize=5.6,ha="right",va="top",color=INK)
cax=ax.inset_axes([1.055,0.05,0.055,0.5]); cb=fig.colorbar(im,cax=cax,ticks=[0,0.3,0.6]); cb.ax.tick_params(labelsize=5.2,length=1.2,pad=0.6); cb.set_label(r"$\bar{v}$ (m/s)",fontsize=5.8,labelpad=0.5); cb.outline.set_linewidth(0.3)
ax.set_title("Section velocity"); letter(ax,3)
# e: ecological rating curve
ax=pax(4,pad_l=0.035)
ax.axvspan(*SURV,color=GRID,lw=0); ax.text(0.915,0.03,"surveyed",fontsize=5.4,color=MUTED,ha="center")
ax.fill_between(cur.stage_m,cur.E_lo,cur.E_hi,color=COL[4],alpha=0.2,lw=0); ax.plot(cur.stage_m,cur.E_med,color=COL[4],lw=1.2,label=r"suitability $E$")
ax.set_xlim(0.65,1.35); ax.set_ylim(0,1); ax.set_xlabel("stage (m)"); ax.set_ylabel(r"mean nesting suitability $E$")
ax.legend(loc="upper right",handlelength=1.6,borderaxespad=0.2,labelspacing=0.25); ax.grid(True); ax.set_axisbelow(True)
ax.set_title("Decision variable vs stage"); letter(ax,4)
# ---------- align every panel's full extent (labels, ticks, colorbar) to its step box column
for it in range(6):
    fig.canvas.draw(); r=fig.canvas.get_renderer(); inv=fig.transFigure.inverted()
    for i,ax in PANELS.items():
        bb=ax.get_tightbbox(r); (L,Bt),(Rr,T)=inv.transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
        p=ax.get_position(); dl=p.x0-L; dr=Rr-p.x1
        ax.set_position([cols[i]+dl,py0,cw-dl-dr,ph])
for i in PANELS:
    fig.text(cols[i],py0+ph+0.022,"abcde"[i],fontweight="bold",fontsize=8.5,va="bottom",ha="left",color=INK)
    fig.text(cols[i]+0.018,py0+ph+0.022,TITLES[i],fontweight="bold",fontsize=7,va="bottom",ha="left",color=COL[i])
# ---------- flow boxes
ov=fig.add_axes([0,0,1,1]); ov.set_xlim(0,1); ov.set_ylim(0,1); ov.axis("off"); ov.set_zorder(-1)
by0,bh=0.625,0.33; hh=0.075
steps=[("Map each profile\nto one number","",r"$v(z) = a\,\ln(z/z_0)$",r"one $z_0$ for the reach","amplitude $\mathbf{a}$\nat each location"),
       ("Remove\nthe stage","",r"$a = A\,(s-h_0)^{\gamma}$","near-equivalent $(h_0,\gamma)$\nlaws retained","rating coefficient $\mathbf{A}$,\nstage-invariant"),
       ("Learn\nthe place","","Gaussian process\nregression on geometry","one length scale\nper feature","$\mathbf{A}$ everywhere,\nwith uncertainty"),
       ("Extend\nthrough time","","Monte Carlo draws\ndriven by stage","all uncertainty\nsources sampled","section velocity\nwith 90% bands"),
       ("Transform\ndownstream","","decision variable\ncomputed from velocity","example: nesting\nhabitat suitability","decision variable\nagainst stage")]
for i,(t,inp,eq,note,out) in enumerate(steps):
    x=cols[i]; c=COL[i]
    ov.add_patch(FancyBboxPatch((x,by0),cw,bh,boxstyle="round,pad=0,rounding_size=0.012",fc="white",ec=c,lw=1.1,transform=ov.transAxes,mutation_aspect=W/H))
    ov.add_patch(FancyBboxPatch((x,by0+bh-hh),cw,hh,boxstyle="round,pad=0,rounding_size=0.012",fc=c,ec=c,lw=1.1,mutation_aspect=W/H))
    ov.add_patch(plt.Rectangle((x,by0+bh-hh),cw,hh*0.4,fc=c,ec=c,lw=0))
    ov.text(x+0.012,by0+bh-hh/2,str(i+1),fontsize=10,fontweight="bold",color="white",va="center")
    ov.text(x+0.034,by0+bh-hh/2,t,fontsize=7,fontweight="bold",color="white",va="center",linespacing=1.0)
    yy=by0+bh-hh
    ov.text(x+cw/2,yy-0.062,eq,fontsize=7.6,color=INK,ha="center",va="center",linespacing=1.1)
    ov.plot([x+0.012,x+cw-0.012],[yy-0.112,yy-0.112],color=GRID,lw=0.6)
    ov.text(x+cw/2,yy-0.153,note,fontsize=5.8,color=MUTED,ha="center",va="center",linespacing=1.05)
    ov.plot([x+0.012,x+cw-0.012],[by0+0.058,by0+0.058],color=GRID,lw=0.6)
    ov.text(x+cw/2,by0+0.03,out,fontsize=6.2,fontweight="bold",color=c,ha="center",va="center",linespacing=1.05)
    if i<4: ov.add_patch(FancyArrowPatch((x+cw+0.003,by0+bh/2),(x+cw+gap-0.002,by0+bh/2),arrowstyle="simple,head_length=0.62,head_width=0.95,tail_width=0.42",mutation_scale=15,fc=INK,ec=INK,lw=1.2,joinstyle="round",capstyle="round",shrinkA=0,shrinkB=0))
from matplotlib.transforms import Bbox
CROP=Bbox.from_extents(0,0.27,W,H)   # trim the empty strip below the panels
fig.savefig(FIGURES/"fig1_method.png",dpi=600,facecolor="white",bbox_inches=CROP); fig.savefig(FIGURES/"fig1_method.pdf",facecolor="white",bbox_inches=CROP)
