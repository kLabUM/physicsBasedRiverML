"""Figure 2. Study area and data: (a) the Huron River watershed with the study reach and the stage sensor (basemap: an
ArcGIS export, georeferenced to the river centerlines), (b) the surveyed sections, the ADCP and the nests used,
(c) the stage record with the survey days and the ADCP period.

Inputs:  data/basemap/, outputs/intermediate/locations.csv, nests_used.csv, stage_adcp_10min.csv, bank_lines.geojson
Output:  outputs/figures/fig2_study_area.png (and .pdf)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.patches as mp
from PIL import Image
import matplotlib.patheffects as pe
from pyproj import Transformer
from riverml.config import WORK, FIGURES, BASEMAP, SURVEYED, FT, CRS, ADCP_E, ADCP_N, SENSOR_LON, SENSOR_LAT
SURV=SURVEYED
INK,INK2,MUTED,AXIS,GRID="#1f2937","#4b5563","#6b7280","#b9b8ae","#e7e6e0"
C25,C26,CADCP,CNEST,CSENS,CREACH="#2563eb","#ea580c","#0d9488","#a21caf","#111827","#dc2626"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],"font.size":7.5,"axes.labelsize":7.5,
 "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,"legend.frameon":False,"axes.edgecolor":AXIS,"axes.linewidth":0.5,"grid.color":GRID,"grid.linewidth":0.35,
 "xtick.major.size":1.8,"ytick.major.size":1.8,"xtick.major.pad":1.5,"ytick.major.pad":1.5,"axes.labelpad":2})
t=Transformer.from_crs(CRS,4326,always_xy=True); merc=lambda la: np.log(np.tan(np.pi/4+np.radians(la)/2))
BM=json.load(open(BASEMAP/"huron_watershed_georef.json")); G=BM["p"]; topx=lambda lo,la:(G[0]*lo+G[1],G[2]*merc(la)+G[3])
loc=pd.read_csv(WORK/"locations.csv"); loc=loc[loc.a>0]
ser=pd.read_csv(WORK/"stage_adcp_10min.csv",parse_dates=["time_utc"]); ser["time_utc"]=pd.to_datetime(ser.time_utc,utc=True)
adcp=ser.dropna(subset=["adcp_ms"])
gj=json.load(open(WORK/"bank_lines.geojson"))
banks=[np.array(c) for f in gj["features"] for c in ([f["geometry"]["coordinates"]] if f["geometry"]["type"]=="LineString" else f["geometry"]["coordinates"])]
to_sp=Transformer.from_crs(4326,CRS,always_xy=True); se,sn=to_sp.transform(SENSOR_LON,SENSOR_LAT); sx,sy=(se-ADCP_E)*FT,(sn-ADCP_N)*FT
order=list(loc.groupby("transect").y_m.median().sort_values(ascending=False).index)
W,H=7.2,2.9; fig=plt.figure(figsize=(W,H))
# ---- a: watershed (ArcGIS export in data/basemap, registered to the georeference fitted on the river centerlines)
LB=BM["label_px"]; s_,ox_,oy_=BM["s"],BM["ox"],BM["oy"]
topx2=lambda lo,la:((G[0]*lo+G[1])*s_-ox_+100,(G[2]*merc(la)+G[3])*s_-oy_+100)
ax=fig.add_axes([0.005,0.2,0.37,0.68]); im=Image.open(BASEMAP/"huron_watershed_arcgis.png").convert("RGB"); x0,x1,y0,y1=BM["crop_px"]
ax.imshow(im.crop((x0,y0,x1,y1)),extent=(x0,x1,y1,y0)); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_edgecolor(AXIS)
lo,la=t.transform(loc.Easting.values,loc.Northing.values); px,py=topx2(lo,la); cxm,cym=np.median(px),np.median(py)
ax.add_patch(mp.Rectangle((cxm-13,cym-20),26,40,fill=False,ec=CREACH,lw=1.2))
ax.annotate("Study reach",(cxm,cym+20),xytext=tuple(LB["reach"]),fontsize=6.5,color=CREACH,fontweight="bold",ha="center",va="center",path_effects=[pe.withStroke(linewidth=2.2,foreground="white")],
            arrowprops=dict(arrowstyle="-",color=CREACH,lw=0.7))
spx,spy=BM["sensor_px"]; ax.plot(spx,spy,"^",ms=3.2,mfc="white",mec=CSENS,mew=0.8,zorder=6)
ax.annotate("stage sensor",(spx-3,spy-5),xytext=tuple(LB["sensor"]),fontsize=6,color=CSENS,ha="center",va="center",fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.2,foreground="white")],arrowprops=dict(arrowstyle="-",color=CSENS,lw=0.6))
kmpx=G[0]*s_/(111.32*np.cos(np.radians(42.35)))
bx,by=x1-370,y1-40; ax.plot([bx,bx+20*kmpx],[by,by],color=INK,lw=1.4); [ax.plot([bx+k*kmpx]*2,[by-6,by],color=INK,lw=0.8) for k in (0,10,20)]
ax.text(bx+10*kmpx,by-9,"20 km",ha="center",va="bottom",fontsize=6)
ax.annotate("",xy=(x1-85,y0+70),xytext=(x1-85,y0+120),arrowprops=dict(arrowstyle="-|>",color=INK,lw=0.9,mutation_scale=8)); ax.text(x1-85,y0+64,"N",ha="center",va="bottom",fontsize=7.5,fontweight="bold")
ax.text(0.005,-0.01,BM["attribution"],transform=ax.transAxes,fontsize=3.9,color=MUTED,va="top")
ax.set_xlim(x0,x1); ax.set_ylim(y1,y0)
# ---- b: reach
ax=fig.add_axes([0.46,0.22,0.14,0.66])
for b in banks: ax.plot((b[:,0]-ADCP_E)*FT,(b[:,1]-ADCP_N)*FT,color="#8a877f",lw=0.5)
for yr,c in [(2025,C25),(2026,C26)]:
    for tt,g in loc[loc.year==yr].groupby("transect"): g=g.sort_values("across_m"); ax.plot(g.x_m,g.y_m,color=c,lw=1.1,solid_capstyle="round")
nn=pd.read_csv(WORK/"nests_used.csv")
ax.scatter(nn.x_m,nn.y_m,s=4,marker="o",facecolor=CNEST,edgecolor="white",lw=0.2,zorder=4)
ax.plot(0,0,"*",ms=8,color=CADCP,mec="white",mew=0.4,zorder=5)
for tt,lab in [(order[0],"T1"),(order[-1],"T20")]:
    g=loc[loc.transect==tt]; ax.text(g.x_m.max()+10,g.y_m.median(),lab,fontsize=6.5,va="center",color=INK)
ang=np.arctan2(sx,sy); ax.annotate("",xy=(150+60*np.sin(ang),205+60*np.cos(ang)),xytext=(150,205),arrowprops=dict(arrowstyle="-|>",color=CSENS,lw=0.9,mutation_scale=8))
ax.text(170,190,"to stage\nsensor",fontsize=6.3,ha="right",va="top",color=INK)
ax.set_xlim(-200,180); ax.set_ylim(-470,275); ax.set_aspect("equal"); ax.set_xticks([-100,0,100]); ax.set_xlabel("east of ADCP (m)"); ax.set_ylabel("north of ADCP (m)",labelpad=1)
ax.grid(True); ax.set_axisbelow(True)
h=[plt.Line2D([],[],color=C25,lw=1.2,label=f"2025 sections ({loc[loc.year==2025].transect.nunique()})"),plt.Line2D([],[],color=C26,lw=1.2,label=f"2026 sections ({loc[loc.year==2026].transect.nunique()})"),
   plt.Line2D([],[],marker="*",ls="",color=CADCP,ms=6,label="ADCP"),plt.Line2D([],[],marker="o",ls="",color=CNEST,ms=3,label=f"fish nests, analyzed ({len(nn)})")]
HB=h
# ---- c: stage and coverage
days=sorted(loc.date.unique())
for k,yr in enumerate([2025,2026]):
    ax=fig.add_axes([0.675,0.57-k*0.33,0.315,0.3]); s=ser[ser.time_utc.dt.year==yr]
    ax.axhspan(*SURV,color="#d6d4cc",alpha=0.55,lw=0,zorder=0.5)
    if yr==2025: ax.axvspan(adcp.time_utc.min(),adcp.time_utc.max(),color="#cfeae7",lw=0,zorder=0)
    ax.plot(s.time_utc,s.stage_m,color=INK,lw=0.5)
    c=C25 if yr==2025 else C26
    for d in [d for d in days if d.startswith(str(yr))]:
        st=loc[loc.date==d].stage_m.median(); ax.plot(pd.Timestamp(d,tz="UTC")+pd.Timedelta(hours=16),st,"o",ms=3,mfc=c,mec="white",mew=0.4,zorder=4)
    for d in sorted(nn[nn.date.str.startswith(str(yr))].date.unique()): ax.plot(pd.Timestamp(d,tz="UTC")+pd.Timedelta(hours=16),0.6,"|",ms=4,color=CNEST,mew=0.9)
    ax.set_ylim(0.5,1.75); ax.set_yticks([0.5,1.0,1.5]); ax.set_yticklabels(["0.5","1.0","1.5"]); ax.set_ylabel("stage (m)")
    ax.set_xlim(pd.Timestamp(f"{yr}-04-01",tz="UTC"),pd.Timestamp(f"{yr}-10-08",tz="UTC"))
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.grid(True); ax.set_axisbelow(True)
    if yr==2025: ax.tick_params(labelbottom=False)
    ax.text(0.985,0.93,str(yr),transform=ax.transAxes,fontsize=7,fontweight="bold",va="top",ha="right")
    if yr==2025: ax.text(adcp.time_utc.min()+pd.Timedelta(days=3),1.6,"ADCP measurement",fontsize=6.3,color=CADCP,va="center",fontweight="bold")
from matplotlib.legend_handler import HandlerTuple
sv=(plt.Line2D([],[],marker="o",ls="",mfc=C25,mec="white",ms=4),plt.Line2D([],[],marker="o",ls="",mfc=C26,mec="white",ms=4))
hc=[plt.Line2D([],[],color=INK,lw=0.8),mp.Patch(color="#d6d4cc",alpha=0.55),sv,plt.Line2D([],[],marker="|",ls="",color=CNEST,ms=5,mew=1)]
fig.legend(HB,[x.get_label() for x in HB],loc="lower center",bbox_to_anchor=(0.53,-0.015),ncol=2,fontsize=6,handlelength=1.2,columnspacing=0.8,handletextpad=0.3)
fig.legend(hc,["stage (10 min)","surveyed stage range","transect survey days","nest survey days"],handler_map={tuple:HandlerTuple(ndivide=None,pad=0.3)},loc="lower center",bbox_to_anchor=(0.832,-0.015),ncol=2,fontsize=6,handlelength=1.4,columnspacing=0.9,handletextpad=0.3)
# titles
for x_,y_,l,tt in [(0.01,0.955,"a","Huron River Watershed"),(0.395,0.955,"b","Field Survey of Study Reach"),(0.655,0.955,"c","Stage Record and Survey Coverage")]:
    fig.text(x_,y_,l,fontweight="bold",fontsize=9.5,va="center",color=INK); fig.text(x_+0.022,y_,tt,fontweight="bold",fontsize=8,va="center",color=INK)
fig.savefig(FIGURES/"fig2_study_area.png",dpi=600,facecolor="white"); fig.savefig(FIGURES/"fig2_study_area.pdf",facecolor="white",dpi=600)

