"""Build shelf.glb + meta.json + placements.json of the snack shelf from catalog.py and tex/ (Hagelslag's build.py,
adapted: two bays of their own widths, cans, tubs, bags). Units in the GLB: metres."""
import json, math, io, numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
import os
from catalog import P, LEFT, RIGHT, LEFT_Y, RIGHT_Y, BAYS

CM = 0.01
DEPTH = lambda y: 60 - (y-12)*(60-44)/(126-12)     # shelf front z (cm): deeper towards the bottom (estimated)
BAYW = {side: w for side, x0, w in BAYS}

# ---- geometry accumulators --------------------------------------------------
class Acc:
    def __init__(s): s.v=[];s.f=[];s.uv=[];s.n=0
    def add(s,v,f,uv):
        v=np.asarray(v,float);f=np.asarray(f,int);uv=np.asarray(uv,float)
        s.v.append(v);s.f.append(f+s.n);s.uv.append(uv);s.n+=len(v)
    def mesh(s):
        return np.vstack(s.v),np.vstack(s.f),np.vstack(s.uv)

def box_geom(x0,x1,y0,y1,z0,z1,fu=(0,1),fv=(0,1),side=(0.5,0.5)):
    """Box with +Z face textured over fu,fv rectangle, others sampling `side` uv."""
    c=[]
    F=[(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    B=[(x1,y0,z0),(x0,y0,z0),(x0,y1,z0),(x1,y1,z0)]
    R=[(x1,y0,z1),(x1,y0,z0),(x1,y1,z0),(x1,y1,z1)]
    L=[(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)]
    T=[(x0,y1,z1),(x1,y1,z1),(x1,y1,z0),(x0,y1,z0)]
    D=[(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)]
    verts=[];uvs=[];faces=[]
    for i,q in enumerate([F,B,R,L,T,D]):
        k=len(verts); verts+=q
        faces+=[(k,k+1,k+2),(k,k+2,k+3)]
        if i==0: uvs+=[(fu[0],fv[0]),(fu[1],fv[0]),(fu[1],fv[1]),(fu[0],fv[1])]
        else: uvs+=[side]*4
    return verts,faces,uvs

def tilt(v,angle,py,pz):
    v=np.array(v,float);c,s=math.cos(angle),math.sin(angle)
    y=v[:,1]-py;z=v[:,2]-pz
    v[:,1]=py+y*c-z*s; v[:,2]=pz+y*s+z*c
    return v

PROFILES={
 'jar':[(0,.86),(.02,.97),(.1,1),(.6,1),(.68,.93),(.76,.86),(.79,.9),(.99,.9),(1,.7),(1,0)],
 'bigjar':[(0,.8),(.03,.93),(.15,1),(.55,1),(.7,.93),(.78,.76),(.83,.76),(.85,.82),(.99,.82),(1,.6),(1,0)],
 'cup':[(0,.92),(.03,1),(.7,1),(.76,.95),(.8,.98),(.99,.98),(1,.7),(1,0)],
 'tall':[(0,.85),(.02,.95),(.3,.92),(.68,1),(.78,.93),(.85,.84),(.87,.9),(.99,.9),(1,.7),(1,0)],
 'tub':[(0,.8),(.03,.83),(.84,.97),(.87,1),(.99,1),(1,.88),(1,0)],
 'tin':[(0,.96),(.02,.98),(.75,.98),(.77,1),(.99,1),(1,.9),(1,0)],
 'can':[(0,.97),(.01,1),(.97,1),(.98,1.02),(1,1.02),(1,0)],
}
SEG=20
def lathe(kind,cx,y0,cz,r,h):
    prof=PROFILES[kind]; rmax=max(p[1] for p in prof)
    verts=[];uvs=[];faces=[]
    for fy,fr in prof:
        for k in range(SEG):
            a=2*math.pi*k/SEG; ca,sa=math.cos(a),math.sin(a)
            verts.append((cx+ca*fr*r/rmax,y0+fy*h,cz+sa*fr*r/rmax))
            uvs.append((0.5+0.5*ca*fr/rmax,fy))
    n=len(prof)
    for i in range(n-1):
        for k in range(SEG):
            a=i*SEG+k;b=i*SEG+(k+1)%SEG;c=(i+1)*SEG+(k+1)%SEG;d=(i+1)*SEG+k
            faces+=[(a,d,c),(a,c,b)]
    return verts,faces,uvs

def carton(x,w,y0,zf,d,h):
    """Gable-top carton; front+roof planar-mapped from the photo (ridge at v=1)."""
    x0,x1=x,x+w;z1=zf;z0=zf-d;zm=zf-d/2
    ys=y0+h*.74;yr=y0+h*.94;yt=y0+h
    side=(.995,.5)
    verts=[];uvs=[];faces=[]
    def q(pts,uv):
        k=len(verts);verts.extend(pts);uvs.extend(uv);faces.extend([(k,k+1,k+2),(k,k+2,k+3)])
    q([(x0,y0,z1),(x1,y0,z1),(x1,ys,z1),(x0,ys,z1)],[(0,0),(1,0),(1,.74),(0,.74)])
    q([(x0,ys,z1),(x1,ys,z1),(x1,yr,zm),(x0,yr,zm)],[(0,.74),(1,.74),(1,.94),(0,.94)])
    q([(x0,yr,zm),(x1,yr,zm),(x1,yt,zm),(x0,yt,zm)],[(0,.94),(1,.94),(1,1),(0,1)])
    q([(x1,y0,z0),(x0,y0,z0),(x0,ys,z0),(x1,ys,z0)],[side]*4)
    q([(x1,ys,z0),(x0,ys,z0),(x0,yr,zm),(x1,yr,zm)],[side]*4)
    q([(x1,y0,z1),(x1,y0,z0),(x1,ys,z0),(x1,ys,z1)],[side]*4)
    q([(x0,y0,z0),(x0,y0,z1),(x0,ys,z1),(x0,ys,z0)],[side]*4)
    k=len(verts);verts+= [(x1,ys,z1),(x1,ys,z0),(x1,yr,zm),(x0,ys,z0),(x0,ys,z1),(x0,yr,zm)];uvs+=[side]*6;faces+=[(k,k+1,k+2),(k+3,k+4,k+5)]
    return verts,faces,uvs

SIDE={'jb_xxl_disco':(246,190,28),'jb_xxl_mix':(40,140,215),'jb_xxl_puur':(120,45,160),'jb_vrucht':(245,200,40),'jb_ha_puur':(205,45,40),'jb_duo_ha':(170,90,190),'jb_vl_puur':(205,60,35),'jb_vl_duo':(120,60,160),'vz_tijger':(120,175,60)}
# ---- textures ---------------------------------------------------------------
def load_tex(pid,pad):
    im=Image.open(f'tex/{pid}.jpg').convert('RGB')
    a=np.asarray(im)
    edge=np.concatenate([a[:,:6].reshape(-1,3),a[:,-6:].reshape(-1,3)])
    col=tuple(int(c) for c in np.median(edge,axis=0))
    col=SIDE.get(pid,col)
    if pad:
        out=Image.new('RGB',(im.width+8,im.height),col);out.paste(im,(0,0));im=out
    b=io.BytesIO();im.save(b,'JPEG',quality=85,subsampling=0);b.seek(0)
    return Image.open(b),col,im.width

# ---- place products ---------------------------------------------------------
products={pid:Acc() for pid in P}
tags=[]; issues=[]; placed=[]; dividers=[]
counts={pid:{'facings':0,'units':0,'price':None,'yellow':False,'where':[]} for pid in P}

def bag_kind(k): return k in('bag',)
def place_bay(bayx,levels,ys,side):
    for li,(groups,y) in enumerate(zip(levels,ys)):
        zf=DEPTH(y)
        if li==len(ys)-1: zf=62
        ceiling=(ys[li-1]-2.0) if li>0 else 999
        x=bayx+0.8; total=0
        for gi,gr in enumerate(groups):
            x+=gr['gap']
            pr=P[gr['pid']]; w,h,d,kind=pr['w'],pr['h'],pr['d'],pr['kind']
            front=zf-1.5-gr['back']
            rows=gr['rows'] or max(1,int((front-4)/d))
            rows=min(rows,max(1,int((front-2)/d)))
            sx=x
            for n in range(gr['n']):
                for r in range(rows):
                    zr=front-r*d
                    for st in range(gr['stack']):
                        yy=y+st*h
                        if kind=='box' or kind=='block':
                            ev=box_geom(x+.1,x+w-.1,yy,yy+h-.02,zr-d+.1,zr,fu=(0,U[gr['pid']]),side=(.995,.5))
                        elif kind=='bag':
                            v,f,u=box_geom(x+.2,x+w-.2,yy,yy+h,zr-d+.5,zr-.5,fu=(0,U[gr['pid']]),side=(.995,.5))
                            ev=(tilt(v,-.14,yy,zr),f,u)
                        elif kind=='carton':
                            ev=carton(x+.1,w-.2,yy,zr,d-.1,h)
                        else:
                            ev=lathe(kind,x+w/2,yy,zr-d/2,w/2-.1,h)
                        products[gr['pid']].add(*ev)
                        counts[gr['pid']]['units']+=1
                        placed.append((side,li,gr['pid'],x,x+w,yy,yy+h,zr-d,zr))
                counts[gr['pid']]['facings']+=1
                x+=w
            c=counts[gr['pid']]
            c['where'].append(f"{'links' if side=='L' else 'rechts'}, plank {li+1}")
            if gr['price'] and not c['price']: c['price']=gr['price']; c['yellow']=gr['yellow']
            if gr['tag']:
                tags.append(dict(x=max(sx,bayx+1),y=y,z=zf,name=pr['name'],price=gr['price'],yellow=gr['yellow']))
            if gi>0: dividers.append((sx-gr['gap']/2 if gr['gap']>1 else sx-.05,y,zf-1.5,min(zf-3,22)))
        top=max(p[6] for p in placed if p[0]==side and p[1]==li)
        total=x-bayx
        if total>BAYW[side]-0.2: issues.append(f"{side}{li+1}: row width {total:.1f} cm exceeds bay")
        if top>ceiling: issues.append(f"{side}{li+1}: top {top:.1f} > clearance {ceiling:.1f}")
        print(f"{side}{li+1} y={y:5.1f} zf={zf:4.1f} width={total:5.1f} top={top:5.1f} clear={ceiling:5.1f}")

# uv extent for padded textures
TEX={};U={}
for pid,pr in P.items():
    pad=pr['kind'] in('box','block','bag','carton')
    img,col,tw=load_tex(pid,pad)
    TEX[pid]=img
    U[pid]=(tw-8)/tw if pad else 1.0

place_bay(0,LEFT,LEFT_Y,'L')
place_bay(BAYW['L'],RIGHT,RIGHT_Y,'R')

# price tags: a tag starts at its lane's left edge, but never overlaps the tag before it (a narrow lane's tag is wider
# than the lane): pushed right with a small gap, then pulled back left if the bay's end is passed
def tag_w(t): return 9.6 if t['yellow'] else 8.6
TAG_GAP=0.2
for _side,bayx,bw in BAYS:
    for key in sorted({(t['y'],t['z']) for t in tags if bayx<=t['x']<bayx+bw}):
        ts=sorted((t for t in tags if bayx<=t['x']<bayx+bw and (t['y'],t['z'])==key),key=lambda t:t['x'])
        for a,b in zip(ts,ts[1:]):
            b['x']=max(b['x'],a['x']+tag_w(a)+TAG_GAP)
        end=bayx+bw-.5
        for t in reversed(ts):
            t['x']=min(t['x'],end-tag_w(t)); end=t['x']-TAG_GAP

# carton uv extent: squeeze u range to exclude pad column
for pid,pr in P.items():
    if pr['kind']=='carton':
        a=products[pid];
        for uv in a.uv:
            m=uv[:,0]<.99
            uv[m,0]*=U[pid]
    if pr['kind']=='block':pass

# ---- fixtures ---------------------------------------------------------------
fx={k:Acc() for k in('dark','wire','rail','red','deck','plinth','glass','silver','plate')}
def bx(key,x0,x1,y0,y1,z0,z1):
    fx[key].add(*box_geom(x0,x1,y0,y1,z0,z1,side=(0,0)))
W=sum(w for _s,_x,w in BAYS)
# back panel & uprights
bx('dark',0,W,0,150,0,2)
for ux in(0,BAYW['L'],W):
    bx('dark',ux-1.5,ux+1.5,0,150,2,5)
# plinth & base decks
# (from the back panel's front, z=2: their back faces in the panel's plane z-fought)
bx('plinth',0,W,0,5,2,60)
bx('deck',0,W,5,12,2,62)
bx('rail',0,W,6,12.5,62,63.2)
# per-bay shelves
for bayx,bw,ys in((0,BAYW['L'],LEFT_Y),(BAYW['L'],BAYW['R'],RIGHT_Y)):
    for li,y in enumerate(ys[:-1]):
        zf=DEPTH(y)
        # brackets
        for ax in(bayx+.5,bayx+bw-.5):
            bx('dark',ax-.6,ax+.6,y-3,y-.3,2,zf-1)
        if li==0:
            bx('plate',bayx,bayx+bw,y-1.2,y,2,zf)
        else:
            nx=int(round(38*bw/100))
            for i in range(nx+1):
                wx=bayx+1+i*(bw-2)/nx
                bx('wire',wx-.13,wx+.13,y-.3,y,2,zf-.5)
            for cz in(5,(zf)/2,zf-2):
                bx('wire',bayx+.5,bayx+bw-.5,y-.7,y-.35,cz-.2,cz+.2)
        bx('dark',bayx,bayx+bw,y-3,y-.8,zf-3,zf-.5)
        # front rail
        bx('rail',bayx+.2,bayx+bw-.2,y-3.2,y+1.3,zf,zf+1.2)
# glass dividers
for dx,y,zf,l in dividers:
    bx('glass',dx-.05,dx+.05,y,y+9,zf-l,zf)
stock={}
sa={}

# ---- price tag atlas --------------------------------------------------------
ATW,ATH=256,128; cols=8; rows=math.ceil((len(tags)+1)/cols)
atlas=Image.new('RGB',(ATW*cols,ATH*rows),(226,228,224)); dr=ImageDraw.Draw(atlas)
FONTS=os.environ.get('FONTS','/usr/share/fonts/truetype/dejavu')
FB=f'{FONTS}/DejaVuSans-Bold.ttf';FR=f'{FONTS}/DejaVuSans.ttf'
fb=ImageFont.truetype(FB,50);fbs=ImageFont.truetype(FB,30);fs=ImageFont.truetype(FR,15)
tagacc=Acc()
def tag_uv(i):
    c,r=i%cols,i//cols
    return (c/cols,(c+1)/cols),(1-(r+1)/rows,1-r/rows)
for i,t in enumerate(tags):
    c,r=i%cols,i//cols; ox,oy=c*ATW,r*ATH
    bg=(250,205,20) if t['yellow'] else (232,233,228)
    dr.rectangle([ox+4,oy+4,ox+ATW-5,oy+ATH-5],fill=bg,outline=(70,70,70),width=3)
    dr.text((ox+14,oy+12),t['name'][:30].upper(),font=fs,fill=(30,30,30))
    for b in range(0,70,4):
        dr.rectangle([ox+14+b,oy+84,ox+15+b+(b%3),oy+110],fill=(40,40,40))
    if t['price']:
        e,cc=t['price'].split('.')
        tx=ox+ATW-122
        dr.text((tx,oy+42),e,font=fb,fill=(10,10,10))
        dr.text((tx+36,oy+40),cc,font=fbs,fill=(10,10,10))
    w,h=(8.6,4.3) if not t['yellow'] else (9.6,4.8)
    fu,fv=tag_uv(i)
    x=t['x'];y0=t['y']-3.4;z=t['z']+1.2
    tagacc.add(*box_geom(x,x+w,y0,y0+h,z,z+.5,fu=fu,fv=fv,side=(.995,.005)))
b=io.BytesIO();atlas.save(b,'JPEG',quality=88);b.seek(0);atlas_img=Image.open(b)

# ---- assemble ---------------------------------------------------------------
scene=trimesh.Scene()
def add(name,acc,material):
    if not acc.v: return 0
    v,f,uv=acc.mesh()
    m=trimesh.Trimesh(vertices=v*CM,faces=f,process=False)
    m.visual=TextureVisuals(uv=uv,material=material)
    scene.add_geometry(m,node_name=name,geom_name=name)
    return len(f)
tri=0
for pid,acc in products.items():
    if not acc.v: print('unused',pid); continue
    mat=PBRMaterial(name=pid,baseColorTexture=TEX[pid],metallicFactor=0.0,roughnessFactor=.55 if P[pid]['kind'] in PROFILES else .75)
    tri+=add('p_'+pid,acc,mat)
FXM={'dark':((46,48,52),.4,.6,None),'wire':((205,205,208),.5,.45,None),'rail':((200,202,218),.2,.25,.82),'red':((205,28,30),0,.45,None),
 'deck':((168,170,172),.5,.4,None),'plinth':((22,22,24),.2,.6,None),'glass':((225,235,240),0,.08,.22),'silver':((200,200,205),.7,.3,None),'plate':((70,72,76),.3,.6,None)}
for k,(col,met,rgh,alpha) in FXM.items():
    kw=dict(alphaMode='BLEND') if alpha else {}
    mat=PBRMaterial(name=k,baseColorFactor=[c/255 for c in col]+[alpha or 1.0],metallicFactor=met,roughnessFactor=rgh,doubleSided=(k in('glass','red')),**kw)
    tri+=add('fx_'+k,fx[k],mat)
for k,col in stock.items():
    mat=PBRMaterial(name=k,baseColorFactor=list(col)+[1.0],metallicFactor=0,roughnessFactor=.8)
    tri+=add('fx_'+k,sa[k],mat)
tri+=add('fx_tags',tagacc,PBRMaterial(name='tags',baseColorTexture=atlas_img,metallicFactor=0,roughnessFactor=.6))
scene.export('shelf.glb',include_normals=True)

meta=dict(products={pid:dict(name=P[pid]['name'],w=P[pid]['w'],h=P[pid]['h'],d=P[pid]['d'],src=P[pid]['src'],
    price=counts[pid]['price'],yellow=counts[pid]['yellow'],facings=counts[pid]['facings'],units=counts[pid]['units'],where=counts[pid]['where'])
    for pid in P if counts[pid]['units']>0},
    totals=dict(facings=sum(c['facings'] for c in counts.values()),units=sum(c['units'] for c in counts.values()),products=sum(1 for c in counts.values() if c['units'])),
    size=dict(w=W,h=150,d=63))
json.dump(meta,open('meta.json','w'),ensure_ascii=False)

# ---- per-package placement, for simulators that need separate rigid bodies ---
units=[dict(product=pid,name=P[pid]['name'],kind=P[pid]['kind'],bay=side,shelf=li+1,
    center_m=[round((x0+x1)/2*CM,4),round((y0+y1)/2*CM,4),round((z0+z1)/2*CM,4)],
    size_m=[round((x1-x0)*CM,4),round((y1-y0)*CM,4),round((z1-z0)*CM,4)])
    for side,li,pid,x0,x1,y0,y1,z0,z1 in placed]
json.dump(dict(frame='metres; origin on the floor at the back-left corner; +x along the shelf, +y up, +z towards the aisle',
    shelves=dict(left_heights_cm=LEFT_Y,right_heights_cm=RIGHT_Y),units=units),open('placements.json','w'),ensure_ascii=False,indent=0)
# ---- checks -----------------------------------------------------------------
over=0
for a in placed:
    side,li,pid,x0,x1,y0,y1,z0,z1=a
    if z0<2 and pid not in('',): issues.append(f'{pid} on {side}{li+1} goes through back panel (z0={z0:.1f})')
    if x0<(0 if side=="L" else BAYW['L']) or x1>(BAYW['L'] if side=="L" else W): issues.append(f'{pid} on {side}{li+1} outside bay')
import os
print('triangles',tri,'glb MB',round(os.path.getsize('shelf.glb')/1e6,2),'units',meta['totals'])
print('ISSUES:' if issues else 'no issues'); [print(' ',i) for i in sorted(set(issues))]
