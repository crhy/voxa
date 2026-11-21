import sys; sys.path.insert(0, ".")
import json, glob
import numpy as np
from pywebrtc_audio import EchoCanceller as AEC3
from voxa.refmask import ReferenceMask
SR=16000
def load(p):
    import wave
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)/32768.0
def aec3_run(mic, ref, mask=False):
    ec=AEC3(sample_rate=SR,num_channels=1,stream_delay_ms=0)
    m=ReferenceMask(rate=SR) if mask else None
    out=[]
    for i in range(0,len(mic),160):
        if i+160>len(mic): break
        o=np.asarray(ec.process(mic[i:i+160].astype(np.float32), ref[i:i+160].astype(np.float32)),dtype=np.float64)
        if m is not None:
            o=np.asarray(m.process(o, ref[i:i+160]), dtype=np.float64)
        out.append(o)
    return np.concatenate(out)
def pool(mic,out,ref):
    vals=[]
    n=min(len(mic),len(out),len(ref))
    for i in range(0,n,160):
        if (i+160)/SR<2.0: continue
        r=ref[i:i+160]
        if np.mean(r*r)<=1e-6: continue
        pm=np.mean(mic[i:i+160]**2); po=np.mean(out[i:i+160]**2)
        if pm<1e-12: continue
        vals.append(10*np.log10(pm/max(po,1e-20)))
    return vals
allv={'raw':[], 'mask':[]}
for d in sorted(glob.glob('/home/rhy/voxa-bench-data/smoke/smoke-000[0-5]-fst*')):
    t=json.load(open(d+'/truth.json')); e=t['echo']
    mic=load(d+'/mic.wav'); ref=load(d+'/ref.wav'); n=min(len(mic),len(ref)); mic,ref=mic[:n],ref[:n]
    for name,mask in (('raw',False),('mask',True)):
        out=aec3_run(mic,ref,mask); v=pool(mic,out,ref); allv[name]+=v
        if v: print(d.split('/')[-1],name,'med',round(float(np.median(v)),1),'p10',round(float(np.percentile(v,10)),1),'n',len(v))
        else: print(d.split('/')[-1],name,'no frames')
for name,v in allv.items():
    print('POOL',name,'median',round(float(np.median(v)),1),'p10',round(float(np.percentile(v,10)),1),'n',len(v))
