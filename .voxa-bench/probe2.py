import json, glob
import numpy as np
from pywebrtc_audio import EchoCanceller as AEC3
SR=16000
def load(p):
    import wave
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)/32768.0
def resample_lin(x, factor):
    n=int(len(x)/factor)
    return np.interp(np.linspace(0,len(x)-1,n), np.arange(len(x)), x)
def aec3_run(mic, ref):
    ec=AEC3(sample_rate=SR,num_channels=1,stream_delay_ms=0)
    out=[]
    for i in range(0,len(mic),160):
        if i+160>len(mic): break
        o=np.asarray(ec.process(mic[i:i+160].astype(np.float32), ref[i:i+160].astype(np.float32)),dtype=np.float64)
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
for d in sorted(glob.glob('/home/rhy/voxa-bench-data/smoke/*fst-0[034]*')):
    t=json.load(open(d+'/truth.json')); e=t['echo']
    mic=load(d+'/mic.wav'); ref=load(d+'/ref.wav'); n=min(len(mic),len(ref)); mic,ref=mic[:n],ref[:n]
    ppm=e['drift_ppm']
    print(d.split('/')[-1],'delay',e['delay_samples'],'rt60',round(e['rt60_s'],2),'drift',round(ppm))
    for name,f in (('raw',1.0),('rs+',1.0+ppm/1e6*0),):
        pass
    out=aec3_run(mic,ref); v=pool(mic,out,ref)
    print('   raw      med',round(float(np.median(v)),1),'p10',round(float(np.percentile(v,10)),1),'mean',round(float(np.mean(v)),1),'n',len(v))
    for sign in (1,-1):
        rr=resample_lin(ref,1+sign*abs(ppm)/1e6)
        rr=np.concatenate([rr, np.zeros(max(0,n-len(rr)))])[:n]
        out2=aec3_run(mic,rr); v2=pool(mic,out2,ref)
        f2=1+sign*abs(ppm)/1e6
        print(f'   rs({sign:+d},f={f2:.6f}) med',round(float(np.median(v2)),1),'p10',round(float(np.percentile(v2,10)),1),'mean',round(float(np.mean(v2)),1))
