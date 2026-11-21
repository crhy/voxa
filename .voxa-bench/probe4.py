import sys; sys.path.insert(0,'.')
import json, glob
import numpy as np
from pywebrtc_audio import EchoCanceller as AEC3
from voxa.refmask import ReferenceMask
SR=16000
def load(p):
    import wave
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)/32768.0
def aec3_run(mic, ref, shift, mask=False):
    ec=AEC3(sample_rate=SR,num_channels=1,stream_delay_ms=0)
    m=ReferenceMask(rate=SR) if mask else None
    rsh=np.concatenate([np.zeros(shift), ref])
    out=[]; curve=[]
    for i in range(0,len(mic),160):
        if i+160>len(mic): break
        rr=rsh[i:i+160]
        o=np.asarray(ec.process(mic[i:i+160].astype(np.float32), rr.astype(np.float32)),dtype=np.float64)
        if m is not None:
            o=np.asarray(m.process(o, rr), dtype=np.float64)
        pm=np.mean(mic[i:i+160]**2); po=np.mean(o**2)
        curve.append(10*np.log10(pm/max(po,1e-20)) if pm>1e-12 else 0.0)
        out.append(o)
    return np.concatenate(out), curve
def pool(curve):
    v=[c for j,c in enumerate(curve) if (j+1)*160/SR>=2.0]
    return v
for d in sorted(glob.glob('/home/rhy/voxa-bench-data/smoke/smoke-000[04]-fst*')):
    t=json.load(open(d+'/truth.json')); e=t['echo']
    mic=load(d+'/mic.wav'); ref=load(d+'/ref.wav'); n=min(len(mic),len(ref)); mic,ref=mic[:n],ref[:n]
    bulk=int(e['delay_samples']); travel=320; queue=640
    print('===',d.split('/')[-1],'delay',bulk,'rt60',round(e['rt60_s'],2))
    for shift in (0, bulk, bulk+travel, bulk+travel+queue):
        for mask in (False,True):
            out,curve=aec3_run(mic,ref,shift,mask)
            v=pool(curve)
            if v: print(f'  shift{shift:5d} mask{int(mask)} med {float(np.median(v)):6.1f} p10 {float(np.percentile(v,10)):6.1f} n {len(v)}')
            else: print(f'  shift{shift:5d} mask{int(mask)} no frames')
