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
def aec3_run(mic, ref, hint_ms, mask=True, recon=None):
    # recon: (at_index, hint) rebuild the canceller mid-stream with a new hint
    ec=AEC3(sample_rate=SR,num_channels=1,stream_delay_ms=hint_ms)
    m=ReferenceMask(rate=SR)
    out=[]; curve=[]
    for i in range(0,len(mic),160):
        if i+160>len(mic): break
        if recon is not None and i==recon[0]:
            ec=AEC3(sample_rate=SR,num_channels=1,stream_delay_ms=recon[1]); m=ReferenceMask(rate=SR)
        o=np.asarray(ec.process(mic[i:i+160].astype(np.float32), ref[i:i+160].astype(np.float32)),dtype=np.float64)
        o=np.asarray(m.process(o, ref[i:i+160]), dtype=np.float64)
        pm=np.mean(mic[i:i+160]**2); po=np.mean(o**2)
        curve.append(10*np.log10(pm/max(po,1e-20)) if pm>1e-12 else 0.0)
        out.append(o)
    return np.concatenate(out), curve
for d in sorted(glob.glob('/home/rhy/voxa-bench-data/smoke/smoke-000[034]-fst*')):
    t=json.load(open(d+'/truth.json')); e=t['echo']
    mic=load(d+'/mic.wav'); ref=load(d+'/ref.wav'); n=min(len(mic),len(ref)); mic,ref=mic[:n],ref[:n]
    bulk=int(e['delay_samples'])
    print('===',d.split('/')[-1],'delay',bulk,'rt60',round(e['rt60_s'],2))
    for hint in (0, round(bulk/16)):
        out,curve=aec3_run(mic,ref,hint)
        v=[c for j,c in enumerate(curve) if (j+1)*160/SR>=2.0 and np.mean(ref[j*160:j*160+160]**2)>1e-6]
        if v: print(f'  hint{hint:4d} med {float(np.median(v)):6.1f} p10 {float(np.percentile(v,10)):6.1f} p25 {float(np.percentile(v,25)):6.1f} n {len(v)}')
    # mid-stream rebuild with hint at 2s
    for hint in (round(bulk/16),):
        out,curve=aec3_run(mic,ref,0,recon=(int(2.0*SR),hint))
        v=[c for j,c in enumerate(curve) if (j+1)*160/SR>=2.0 and np.mean(ref[j*160:j*160+160]**2)>1e-6]
        if v: print(f'  recon@2s hint{hint:4d} med {float(np.median(v)):6.1f} p10 {float(np.percentile(v,10)):6.1f} n {len(v)}')
print('--- curve 0004 ---')
d='/home/rhy/voxa-bench-data/smoke/smoke-0004-fst-04'
t=json.load(open(d+'/truth.json')); e=t['echo']
mic=load(d+'/mic.wav'); ref=load(d+'/ref.wav'); n=min(len(mic),len(ref)); mic,ref=mic[:n],ref[:n]
out,curve=aec3_run(mic,ref,0)
import sys
for j,c in enumerate(curve):
    t0=j*160/SR
    rp=ref[j*160:j*160+160]
    if len(rp)<160: break
    if np.mean(rp*rp)>1e-6: print(f'{t0:5.2f}s {c:6.1f} rp {10*np.log10(np.mean(rp*rp)):6.1f} mp {10*np.log10(np.mean(mic[j*160:j*160+160]**2)):6.1f}')
