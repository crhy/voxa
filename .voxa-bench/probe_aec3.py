import json, glob, sys
import numpy as np
sys.path.insert(0, '.')
from pywebrtc_audio import EchoCanceller as AEC3
from voxa.anc import EchoCanceller, estimate_delay

SR = 16000
def load(p):
    import wave
    with wave.open(p) as w:
        assert w.getframerate()==SR, w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)/32768.0
    return x

def erle_pool(mic, out, ref):
    vals=[]
    n=min(len(mic),len(out),len(ref))
    for i in range(0,n,160):
        m=mic[i:i+160]; r=ref[i:i+160]; o=out[i:i+160]
        t=(i+160)/SR
        if t<2.0: continue
        if np.mean(r*r)<=1e-6: continue
        pm=np.mean(m*m); po=np.mean(o*o)
        if pm<1e-12: continue
        vals.append(10*np.log10(pm/max(po,1e-20)))
    return vals

def aec3_run(mic, ref, shift=0):
    ec=AEC3(sample_rate=SR,num_channels=1,stream_delay_ms=0)
    ref_s=np.concatenate([np.zeros(shift), ref])[:len(ref)]
    out=[]
    for i in range(0,len(mic),160):
        m=mic[i:i+160]; r=ref_s[i:i+160]
        if len(m)<160: break
        o=np.asarray(ec.process(m.astype(np.float32), r.astype(np.float32)),dtype=np.float64)
        out.append(o)
    return np.concatenate(out)

for p in sorted(glob.glob('/home/rhy/voxa-bench-data/smoke/*fst-*/truth.json'))[:4]:
    d=p.rsplit('/',1)[0]
    t=json.load(open(p)); e=t['echo']
    mic=load(d+'/mic.wav'); ref=load(d+'/ref.wav'); n=min(len(mic),len(ref))
    mic,ref=mic[:n],ref[:n]
    bulk=int(e['delay_samples'])
    print(d.split('/')[-1], 'delay',bulk,'rt60',round(e['rt60_s'],2),'drift',round(e['drift_ppm']))
    for name,out in (('aec3',aec3_run(mic,ref)),
                     ('aec3+shift',aec3_run(mic,ref,bulk)),
                     ):
        v=erle_pool(mic,out,ref)
        if v: print('   ',name,'median',round(float(np.median(v)),1),'p10',round(float(np.percentile(v,10)),1),'n',len(v))
        else: print('   ',name,'no frames')
