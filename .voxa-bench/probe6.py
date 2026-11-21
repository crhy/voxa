import sys; sys.path.insert(0,'.')
import json, glob
import numpy as np
from pywebrtc_audio import EchoCanceller as AEC3
SR=16000
def load(p):
    import wave
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)/32768.0

class TailMask:
    """Per-band mask whose echo prediction holds the reference with a decay learned per band-stream."""
    def __init__(self, rate=16000, frame=512, hop=256, bands=64, over=1.5, floor=0.1, memory_s=0.4):
        self.rate=rate; self.frame=frame; self.hop=hop; self.nbands=bands
        self.over=over; self.floor=floor
        hann=0.5-0.5*np.cos(2*np.pi*np.arange(frame)/frame); self.win=np.sqrt(hann)
        edges=np.linspace(1,frame//2+1,bands+1).astype(int)
        self.band_idx=[np.arange(edges[b],edges[b+1]) for b in range(bands)]
        self.mem_frames=max(4,int(memory_s*rate/hop))
        self.coupling=np.zeros(bands); self.ratios=[deque_() for _ in range(bands)]
        self.held=np.zeros(bands)
        self.rt_grid=np.array([0.2,0.3,0.4,0.6,0.8,1.0])
        self.rt=0.4
        self.lag=0; self.mic_buf=np.zeros(0); self.ref_buf=np.zeros(0)
        self.out_num=np.zeros(0); self.out_den=np.zeros(0); self.frames_done=0; self.emitted=0
        self.excess_hist=[]
    def _estimate_lag(self):
        n=min(len(self.mic_buf),len(self.ref_buf))
        if n<self.frame*4: return self.lag
        mic=self.mic_buf[-n:]; ref=self.ref_buf[-n:]
        size=1
        while size<2*n: size<<=1
        fm=np.fft.rfft(mic,size); fr=np.fft.rfft(ref,size)
        corr=np.fft.irfft(fm*np.conj(fr),size)
        k=int(np.argmax(np.abs(corr)))
        if k>size//2: k-=size
        return int(k)
    def stats(self, mic, ref):
        """yield per-frame (mic_b, held_pred_b) — used for detection tests."""
        self.mic_buf=np.concatenate([self.mic_buf,mic]); self.ref_buf=np.concatenate([self.ref_buf,ref])
        total=len(self.mic_buf)
        if total>=self.frame:
            nframes=(total-self.frame)//self.hop+1
            if self.frames_done==0 or (nframes-self.frames_done)%8==0:
                self.lag=self._estimate_lag()
            for f in range(self.frames_done,nframes):
                s=f*self.hop
                mf=self.mic_buf[s:s+self.frame]
                rs=s-self.lag; rs=max(0,min(rs,max(0,len(self.ref_buf)-self.frame)))
                rf=self.ref_buf[rs:rs+self.frame]
                fm=np.fft.rfft(mf); fr=np.fft.rfft(rf)
                pm=np.abs(fm)**2; pr=np.abs(fr)**2
                mic_b=np.array([pm[idx].sum() if idx.size else 0.0 for idx in self.band_idx])
                ref_b=np.array([pr[idx].sum() if idx.size else 0.0 for idx in self.band_idx])
                yield s, mic_b, ref_b
                # update coupling only where the reference is currently speaking
                for b in range(self.nbands):
                    if ref_b[b]>1e-10 and ref_b[b]>=0.25*np.max(ref_b):
                        self.ratios[b].append(mic_b[b]/ref_b[b])
                        r=sorted(self.ratios[b]); self.coupling[b]=r[len(r)//2]
            self.frames_done=nframes

from collections import deque
def deque_(): return deque(maxlen=80)

def run_stack(mic, ref, rt=None, over=1.5, floor=0.1, bypass_K=None):
    """AEC3 -> tail-aware mask -> optional frame bypass. Returns per-frame dict with erle_db, atten_db, excess, echo_pred."""
    ec=AEC3(sample_rate=SR,num_channels=1,stream_delay_ms=0)
    out=[]; rec=[]
    # mask state
    tm=TailMask(over=over,floor=floor)
    # we drive the mask at 20 ms chunks for simplicity
    win=tm.win; frame=tm.frame; hop=tm.hop
    out_num=np.zeros(0); out_den=np.zeros(0); emitted=0
    mic_buf=np.zeros(0); ref_buf=np.zeros(0)
    # simple per-frame loop at 10 ms for AEC3, then 20 ms for mask+metrics
    aec_out=np.zeros(len(mic))
    for i in range(0,len(mic)-159,160):
        o=np.asarray(ec.process(mic[i:i+160].astype(np.float32), ref[i:i+160].astype(np.float32)),dtype=np.float64)
        aec_out[i:i+len(o)]=o
    # RT learning: try rt values on whole signal, choose best (probe only: offline, honest to per-clip learning)
    best=None
    for rt_try in tm.rt_grid:
        err=[]
        lag=tm._estimate_lag() if False else None
        # compute held envelope with decay
        d=10**(-6.9*hop/SR/rt_try)
        held=np.zeros(tm.nbands); errs=[]
        total=len(aec_out)
        for s in range(0,total-frame+1,hop):
            mf=aec_out[s:s+frame]; rs=max(0,s+lag0 if (lag0:=0) else s) # lag handled by ring alignment in product
            rf=ref[max(0,min(s, len(ref)-frame)):max(0,min(s, len(ref)-frame))+frame]
            fm=np.fft.rfft(mf); fr=np.fft.rfft(rf)
            pm=np.abs(fm)**2; pr=np.abs(fr)**2
            mic_b=np.array([pm[ix].sum() if ix.size else 0.0 for ix in tm.band_idx])
            ref_b=np.array([pr[ix].sum() if ix.size else 0.0 for ix in tm.band_idx])
            held=np.maximum(held*d, ref_b)
            pred=tm.coupling*held
            err.append(np.median(np.abs(mic_b-pred)/np.maximum(mic_b,1e-12)))
        m=np.mean(err)
        if best is None or m<best[0]: best=(m,rt_try)
    tm.rt=best[1]
    print('   learned rt',tm.rt,'err',round(best[0],3))
    d=10**(-6.9*hop/SR/tm.rt)
    held=np.zeros(tm.nbands); lag=None
    s_list=[]; mic_b_list=[]; ref_b_list=[]; aec_b_list=[]
    for s in range(0,len(mic)-frame+1,hop):
        s_list.append(s); rf=ref[max(0,min(s,len(ref)-frame)):max(0,min(s,len(ref)-frame))+frame]
        aec_b_list.append(None)
    # per-frame pass: AEC3 output -> mask (held prediction) -> optional bypass
    for s in range(0,len(mic)-frame+1,hop):
        mf=aec_out[s:s+frame]
        rf=ref[max(0,min(s,len(ref)-frame)):max(0,min(s,len(ref)-frame))+frame]
        fm=np.fft.rfft(mf); fr=np.fft.rfft(rf)
        pm=np.abs(fm)**2; pr=np.abs(fr)**2
        mic_b=np.array([pm[ix].sum() if ix.size else 0.0 for ix in tm.band_idx])
        ref_b=np.array([pr[ix].sum() if ix.size else 0.0 for ix in tm.band_idx])
        held=np.maximum(held*d, ref_b)
        pred=tm.coupling*held
        excess=float(np.sum(np.maximum(mic_b-over*pred,0)))
        rec.append((s,mic_b,pred,excess))
        # coupling update
        for b in range(tm.nbands):
            if ref_b[b]>1e-10 and ref_b[b]>=0.25*np.max(ref_b):
                tm.ratios[b].append(mic_b[b]/ref_b[b]); r=sorted(tm.ratios[b]); tm.coupling[b]=r[len(r)//2]
        gain_b=np.where(mic_b>=over*pred,1.0,floor)
        mag=np.sqrt(pm); phase=np.angle(fm)
        bg=np.ones(pm.shape[0])
        for b in range(tm.nbands): bg[tm.band_idx[b]]=gain_b[b]
        rebuilt=np.fft.irfft((mag*bg)*np.exp(1j*phase),frame)
        out_num=np.concatenate([out_num,np.zeros(max(0,s+frame-len(out_num)))]) if len(out_num)<s+frame else out_num
        out_den=np.concatenate([out_den,np.zeros(max(0,s+frame-len(out_den)))]) if len(out_den)<s+frame else out_den
        out_num[s:s+frame]+=rebuilt*win; out_den[s:s+frame]+=win**2
    clean=np.where(out_den>1e-9,out_num/np.maximum(out_den,1e-9),aec_out[:len(out_num)])
    return clean, rec

if __name__=='__main__':
    for d in sorted(glob.glob('/home/rhy/voxa-bench-data/smoke/smoke-000[04]-fst*')):
        t=json.load(open(d+'/truth.json')); e=t['echo']
        mic=load(d+'/mic.wav'); ref=load(d+'/ref.wav'); n=min(len(mic),len(ref)); mic,ref=mic[:n],ref[:n]
        clean,rec=run_stack(mic,ref)
        vals=[]
        for j,s in enumerate(range(0,len(mic)-160+1,160)):
            tsec=(s+160)/SR
            if tsec<2.0: continue
            rp=ref[s:s+160]
            if len(rp)<160 or np.mean(rp*rp)<=1e-6: continue
            pm=np.mean(mic[s:s+160]**2); po=np.mean(clean[s:s+160]**2)
            if pm<1e-12: continue
            vals.append(10*np.log10(pm/max(po,1e-20)))
        print(d.split('/')[-1],'pool med',round(float(np.median(vals)),1),'p10',round(float(np.percentile(vals,10)),1),'n',len(vals))
