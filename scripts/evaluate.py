"""Reproduce the report evidence: python -m scripts.evaluate from the repo root."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time
from statistics import median
import importlib.metadata
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seamcarve.core import SOLVERS, carve, energy_map, seam_cost, resize_max_side

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/evidence'
OUT.mkdir(exist_ok=True)
METHODS = list(SOLVERS)


def timed(function, repeats=5):
    function()  # excluded warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter()-start)*1000)
    return result, samples


def main():
    records=[]
    examples=[('Synthetic', ROOT/'assets/example.png')]
    examples += [(n.title(),ROOT/'assets/photos'/f) for n,f in [('astronaut','astronaut.png'),('coffee','coffee.png'),('rocket','rocket.jpg')]]
    fig, axes=plt.subplots(3,4,figsize=(11.6,8.3),layout='constrained')
    for row, (name,path) in enumerate(examples):
        pil=Image.open(path).convert('RGB')
        image=np.asarray(pil) if name=='Synthetic' else resize_max_side(pil,60)
        h,w=image.shape[:2]
        tw,th=(58,44) if name=='Synthetic' else (round(w*.75),h)
        energy, etimes=timed(lambda:energy_map(image))
        saved={}
        for method in METHODS:
            solver=SOLVERS[method]
            first, stimes=timed(lambda:solver(energy))
            (result,steps), times=timed(lambda:carve(image,tw,th,method,record=True))
            saved[method]=result
            records.append({'image':name,'method':method,'source':[w,h],'target':[tw,th],
                'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'median_total_ms':median(times),'trials_ms':times,
                'initial_energy_ms':median(etimes),'initial_solver_ms':median(stimes),
                'first_seam_energy':seam_cost(energy,first),
                'sum_seam_energies':sum(s.cost for s in steps),
                'step_costs':[s.cost for s in steps]})
            print(f'{name} / {method}: {median(times):.3f} ms',flush=True)
        if row>0:
            for col, (label,im) in enumerate([('Input',image)]+list(saved.items())):
                axes[row-1,col].imshow(im,interpolation='nearest')
                axes[row-1,col].set_xlim(-0.5,w-0.5)
                axes[row-1,col].set_ylim(h-0.5,-0.5)
                axes[row-1,col].set_title(f'{name}: {label}\n{im.shape[1]} x {im.shape[0]}',fontsize=9)
                axes[row-1,col].axis('off')
        records[-1]['same_pixels_as_dp']=bool(np.array_equal(saved[METHODS[0]],saved[METHODS[2]]))
    fig.suptitle('Real photographs | same 60-pixel maximum input side | 25% width reduction',fontsize=12)
    fig.savefig(OUT/'figure5_photos.png',dpi=185,bbox_inches='tight')
    plt.close(fig)

    # Show input seams separately from final appearance.
    fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained')
    synthetic=[r for r in records if r['image']=='Synthetic']
    vals=[r['median_total_ms'] for r in synthetic]
    ax.bar(['DP','Greedy','Graph cut'],vals,color=['#245D80','#B96C30','#4C7A62'])
    ax.set_yscale('log')
    ax.set_ylabel('Median end-to-end time (ms, log scale)')
    ax.set_title('Synthetic 64 x 48 to 58 x 44 | 1 warm-up + 5 measured runs')
    for i,v in enumerate(vals):ax.annotate(f'{v:.2f}',(i,v),xytext=(0,4),textcoords='offset points',ha='center')
    ax.set_ylim(min(vals)*.5,max(vals)*3)
    fig.savefig(OUT/'figure3_timing.png',dpi=185,bbox_inches='tight')
    plt.close(fig)
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended']]
    ram=None
    if sys.platform=='win32':
        m=MemoryStatus();m.length=ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram=round(m.total_phys/(1024**3),2)
    metadata={'environment':{'python':platform.python_version(),'os':platform.platform(),
        'processor':os.environ.get('PROCESSOR_IDENTIFIER','not recorded'),'logical_cpus':os.cpu_count(),
        'ram_gib':ram,'gpu_used':False,
        'packages':{n:importlib.metadata.version(n) for n in ['numpy','Pillow','streamlit','matplotlib']}},
        'protocol':'Width first; Sobel L1; edge padding; one warm-up and five timed runs; total includes provenance and full frame history; first-map and first-solver timings are independent medians, not additive components of total.',
        'records':records}
    (OUT/'evaluation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
