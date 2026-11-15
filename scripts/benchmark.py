"""Create honest local benchmark/evidence figures with fixed seed and source."""
from pathlib import Path
from statistics import median
import time, json, platform, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seamcarve.core import carve, energy_map, overlay_seam, all_seams_overlay, dp_seam,greedy_seam,graph_cut_seam,seam_cost

root=Path(__file__).resolve().parents[1]
output=root/'assets'/'evidence'
output.mkdir(exist_ok=True)
img=np.asarray(Image.open(root/'assets'/'example.png').convert('RGB'))
h,w=img.shape[:2]
tw,th=w-6,h-4
methods=['Dynamic programming','Greedy','Graph cut (max-flow)']
records=[]
results={}
for method in methods:
    times=[]
    result=None
    steps=None
    for k in range(5):
        start=time.perf_counter()
        result,steps=carve(img,tw,th,method,record=True)
        times.append((time.perf_counter()-start)*1000)
    results[method]=(result,steps)
    records.append({'method':method,'source':f'{w}x{h}','target':f'{tw}x{th}',
                    'median_total_ms':round(median(times),3),
                    'trials_ms':[round(x,3) for x in times],
                    'first_seam_energy':round(steps[0].cost,3),
                    'sum_seam_energies':round(sum(s.cost for s in steps),3)})

# Figure 1: input + Sobel energy + seam overlays (3 methods)
fig,axes=plt.subplots(2,3,figsize=(11.3,7.5),layout='constrained')
for ax in axes.flat: ax.axis('off')
axes[0,0].imshow(img);axes[0,0].set_title('Original (64 x 48)')
axes[0,1].imshow(energy_map(img),cmap='magma');axes[0,1].set_title('Input Sobel energy map')
axes[0,2].imshow(all_seams_overlay(img,results['Dynamic programming'][1]));axes[0,2].set_title('All 10 DP seams (original coords)')
for col,method in enumerate(methods):
    result,steps=results[method]
    axes[1,col].imshow(result)
    axes[1,col].set_title(method+'\nFinal '+str(tw)+' x '+str(th))
fig.suptitle('Sample implementation output | synthetic image; 6 width + 4 height reductions',fontsize=13,weight='bold')
fig.savefig(output/'figure1_comparison.png',dpi=190,bbox_inches='tight');plt.close(fig)

# Figure 2: first removal overlay and final original overlay for all algorithms
fig, axes=plt.subplots(2,3,figsize=(10.3,6.8),layout='constrained')
for i,method in enumerate(methods):
    _,steps=results[method]
    axes[0,i].imshow(overlay_seam(steps[0].before,steps[0].seam,steps[0].orientation))
    axes[0,i].set_title(method+'\nFirst seam, E='+str(round(steps[0].cost,1)))
    axes[1,i].imshow(all_seams_overlay(img,steps))
    axes[1,i].set_title('All 10 removed seams')
    for ax in (axes[0,i],axes[1,i]):ax.axis('off')
fig.suptitle('Step view and complete seam history',fontsize=13,weight='bold')
fig.savefig(output/'figure2_history.png',dpi=190,bbox_inches='tight');plt.close(fig)

# Figure 3 measured timings
fig,ax=plt.subplots(figsize=(8.2,3.8),layout='constrained')
labels=['DP','Greedy','Graph cut']
vals=[r['median_total_ms'] for r in records]
ax.bar(labels,vals)
ax.set_ylabel('Median end-to-end time (ms)')
ax.set_title('Synthetic 64 x 48 image -> 58 x 44 | 5 local runs')
for i,v in enumerate(vals):ax.text(i,v,f'{v:.1f} ms',ha='center',va='bottom',fontsize=9)
ax.set_ylim(0,max(vals)*1.18)
fig.savefig(output/'figure3_timing.png',dpi=185,bbox_inches='tight');plt.close(fig)

# Figure 4 exact fixed matrix cost
E=np.array([[1,2,50,50],[8,9,1,50],[50,49,1,50],[50,49,1,50]],dtype=float)
fixed={method:{'seam_1_based':(fun(E)+1).tolist(),'total_energy':seam_cost(E,fun(E))} for method,fun in [('DP',dp_seam),('Greedy',greedy_seam),('Graph cut',graph_cut_seam)]}
metadata={'environment':{'python':sys.version.split()[0],'platform':platform.platform(),'numpy':np.__version__},
          'benchmark_notes':'Synthetic example image from scripts/create_example.py. Five consecutive trials, same process, end-to-end carve including energy recomputation. Hardware-dependent; repeat on team machine.',
          'measured_records':records,'analytical_matrix':fixed}
(output/'benchmark.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata,indent=2))
