from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
out=Path(__file__).resolve().parents[1]/'assets'/'evidence'/'figure4_ci_cd.png'
fig,ax=plt.subplots(figsize=(11.2,2.7));ax.set_xlim(0,11.2);ax.set_ylim(0,2.7);ax.axis('off')
labels=[('Push / PR','GitHub trigger'),('Validate','Compile + Ruff'),('Test','pytest 36 cases'),('Build','Docker image'),('Publish','GHCR tags')]
for i,(name,sub) in enumerate(labels):
    x=.22+i*2.2
    b=FancyBboxPatch((x,.72),1.88,1.25,boxstyle='round,pad=.12,rounding_size=.13',lw=1.3,ec='#244363',fc='#e9f2fa')
    ax.add_patch(b)
    ax.text(x+.94,1.5,name,ha='center',va='center',weight='bold',fontsize=12,color='#1b3656')
    ax.text(x+.94,1.15,sub,ha='center',va='center',fontsize=9,color='#40566e')
    if i<4:
      ax.add_patch(FancyArrowPatch((x+2.0,1.34),(x+2.15,1.34),arrowstyle='-|>',mutation_scale=12,color='#317485',lw=2))
ax.text(7.7,.27,'Main branch only; publish requires passing tests. PRs stop after testing.',ha='center',fontsize=9,color='#71552d')
fig.savefig(out,dpi=200,bbox_inches='tight',facecolor='white')
plt.close(fig)
