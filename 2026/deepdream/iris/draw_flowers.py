"""Draw measured Iris feature vectors as schematics, not reconstructed photos."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from PIL import Image
import io

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'results/model.json').read_text())


def draw_flower(ax, frame, title):
    sepal_length,sepal_width,petal_length,petal_width=frame['x']
    # All panels use the same length scale. Each ellipse's full long/short
    # axes equal the actual measured or optimized length/width in cm.
    for length,width,offset,color in [(sepal_length,sepal_width,30,'#74a77a'),
                                     (petal_length,petal_width,90,'#9b79cf')]:
        for angle in [offset,offset+120,offset+240]:
            rad=np.deg2rad(angle)
            center=(length*.46*np.cos(rad),length*.46*np.sin(rad))
            ax.add_patch(Ellipse(center,length,width,angle=angle,facecolor=color,
                                 edgecolor='white',linewidth=.8,alpha=.85))
    ax.add_patch(Circle((0,0),.25,facecolor='#e8bc58',edgecolor='white'))
    ax.set(xlim=(-7.5,7.5),ylim=(-7.5,7.5),aspect='equal')
    ax.axis('off');ax.set_title(title,fontsize=13,pad=10)
    ax.plot([-6.6,-4.6],[-6.5,-6.5],color='#4b5563',linewidth=2)
    ax.text(-5.6,-6.05,'2 cm',ha='center',fontsize=9,color='#4b5563')
    ax.text(.5,-.01,f'Sepal {sepal_length:.2f} × {sepal_width:.2f} cm\n'
            f'Petal {petal_length:.2f} × {petal_width:.2f} cm\n'
            f'h3 = {frame["h"][data["neuron"]]:+.3f}',transform=ax.transAxes,
            ha='center',va='top',fontsize=11,linespacing=1.5)


def figure_at(step):
    fig,axes=plt.subplots(1,3,figsize=(12,5.4))
    fig.subplots_adjust(top=.83,bottom=.21,wspace=.04,left=.02,right=.98)
    frames=[data['paths']['increase'][step],data['paths']['increase'][0],data['paths']['decrease'][step]]
    titles=['Increase h3: preferred input','Original Iris sample','Decrease h3: opposite input']
    for ax,frame,title in zip(axes,frames,titles):draw_flower(ax,frame,title)
    fig.suptitle(f'Iris feature visualization — input optimization, step {step}/40',fontsize=16,y=.97)
    fig.text(.5,.025,'Green = sepals  |  Purple = petals  |  Same scale in every panel\n'
             'Schematic from four numerical features; not a generated photograph or botanical reconstruction.',
             ha='center',fontsize=10,color='#4b5563')
    return fig


fig=figure_at(40)
fig.savefig(ROOT/'results/flower-input-ascent.png',dpi=160,facecolor='white')
plt.close(fig)
frames=[]
for step in range(0,41,2):
    fig=figure_at(step);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=90,facecolor='white')
    plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB'))
frames[0].save(ROOT/'results/flower-input-ascent.gif',save_all=True,append_images=frames[1:],
               duration=[150]*20+[1800],loop=0)
print('Saved flower schematics and optimization animation.')
