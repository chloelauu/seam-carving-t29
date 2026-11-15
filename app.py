"""Interactive Streamlit seam-carving demonstration (bonus a and b)."""
from __future__ import annotations
import io
import json
import time
import hashlib
import numpy as np
from PIL import Image
import streamlit as st
from seamcarve.core import (MAX_GRAPH_PIXELS, SOLVERS, carve, energy_map,
                            overlay_seam, all_seams_overlay, resize_max_side)

st.set_page_config(page_title='Seam Carving Lab', page_icon='🧵', layout='wide')
st.title('Seam Carving Lab')
st.caption('Dynamic Programming vs Greedy vs Graph Cut | interactive target sizes | seam-by-seam trace')

with st.sidebar:
    st.header('Image and algorithm')
    uploaded = st.file_uploader('Upload an image', type=['png','jpg','jpeg','webp'])
    algorithm = st.selectbox('Seam-selection method', list(SOLVERS))
    scale_limit = st.slider('Maximum image side (preprocessing)', 32, 256,
                            64 if algorithm.startswith('Graph') else 160, step=8,
                            help='Graph cut uses a pure Python max-flow solver. Keep its pixel count <= 3600.')
    order = st.radio('Seam-removal order', ['Width first','Height first'], horizontal=True)

if uploaded:
    try:
        original = resize_max_side(Image.open(uploaded), scale_limit)
    except Exception as exc:
        st.error(f'Unable to decode image: {exc}')
        st.stop()
else:
    st.info('Upload a photo to start, or use the included example image.')
    original = np.asarray(Image.open('assets/example.png').convert('RGB'))
    if original.shape[1] > scale_limit or original.shape[0] > scale_limit:
        original = resize_max_side(Image.fromarray(original),scale_limit)

h,w=original.shape[:2]
if algorithm.startswith('Graph') and h*w > MAX_GRAPH_PIXELS:
    st.warning(f'Graph cut allows up to {MAX_GRAPH_PIXELS} pixels (current: {h*w}). Reduce the maximum image side.')
with st.sidebar:
    st.header('Target size')
    if w == 1:
        target_w = 1
        st.caption('Target width: 1 px (already at minimum)')
    else:
        target_w=st.slider('Target width (px)',1,w,max(1,w-min(10,w-1)))
    if h == 1:
        target_h = 1
        st.caption('Target height: 1 px (already at minimum)')
    else:
        target_h=st.slider('Target height (px)',1,h,max(1,h-min(6,h-1)))
    st.caption(f'Current source: {w} x {h} px | removing {w-target_w} vertical and {h-target_h} horizontal seams')
    run=st.button('Carve image',type='primary',width="stretch")

request = (hashlib.sha256(original.tobytes()).hexdigest(), original.shape,
           target_w, target_h, algorithm, order)
if st.session_state.get('carve_request') != request:
    st.session_state.pop('carve_result', None)
if run:
    try:
        start=time.perf_counter()
        result,steps=carve(original,target_w,target_h,algorithm,order,record=True)
        elapsed=time.perf_counter()-start
        st.session_state['carve_result'] = (result, steps, elapsed)
        st.session_state['carve_request'] = request
    except Exception as exc:
        st.error(f'Carving was not completed: {exc}')
        st.stop()
if 'carve_result' not in st.session_state:
    st.image(original,caption=f'Input ({w} x {h})',width='stretch')
    st.stop()
result, steps, elapsed = st.session_state['carve_result']
st.success(f'Completed {len(steps)} seams in {elapsed:.3f} s. Output: {result.shape[1]} x {result.shape[0]} px.')

out_png=io.BytesIO()
Image.fromarray(result).save(out_png,format='PNG')
st.download_button('Download carved PNG',out_png.getvalue(),file_name='carved.png',mime='image/png',on_click='ignore')

results_tab,step_tab,all_tab,energy_tab,metrics_tab=st.tabs(['Before / after','Step-by-step','All seams on original','Energy & gradients','Metrics / export'])
with results_tab:
    left,right=st.columns(2)
    with left: st.image(original,caption=f'Original {w} x {h}',width='stretch')
    with right: st.image(result,caption=f'Carved {target_w} x {target_h}',width='stretch')
with step_tab:
    if not steps:
        st.info('No seams were removed: choose a smaller target dimension.')
    else:
        if len(steps) == 1:
            index = 1
            st.caption('Showing the only removal step')
        else:
            history_key = 'history_' + hashlib.sha256(repr(request).encode()).hexdigest()
            index=st.slider('Show removal step',1,len(steps),1,key=history_key)
        step=steps[index-1]
        annotated=overlay_seam(step.before,step.seam,step.orientation)
        st.image(annotated,caption=f'Step {index}/{len(steps)} - {step.orientation} seam, energy {step.cost:.1f} (pink = next removed seam)',width='stretch')
        st.caption('The image is shown immediately before removal. Move the slider to inspect every seam.')
with all_tab:
    st.image(all_seams_overlay(original,steps),caption='Every removed source pixel mapped back to the original image; hue shows removal order',width='stretch')
    st.caption('All seams include both vertical and horizontal removals. Overlap is impossible because each source pixel is removed at most once.')
with energy_tab:
    e=energy_map(original)
    hi=float(np.max(e))
    view=(255*e/(hi if hi>0 else 1)).astype(np.uint8)
    st.image(view,caption='Input Sobel L1 energy map: brighter pixels have stronger gradients',width='stretch',clamp=True)
with metrics_tab:
    st.metric('Seams removed',len(steps))
    st.metric('Elapsed (seconds)',f'{elapsed:.4f}')
    if steps:
        st.line_chart({'Seam energy':[s.cost for s in steps]},x_label='Removal step',y_label='Summed backward energy')
    metadata={'algorithm':algorithm,'source':[w,h],'target':[target_w,target_h],
              'order':order,'duration_s':elapsed,
              'steps':[{'number':s.number,'direction':s.orientation,'cost':s.cost,
                        'seam':s.seam.tolist()} for s in steps]}
    st.download_button('Download step-by-step JSON',json.dumps(metadata,indent=2).encode(),file_name='seam_history.json',mime='application/json',on_click='ignore')
