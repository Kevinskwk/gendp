import sys
sys.path.append('/users/kevinma/gendp/GelsightKCL')
from A_utility import marker_center, process_frame
import find_marker
import numpy as np

def force_field_proc(frames, setting):
    m = find_marker.Matching(
        N_=setting['N'], 
        M_=setting['M'], 
        fps_=setting['fps'], 
        x0_=setting['x0'], 
        y0_=setting['y0'], 
        dx_=setting['dx'], 
        dy_=setting['dy'])
    
    force_fields = []
    
    for i, frame in enumerate(frames):
        frame = process_frame(frame)
        mc = marker_center(frame, debug=False)
        m.init(mc)
        m.run()
        flow = m.get_flow()  # (5, N, M)

        # points = np.asarray(flow)[:4, :, :].reshape(setting['N'] * setting['M'], 4)
        points = np.asarray(flow, dtype=np.float32)[:4, :, :].reshape(4, setting['N'] * setting['M'])

        force_fields.append(points)

    return np.stack(force_fields, axis=0)
