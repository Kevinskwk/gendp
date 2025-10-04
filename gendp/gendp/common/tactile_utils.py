import sys
import cv2
import numpy as np

sys.path.append('/home/kevin/gendp/GelsightKCL')
from A_utility import marker_center #, process_frame
import find_marker

sys.path.append("/home/kevin/gendp/gsrobotics")
from utilities.reconstruction import Reconstruction3D

""" example setting:
left (sensor 1) setting:
    N: 9
    M: 7
    fps: 10
    x0: 37.5
    y0: 34.5
    dx: 29.6
    dy: 28.7

right (sensor 0) setting:
    N: 9
    M: 7
    fps: 10
    x0: 46
    y0: 40.6
    dx: 29.1
    dy: 28.6
"""

"""
sensor 0 four corners: [45.24346161 42.05607986] [278.15216064  39.16341782] [ 46.84062576 213.95736694] [279.32937622 210.70906067]
depth_data -0.6744279443537669
sensor 1 four corners: [37.05445099 35.32085037] [273.55630493  33.63690567] [ 37.90210724 208.36355591] [275.09109497 204.98249817]
depth_data 0.15854205770625007
sensor 0 four corners: [45.25456619 42.05829239] [278.13012695  39.16272354] [ 46.85464096 213.94456482] [279.32089233 210.7303009 ]
depth_data -0.7526107542955923
sensor 1 four corners: [37.05379868 35.32396698] [273.55322266  33.64050293] [ 37.90234375 208.36663818] [275.07794189 205.01512146]
depth_data 0.17082087009672134
sensor 0 four corners: [45.25065231 42.05060196] [278.14535522  39.14365768] [ 46.85033798 213.95404053] [279.34075928 210.71395874]
depth_data -0.5626435200018542
sensor 1 four corners: [37.04814529 35.3170166 ] [273.53894043  33.63344193] [ 37.88949203 208.36857605] [275.09564209 204.98152161]
"""

def get_initial_marker_positions(marker_config):
    x0 = marker_config['x0']
    y0 = marker_config['y0']
    dx = marker_config['dx']
    dy = marker_config['dy']
    N = marker_config['N']
    M = marker_config['M']
    xs = np.arange(x0, x0 + dx * N, dx)
    ys = np.arange(y0, y0 + dy * M, dy)
    xv, yv = np.meshgrid(xs, ys)
    positions = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # (N*M, 2)
    return positions

class TactileProcessor:
    def __init__(self,
                 width=320,
                 height=240,
                 nn_model_path='~/gendp/gsrobotics/models/nnmini.pt',
                 ref_img : str=None,  # if provided, perform 50 runs with this ref img to zero depth estimation
                 marker_config=None,
                 use_gpu=True,
                 marker_mask_min=0,
                 marker_mask_max=70):
        
        self.marker_config = marker_config
        self.marker_mask_min = marker_mask_min
        self.marker_mask_max = marker_mask_max
        
        self.reconstruction = Reconstruction3D(
            image_width=width,
            image_height=height,
            use_gpu=use_gpu
        )
        
        if self.reconstruction.load_nn(nn_model_path) is None:
            raise ValueError(f"Failed to load neural network model from {nn_model_path}")

        self.initial_positions = get_initial_marker_positions(self.marker_config)  # (N, M, 2)
        self.m = find_marker.Matching(
            N_=self.marker_config['N'], 
            M_=self.marker_config['M'], 
            fps_=self.marker_config['fps'], 
            x0_=self.marker_config['x0'], 
            y0_=self.marker_config['y0'], 
            dx_=self.marker_config['dx'], 
            dy_=self.marker_config['dy'])
        
        if ref_img is not None:
            img = cv2.imread(ref_img)
            print("Warming up depth estimation with reference image...")
            for _ in range(51):
                self.reconstruction.get_depthmap(
                    image=img,
                    markers_threshold=(self.marker_mask_min, self.marker_mask_max)
                )
            print("Warming up done.")

    def get_depth(self, frame, marker_positions=None):
        depth_map, contact_mask, grad_x, grad_y = self.reconstruction.get_depthmap(
            image=frame,
            markers_threshold=(self.marker_mask_min, self.marker_mask_max)
        )
        if marker_positions is None:
            marker_positions = self.initial_positions
        # Sample depth values at initial marker positions
        marker_depths = self._sample_depth_at_markers(depth_map, marker_positions)
        return depth_map, contact_mask, grad_x, grad_y, marker_depths
    
    def _sample_depth_at_markers(self, depth_map, marker_positions):
        """
        Sample depth values at given marker positions from the depth map.
        
        Args:
            depth_map (np.ndarray): Depth map (H, W) as float32
            marker_positions (np.ndarray): Marker positions (N, M, 2) as [x, y]
            
        Returns:
            np.ndarray: Depth values at each marker position (N, M) with NaN for out-of-bounds
        """
        sampled_depths = np.zeros(marker_positions.shape[:2], dtype=np.float32)  # (N, M)
        h, w = depth_map.shape

        for i in range(marker_positions.shape[0]):
            for j in range(marker_positions.shape[1]):
                pos = marker_positions[i, j]
                x, y = int(round(pos[0])), int(round(pos[1]))
                if 0 <= y < w and 0 <= x < h:
                    sampled_depths[i, j] = depth_map[x, y]
                else:
                    sampled_depths[i, j] = 0

        return sampled_depths

    def get_marker_flow(self, frame):    
        # frame = process_frame(frame)
        mc = marker_center(frame, debug=False)
        self.m.init(mc)
        self.m.run()
        flow = self.m.get_flow()  # (5, N, M)

        # points = np.asarray(flow)[:4, :, :].reshape(setting['N'] * setting['M'], 4)
        initial_positions = np.asarray(flow, dtype=np.float32)[:2, :, :].transpose(1, 2, 0) # (N, M, 2)
        points = np.asarray(flow, dtype=np.float32)[2:4, :, :].transpose(1, 2, 0) # (N, M, 2)

        return initial_positions, points

    def process_frame(self, frame):
        initial_positions, points = self.get_marker_flow(frame)

        depth_map, contact_mask, grad_x, grad_y, marker_depths = self.get_depth(frame, initial_positions)
        
        # Calculate displacement field and combine with depth
        displacement = points - initial_positions
        force_field = np.concatenate([displacement, marker_depths[:, :, None]], axis=-1)
        
        # Note: force_field is now sorted using M * (x-x0)/dx + (y-y0)/dy formula
        # and can be reshaped to (N, M) where N=cols, M=rows
        # Grid[x,y] corresponds to column x, row y in the marker grid
        return force_field

    def process_sequence(self, frames):
        force_fields = []
        for frame in frames:
            force_field = self.process_frame(frame)
            force_fields.append(force_field)
        return np.stack(force_fields, axis=0)


# Legacy function for backward compatibility
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
        # frame = process_frame(frame)
        mc = marker_center(frame, debug=False)
        m.init(mc)
        m.run()
        flow = m.get_flow()  # (5, N, M)

        # points = np.asarray(flow)[:4, :, :].reshape(setting['N'] * setting['M'], 4)
        points = np.asarray(flow, dtype=np.float32)[:4, :, :].reshape(4, setting['N'] * setting['M'])

        force_fields.append(points)

    return np.stack(force_fields, axis=0)
