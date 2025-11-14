import os
from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
from gendp.real_world.single_realsense import SingleRealsense
from gendp.real_world.video_recorder import VideoRecorder

class MultiRealsense:
    def __init__(self,
        serial_numbers: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(1280,720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]]=None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
        verbose=False
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)

        advanced_mode_config = repeat_to_list(
            advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        vis_transform = repeat_to_list(
            vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(
            recording_transform, n_cameras, Callable)

        video_recorder = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)

        cameras = dict()
        camera_params = dict()
        for i, serial in enumerate(serial_numbers):
            params = {
                'shm_manager': shm_manager,
                'serial_number': serial,
                'resolution': resolution,
                'capture_fps': capture_fps,
                'put_fps': put_fps,
                'put_downsample': put_downsample,
                'record_fps': record_fps,
                'enable_color': enable_color,
                'enable_depth': enable_depth,
                'enable_infrared': enable_infrared,
                'get_max_k': get_max_k,
                'advanced_mode_config': advanced_mode_config[i],
                'transform': transform[i],
                'vis_transform': vis_transform[i],
                'recording_transform': recording_transform[i],
                'video_recorder': video_recorder[i],
                'verbose': verbose
            }
            cameras[serial] = SingleRealsense(**params)
            camera_params[serial] = params
        
        self.cameras = cameras
        self.camera_params = camera_params
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None, max_retries=3, retry_delay=2.0, stagger_delay=0.5):
        """
        Start all cameras.
        
        Args:
            wait: Whether to wait for all cameras to be ready
            put_start_time: Timestamp for synchronized frame capture
            max_retries: Maximum retry attempts for failed cameras
            retry_delay: Delay between retry attempts
            stagger_delay: Delay between starting each camera (helps with USB bandwidth)
        """
        if put_start_time is None:
            put_start_time = time.time()
        
        # Start cameras with staggered delay to avoid USB bandwidth issues
        for i, camera in enumerate(self.cameras.values()):
            camera.start(wait=False, put_start_time=put_start_time)
            if i < len(self.cameras) - 1 and stagger_delay > 0:
                time.sleep(stagger_delay)
        
        if wait:
            self.start_wait(max_retries=max_retries, retry_delay=retry_delay)
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self, max_retries=5, retry_delay=3.0):
        """
        Wait for all cameras to be ready, with retry logic for cameras that fail to start.
        
        Args:
            max_retries: Maximum number of retry attempts for each camera (default: 5)
            retry_delay: Delay in seconds between retry attempts (default: 3.0)
        """
        for retry_attempt in range(max_retries):
            # Wait for cameras to start
            time.sleep(retry_delay)
            
            # Check which cameras are not ready
            failed_cameras = []
            for serial, camera in self.cameras.items():
                if not camera.is_ready:
                    failed_cameras.append(serial)
            
            if len(failed_cameras) == 0:
                # All cameras are ready
                print(f"✅ All {self.n_cameras} cameras started successfully.")
                return
            
            # Some cameras failed
            if retry_attempt < max_retries - 1:
                print(f"⚠️  Retry {retry_attempt + 1}/{max_retries}: {len(failed_cameras)} camera(s) failed to start: {failed_cameras}")
                print(f"   Stopping and recreating failed cameras...")
                
                # Stop and recreate failed cameras sequentially with delay
                for idx, serial in enumerate(failed_cameras):
                    old_camera = self.cameras[serial]
                    # Stop the old camera process
                    old_camera.stop(wait=True)
                    
                    # Small delay before recreating
                    time.sleep(0.5)
                    
                    # Create new camera instance using stored parameters
                    params = self.camera_params[serial].copy()
                    new_camera = SingleRealsense(**params)
                    self.cameras[serial] = new_camera
                    
                    # Start the new camera
                    put_start_time = time.time()
                    new_camera.start(wait=False, put_start_time=put_start_time)
                    
                    # Stagger the startup
                    if idx < len(failed_cameras) - 1:
                        time.sleep(0.5)
            else:
                # Final attempt failed
                print(f"❌ ERROR: {len(failed_cameras)} camera(s) failed to start after {max_retries} attempts: {failed_cameras}")
                # Stop all cameras for cleanup
                self.stop(wait=True)
                raise RuntimeError(f"Failed to start cameras: {failed_cameras}")
        
        # All cameras should be ready now
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out
    
    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])

    def set_depth_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_option(option, value[i])
    
    def set_depth_preset(self, preset):
        n_camera = len(self.cameras)
        preset = repeat_to_list(preset, n_camera, str)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_preset(preset[i])

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_depth_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_depth_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)
    
    def get_intrinsics(self):
        return np.array([c.get_intrinsics() for c in self.cameras.values()])
    
    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])
    
    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

    def calibrate_extrinsics(self, visualize=True, board_size=(6, 9), squareLength=0.03, markerLength=0.022):
        for camera in self.cameras.values():
            camera.calibrate_extrinsics(visualize=visualize, board_size=board_size, squareLength=squareLength, markerLength=markerLength)

def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x

def main():
    """
    Main function to visualize multiple RealSense cameras using MultiCameraVisualizer.
    Press Ctrl+C to exit.
    """
    from gendp.real_world.multi_camera_visualizer import MultiCameraVisualizer
    from gendp.common.cv2_util import optimal_row_cols
    
    # Configuration
    resolution = (640, 480)
    capture_fps = 30
    
    # Get connected camera serial numbers
    serial_numbers = SingleRealsense.get_connected_devices_serial()
    print(f"Found {len(serial_numbers)} camera(s): {serial_numbers}")
    
    if len(serial_numbers) == 0:
        print("No RealSense cameras detected!")
        return
    
    # Calculate optimal row/col layout
    rw, rh, col, row = optimal_row_cols(
        n_cameras=len(serial_numbers),
        in_wh_ratio=resolution[0] / resolution[1],
        max_resolution=(1920, 1080)
    )
    
    print(f"Using {row}x{col} layout with individual camera resolution {rw}x{rh}")
    
    # Create MultiRealsense instance
    with MultiRealsense(
        serial_numbers=serial_numbers,
        resolution=resolution,
        capture_fps=capture_fps,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        verbose=True
    ) as realsense:
        
        # Create visualizer
        multi_cam_vis = MultiCameraVisualizer(
            realsense=realsense,
            row=row,
            col=col,
            rgb_to_bgr=True  # RealSense outputs RGB, OpenCV expects BGR
        )
        
        print("Starting camera visualization...")
        print("Press Ctrl+C to exit")
        
        multi_cam_vis.start(wait=False)
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping visualization...")
        finally:
            multi_cam_vis.stop(wait=True)
            print("Visualization stopped.")

if __name__ == '__main__':
    main()
