"""
Web-based demo for real Franka robot with real-time visualization.
This version uses a web interface to avoid OpenCV display issues.

Usage:
python demo_real_franka_web.py -o <demo_save_dir> --robot_ip <ip_of_franka>

Then open your browser to: http://localhost:8000

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Web interface buttons or keyboard shortcuts:
- "C" or "c": Start recording
- "S" or "s": Stop recording  
- "Q" or "q": Exit program
- "Backspace": Delete the previously recorded episode
- "G" or "g": Close gripper
- "O" or "o": Open gripper
- "Space": Save episode for current stage and start next stage
"""

import os
import time
import threading
import queue
import base64
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import eventlet

from gendp.real_world.real_env_franka_gripper import RealEnvFranka, CAMERA_NAMES
from gendp.common.precise_sleep import precise_wait
from gendp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'robot_demo_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables for communication between threads
command_queue = queue.Queue()
robot_state = {
    'running': False,
    'recording': False,
    'episode_id': 0,
    'stage': 0,
    'gripper_pos': 0.08,
    'stop': False
}

@app.route('/')
def index():
    return render_template('robot_control.html')

@app.route('/status')
def status():
    return jsonify(robot_state)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status_update', robot_state)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('command')
def handle_command(data):
    command = data.get('command', '').lower()
    print(f"Received command: {command}")
    command_queue.put(command)
    emit('command_received', {'command': command})

def process_commands(key_counter, env):
    """Process commands from both web interface and keyboard"""
    global robot_state
    
    # Process web commands
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
            if command == 'q':
                robot_state['stop'] = True
                print('Quitting.')
            elif command == 'c':
                env.start_episode(time.time(), curr_outdir=robot_state['output_dir'])
                key_counter.clear()
                robot_state['recording'] = True
                print('Recording!')
            elif command == 's':
                env.end_episode(curr_outdir=robot_state['output_dir'], incr_epi=True)
                key_counter.clear()
                robot_state['recording'] = False
                print('Stopped.')
            elif command == 'space':
                env.end_episode(curr_outdir=robot_state['output_dir'], incr_epi=False)
                robot_state['recording'] = False
                env.start_episode(time.time(), curr_outdir=robot_state['output_dir'])
                robot_state['recording'] = True
            elif command == 'backspace':
                env.drop_episode()
                key_counter.clear()
                robot_state['recording'] = False
                print('Dropped episode.')
            elif command == 'g':
                robot_state['gripper_pos'] = 0.01
                print('Closing gripper.')
            elif command == 'o':
                robot_state['gripper_pos'] = 0.08
                print('Opening gripper.')
        except queue.Empty:
            break
    
    # Process keyboard commands
    press_events = key_counter.get_press_events()
    for key_stroke in press_events:
        if key_stroke == KeyCode(char='q'):
            robot_state['stop'] = True
            print('Quitting.')
        elif key_stroke == KeyCode(char='c'):
            env.start_episode(time.time(), curr_outdir=robot_state['output_dir'])
            key_counter.clear()
            robot_state['recording'] = True
            print('Recording!')
        elif key_stroke == KeyCode(char='s'):
            env.end_episode(curr_outdir=robot_state['output_dir'], incr_epi=True)
            key_counter.clear()
            robot_state['recording'] = False
            print('Stopped.')
        elif key_stroke == Key.space:
            env.end_episode(curr_outdir=robot_state['output_dir'], incr_epi=False)
            robot_state['recording'] = False
            env.start_episode(time.time(), curr_outdir=robot_state['output_dir'])
            robot_state['recording'] = True
        elif key_stroke == Key.backspace:
            env.drop_episode()
            key_counter.clear()
            robot_state['recording'] = False
            print('Dropped episode.')
        elif key_stroke == KeyCode(char='g'):
            robot_state['gripper_pos'] = 0.01
            print('Closing gripper.')
        elif key_stroke == KeyCode(char='o'):
            robot_state['gripper_pos'] = 0.08
            print('Opening gripper.')

def encode_image_to_base64(image):
    """Encode OpenCV image to base64 string for web display"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

def robot_control_loop(output_dir, robot_ip, init_joints, frequency, command_latency):
    """Main robot control loop running in separate thread"""
    global robot_state
    
    dt = 1/frequency    
    robot_state['output_dir'] = output_dir
    try:
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                RealEnvFranka(
                output_dir=output_dir, 
                robot_ip=robot_ip, 
                frequency=frequency,
                n_obs_steps=2,
                obs_float32=False,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager) as env:
                print('Robot ready! Open http://localhost:8000 in your browser')
                robot_state['running'] = True
                socketio.sleep(1.0)
                state = env.get_robot_state()
                t_start = time.monotonic()
                iter_idx = 0
                while not robot_state['stop']:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt
                    obs = env.get_obs()
                    process_commands(key_counter, env)
                    robot_state['stage'] = key_counter[Key.space]
                    robot_state['episode_id'] = env.episode_id
                    rs_left = obs['camera_left_color'][-1,:,:,::-1].copy()
                    rs_wrist = obs['camera_wrist_color'][-1,:,:,::-1].copy()
                    vis_img = np.concatenate([rs_left, rs_wrist], axis=1)
                    vis_img = cv2.resize(vis_img, (960, 360))
                    episode_id = robot_state['episode_id']
                    stage = robot_state['stage']
                    text = f'Episode: {episode_id}, Stage: {stage}'
                    if robot_state['recording']:
                        text += ', Recording!'
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        thickness=2,
                        color=(0, 255, 0) if robot_state['recording'] else (255, 255, 255)
                    )
                    image_data = encode_image_to_base64(vis_img)
                    socketio.emit('image_update', {
                        'image': image_data,
                        'status': robot_state
                    })
                    socketio.sleep(0)
                    joint_pos = obs['full_joint_pos']
                    actions = joint_pos[-1, :8]
                    actions[-1] = robot_state['gripper_pos']
                    env.exec_actions(
                        actions=[actions],
                        timestamps=[t_command_target-time.monotonic()+time.time()])
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                    if iter_idx % 3 == 0:
                        socketio.emit('status_update', robot_state)
                        socketio.sleep(0)
    except Exception as e:
        print(f"Error in robot control loop: {e}")
        robot_state['stop'] = True
    finally:
        robot_state['running'] = False
        print("Robot control loop ended")

@click.command()
@click.option('--output_dir', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri ', default="192.168.1.143", help="Franka's IP address ")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--vis_camera_idx', default=1, type=int, help="Which RealSense camera to visualize.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--port', '-p', default=8000, type=int, help="Web server port.")
def main(output_dir, robot_ip, init_joints, vis_camera_idx, frequency, command_latency, port):
    os.system(f'mkdir -p {output_dir}')
    # Start robot control as a Flask-SocketIO background task
    def start_robot_task():
        socketio.start_background_task(robot_control_loop, output_dir, robot_ip, init_joints, frequency, command_latency)
    start_robot_task()
    print(f"Starting web server on http://localhost:{port}")
    print("Press Ctrl+C to exit")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        robot_state['stop'] = True

if __name__ == '__main__':
    main()
