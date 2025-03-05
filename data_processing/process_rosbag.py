#!/usr/bin/env python3
import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge
import os
import glob
import argparse
import yaml
from tqdm import tqdm
import transforms3d
# import pickle
# import tf
import rospy
# from tf.transformations import quaternion_matrix, translation_from_matrix, quaternion_from_matrix

from gendp.common.data_utils import save_dict_to_hdf5
from tf_bag import BagTfTransformer
from utils import get_extrinsic, combine_image_arrays_to_video_2x3


def extract_data_from_rosbag(bag_path, output_dir, topics=None):
    """
    Extract data from a rosbag file into a dictionary.
    
    Args:
        bag_path (str): Path to the rosbag file.
        output_dir (str): Directory to save extracted data.
        topics (list, optional): List of topics to extract. If None, extract all topics.
    
    Returns:
        dict: Data dictionary containing extracted information.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CvBridge for image conversion
    bridge = CvBridge()

    # Open the bag file
    bag = rosbag.Bag(bag_path)
    
    # Get information about the bag
    info_dict = yaml.load(bag._get_yaml_info(), Loader=yaml.SafeLoader)
    duration = info_dict['duration']
    start_time = info_dict['start']
    
    # Get topic information
    topic_info = {}
    for topic_dict in info_dict['topics']:
        topic_info[topic_dict['topic']] = {
            'type': topic_dict['type'],
            'msg_count': topic_dict['messages']
        }
    
    print(f"Bag duration: {duration:.2f} seconds")
    print(f"Available topics: {list(topic_info.keys())}")

    # Ensure TF topics are included
    tf_topics = ['/tf', '/tf_static']
    for tf_topic in tf_topics:
        if tf_topic in topic_info and (topics is None or tf_topic not in topics):
            if topics is None:
                topics = list(topic_info.keys())
            else:
                topics.append(tf_topic)
    
    # Filter topics if specified
    if topics:
        available_topics = set(topic_info.keys())
        topics = [t for t in topics if t in available_topics]
        print(f"Extracting {len(topics)} topics: {topics}")
    else:
        topics = list(topic_info.keys())
        print(f"Extracting all {len(topics)} topics")
    
    # Initialize data dictionary
    data = {
        'meta': {
            'bag_path': bag_path,
            'duration': duration,
            'start_time': start_time,
            'topics': topic_info
        },
        'timestamps': [],
        'wrist_rgb': [],
        'wrist_depth': [],
        'wrist_info': [],
        'fixed_rgb': [],
        'fixed_depth': [],
        'fixed_info': [],
        'tactile_left': [],
        'tactile_right': [],
        'joint_states': [],
        'gripper_state': [],
        'force_torque': [],
        'transforms': [],
        # 'ee_vel': [],
    }
    
    # Define topic mappings (update these to match your specific topics)
    topic_mappings = {
        'wrist_rgb': '/wrist_rs/color/image_raw/compressed',
        'wrist_info': '/wrist_rs/color/camera_info',
        'wrist_depth': '/wrist_rs/aligned_depth_to_color/image_raw/compressedDepth',
        'fixed_rgb': '/fixed_rs/color/image_raw/compressed',
        'fixed_info': '/fixed_rs/color/camera_info',
        'fixed_depth': '/fixed_rs/aligned_depth_to_color/image_raw/compressedDepth',
        'tactile_left': '/gsmini_rawimg_0',
        'tactile_right': '/gsmini_rawimg_1',
        'joint_states': '/joint_states',
        'gripper_state': '/Robotiq2FGripperRobotInput',
        'force_torque': '/robotiq_ft_sensor'
    }

    tf_frames = [
        'wrist_rs_color_optical_frame',
        'fixed_rs_color_optical_frame',
        'right_inner_finger',
        'left_inner_finger',
        'panda_EE',
    ]

    reference_frame = 'base'
    
    # Create reverse mapping
    data_key_for_topic = {}
    for key, topic in topic_mappings.items():
        data_key_for_topic[topic] = key
    
    # Add TF topics to data key mapping
    for tf_topic in tf_topics:
        if tf_topic in topic_info:
            data_key_for_topic[tf_topic] = 'tf'
    
    # First pass: process TF messages to build transform buffer
    print("First pass: processing TF messages...")
    bag_transformer = BagTfTransformer(bag)
    # print(bag_transformer.getTransformGraphInfo())
    # tf_t = tf.Transformer(True, rospy.Duration(3600.0))
    
    # Second pass: collect all timestamps for sensor data
    print("Second pass: collecting timestamps for sensor data...")
    all_timestamps = {}
    
    for topic, msg, t in tqdm(bag.read_messages(topics=[t for t in topics if t not in tf_topics])):
        timestamp = t.to_sec()
        if topic in data_key_for_topic:
            data_key = data_key_for_topic[topic]
            if data_key not in all_timestamps:
                all_timestamps[data_key] = []
            all_timestamps[data_key].append(timestamp)
    
    # Find common timestamps (optional, remove if you want all data points)
    # This will find the closest timestamps across all data types
    if all(len(ts) > 0 for ts in all_timestamps.values()):
        print("Finding synchronized timestamps...")
        # Use the topic with fewest messages as reference
        ref_key = min(all_timestamps.keys(), key=lambda k: len(all_timestamps[k]))
        ref_timestamps = np.array(all_timestamps[ref_key])
        
        synchronized_timestamps = []
        for ref_ts in ref_timestamps:
            closest_ts = {}
            closest_ts[ref_key] = ref_ts
            
            for key, timestamps in all_timestamps.items():
                if key == ref_key:
                    continue
                timestamps_array = np.array(timestamps)
                idx = np.argmin(np.abs(timestamps_array - ref_ts))
                closest_ts[key] = timestamps_array[idx]
            
            # Only include if all timestamps are within a threshold (e.g., 0.1 seconds)
            if max(abs(ts - ref_ts) for ts in closest_ts.values()) < 0.1:
                synchronized_timestamps.append(closest_ts)
        
        print(f"Found {len(synchronized_timestamps)} synchronized data points")
    else:
        print("Some topics have no messages, skipping synchronization")
        synchronized_timestamps = []
    
    # Third pass: extract data
    print("Third pass: extracting data...")
    
    # Initialize empty data lists
    for key in topic_mappings.keys():
        data[key] = []
    
    # Process synchronized timestamps if available
    if synchronized_timestamps:
        # Create lookup tables for each topic
        topic_data = {key: {} for key in topic_mappings.keys()}
        
        for topic, msg, t in tqdm(bag.read_messages(topics=[t for t in topics if t not in tf_topics])):
            timestamp = t.to_sec()
            if topic in data_key_for_topic:
                data_key = data_key_for_topic[topic]
                topic_data[data_key][timestamp] = msg
        
        # Extract synchronized data
        for i, ts_dict in enumerate(synchronized_timestamps[5:-5]):
            timestamp = list(ts_dict.values())[0]  # Use reference timestamp
            data['timestamps'].append(timestamp)
            
            # Extract sensor data
            for key, topic in topic_mappings.items():
                ts = ts_dict.get(key)
                if ts and ts in topic_data[key]:
                    msg = topic_data[key][ts]
                    data[key].append(extract_message_data(msg, key, bridge))
                else:
                    data[key].append(None)
            
            # Extract transform data for camera frames at this timestamp
            transform_data = {}
            for frame in tf_frames:
                # transform = tf_t.lookupTransform(frame, reference_frame, rospy.Time(secs=timestamp))
                try:
                    transform = bag_transformer.lookupTransform(reference_frame, frame, rospy.Time(secs=timestamp))
                except:
                    transform = ([0, 0, 0], [0, 0, 0, 1])
                if transform:
                    # pos, quat = transform
                    # rpy = transforms3d.euler.quat2euler(quat)
                    transform_data[frame] = transform
            
            # twist = bag_transformer.lookupTwist('panda_EE', reference_frame, rospy.Time(secs=timestamp))
            # transform = lookup_transform(tf_transforms, static_transforms, reference_frame, frame, timestamp)
            # if twist:
            #     twist = np.concatenate(twist)
            # data['ee_vel'].append(twist)

            data['transforms'].append(transform_data)
    
    # Close the bag
    bag.close()
    
    # Save data dictionary
    # output_path = os.path.join(output_dir, os.path.basename(bag_path).replace('.bag', '.pkl'))
    # with open(output_path, 'wb') as f:
    #     pickle.dump(data, f)
    
    # print(f"Data saved to {output_path}")
    
    return data

def extract_message_data(msg, data_type, bridge):
    """
    Extract data from a ROS message based on data type.
    
    Args:
        msg: ROS message.
        data_type (str): Type of data ('rgb', 'depth', 'tactile', etc.).
        bridge: CvBridge instance.
    
    Returns:
        Extracted data in appropriate format.
    """
    if 'rgb' in data_type:
        try:
            # CompressedImage message from sensor_msgs/CompressedImage
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Convert from BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return cv_image
        except Exception as e:
            print(f"Error converting RGB image: {e}")
            return None

    elif 'tactile' in data_type:
        try:
            # Convert image message to numpy array
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            return cv_image
        except Exception as e:
            print(f"Error converting {data_type} image: {e}")
            return None
    
    elif 'depth' in data_type:
        try:
            # Handle compressed depth image
            depth_header_size = 12
            raw_data = msg.data[depth_header_size:]
            np_arr = np.frombuffer(raw_data, np.uint8)
            
            # For PNG format (commonly used for depth images)
            # if msg.format == 'png':
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            return cv_image
            # For other formats like 'jpeg' (less common for depth)
            # else:
            #     cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            #     return cv_image
            # else:
            #     # Regular Image message
            #     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #     return cv_image
        except Exception as e:
            print(f"Error converting depth image: {e}")
            return None
    
    elif 'info' in data_type:
        # Extract camera intrinsic
        return np.asarray(msg.K).reshape((3, 3))

    elif data_type == 'joint_states':
        # Extract joint positions, velocities, and efforts
        return {
            'name': msg.name,
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'effort': list(msg.effort)
        }
    
    elif data_type == 'gripper_state':
        return {
            'goal_pos': msg.gPR,
            'curr_pos': msg.gPO,
            'force': msg.gCU,
            # 'is_closed': msg.closed
        }
    
    elif data_type == 'force_torque':
        # Extract force/torque readings
        # This is a placeholder, adjust based on your message structure
        if hasattr(msg, 'wrench'):
            return [msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                    msg.wrench.torque.x,
                    msg.wrench.torque.y,
                    msg.wrench.torque.z]
        else:
            # Alternative structure for force/torque messages
            return [msg.Fx,
                    msg.Fy,
                    msg.Fz,
                    msg.Mx,
                    msg.My,
                    msg.Mz]
    else:
        # Default: return message as is
        return msg

def main(bag_path, output_dir, episode_idx):
    data = extract_data_from_rosbag(bag_path, output_dir)

    attr_dict = {
        'sim': False,
    }
    config_dict = {
        'observations':
            {
                'images': {},
                'tactile': {}
            }
    }
    for cam in ['wrist', 'fixed']:
        color_save_kwargs = {
            'chunks': (1, 480, 640, 3), # (1, 480, 640, 3)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint8',
        }
        depth_save_kwargs = {
            'chunks': (1, 480, 640), # (1, 480, 640)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint16',
        }
        config_dict['observations']['images'][f'{cam}_color'] = color_save_kwargs
        config_dict['observations']['images'][f'{cam}_depth'] = depth_save_kwargs
    for tactile in ['tactile_left', 'tactile_right']:
        tactile_save_kwargs = {
            'chunks': (1, 240, 320, 3), # (1, 480, 640, 3)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint8',
        }
        config_dict['observations']['tactile'][f'{tactile}'] = tactile_save_kwargs


    dataset_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
    # dataset_path = os.path.join(output_dir, 'episode_0.hdf5')

    data_dict = {
        # 'meta': data['meta'],
        'timestamps': np.asarray(data['timestamps']),
        'observations': 
            {'joint_pos': [],
             'joint_vel': [],
             'full_joint_pos': [], # this is to compute FK
            #  'robot_base_pose_in_world': [],
             'ee_pos': [],
            #  'ee_vel': np.asarray(data['ee_vel']),
             'left_finger_pos': {},
             'right_finger_pos': {},
             'force_torque': np.asarray(data['force_torque']),
             'images': {
                 'wrist_color': np.asarray(data['wrist_rgb'], dtype=np.uint8),
                 'wrist_depth': np.asarray(data['wrist_depth'], dtype=np.uint16),
                 'wrist_intrinsic': np.asarray(data['wrist_info']),
                 'fixed_color': np.asarray(data['fixed_rgb'], dtype=np.uint8),
                 'fixed_depth': np.asarray(data['fixed_depth'], dtype=np.uint16),
                 'fixed_intrinsic': np.asarray(data['fixed_info']),
             },
             'tactile': {
                 'tactile_left': np.asarray(data['tactile_left'], dtype=np.uint8),
                 'tactile_right': np.asarray(data['tactile_right'], dtype=np.uint8)
             },
            },
        'joint_action': [],
        'cartesian_action': [],
    }

    joint_pos = [(joint_state['position'][6:] + [gripper_state['curr_pos'], gripper_state['curr_pos']])
                 for joint_state, gripper_state in zip(data['joint_states'], data['gripper_state'])]
    joint_vel = [(joint_state['velocity'][6:] + [0.0, 0.0])
                 for joint_state in data['joint_states']]
    ee_pos = [np.concatenate([transforms['panda_EE'][0], transforms3d.euler.quat2euler(transforms['panda_EE'][1]), [gripper_state['curr_pos']]])
              for transforms, gripper_state in zip(data['transforms'], data['gripper_state'])]
    wrist_extrinsic = [get_extrinsic(*transforms['wrist_rs_color_optical_frame']) for transforms in data['transforms']]
    fixed_extrinsic = [get_extrinsic(*transforms['fixed_rs_color_optical_frame']) for transforms in data['transforms']]
    left_finger_pos = [np.concatenate(transforms['left_inner_finger']) for transforms in data['transforms']]
    right_finger_pos = [np.concatenate(transforms['right_inner_finger']) for transforms in data['transforms']]

    joint_pos = np.asarray(joint_pos)
    joint_vel = np.asarray(joint_vel)
    ee_pos = np.asarray(ee_pos)
    wrist_extrinsic = np.asarray(wrist_extrinsic)
    fixed_extrinsic = np.asarray(fixed_extrinsic)
    left_finger_pos = np.asarray(left_finger_pos)
    right_finger_pos = np.asarray(right_finger_pos)

    data_dict['observations']['joint_pos'] = joint_pos
    data_dict['observations']['full_joint_pos'] = joint_pos
    data_dict['observations']['joint_vel'] = joint_vel
    data_dict['observations']['ee_pos'] = ee_pos
    data_dict['observations']['left_finger_pos'] = left_finger_pos
    data_dict['observations']['right_finger_pos'] = right_finger_pos
    data_dict['observations']['images']['wrist_extrinsic'] = wrist_extrinsic
    data_dict['observations']['images']['fixed_extrinsic'] = fixed_extrinsic
    data_dict['joint_action'] = np.concatenate([joint_pos[1:], joint_pos[-1:]])
    data_dict['cartesian_action'] = np.concatenate([ee_pos[1:], ee_pos[-1:]])

    # combine_image_arrays_to_video_2x3(
    #     data_dict['observations']['images']['wrist_depth'],
    #     data_dict['observations']['images']['fixed_depth'],
    #     data_dict['observations']['images']['wrist_color'],
    #     data_dict['observations']['images']['fixed_color'],
    #     data_dict['observations']['tactile']['tactile_left'],
    #     data_dict['observations']['tactile']['tactile_right'])

    save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract data from a ROS bag file')

    # parser.add_argument('bag_path', type=str, help='Path to the rosbag file')
    # parser.add_argument('episode_idx', help='random seed for the episode')
    parser.add_argument('--demo_name', type=str, default='bosch_hex_long',
                        help='name of the demonstration')
    parser.add_argument('--input_dir', type=str, default='../rosbags/', 
                        help='Directory to save extracted data')
    parser.add_argument('--output_dir', type=str, default='./data/rosbag/', 
                        help='Directory to save extracted data')
    args = parser.parse_args()
    # main(args.bag_path, args.output_dir, args.episode_idx)

    ros_bags = glob.glob(f"{args.input_dir}{args.demo_name}/*.bag")
    output_dir = f"{args.output_dir}{args.demo_name}/"
    for i, bag_path in enumerate(ros_bags[1:]):
        print(f'Processing episode {i}')
        main(bag_path, output_dir, i)


