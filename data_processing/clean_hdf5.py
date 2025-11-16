#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗HDF5数据: 智能过滤异常卡顿  (保留正常夹取动作)
"""

import h5py
import numpy as np
import os
import glob
from tqdm import tqdm


def detect_static_segments(ee_pose, gripper_threshold=0.005, pos_threshold=0.001, rot_threshold=0.01, min_static_frames=10):
    """
    检测异常静止片段  (排除正常夹取动作)

    关键逻辑:
    - 如果夹爪在动作  (gripper变化大)   即使EE静止 也保留  (这是正常夹取)
    - 只过滤: EE静止 且 gripper也静止 的片段  (这是异常卡顿)

    Args:
        ee_pose: (T, 7) [x,y,z,rx,ry,rz,gripper]
        gripper_threshold: 夹爪变化阈值  (米)
        pos_threshold: 位置变化阈值  (米)
        rot_threshold: 旋转变化阈值  (弧度)
        min_static_frames: 最小静止帧数  (低于此值不过滤)

    Returns:
        mask: (T,) bool array, True表示保留 False表示过滤
    """
    T = len(ee_pose)
    if T == 0:
        return np.array([], dtype=bool)

    # 计算帧间变化
    pos_delta = np.linalg.norm(np.diff(ee_pose[:, :3], axis=0), axis=1)  # (T-1,)
    rot_delta = np.linalg.norm(np.diff(ee_pose[:, 3:6], axis=0), axis=1)  # (T-1,)
    gripper_delta = np.abs(np.diff(ee_pose[:, 6], axis=0))  # (T-1,)

    # 判断各部分是否静止
    ee_static = (pos_delta < pos_threshold) & (rot_delta < rot_threshold)  # (T-1,)
    gripper_static = gripper_delta < gripper_threshold  # (T-1,)

    # 关键: 只有当EE静止 且 gripper也静止时 才认为是异常卡顿
    is_abnormal_static = ee_static & gripper_static  # (T-1,)

    # 第一帧总是保留
    is_abnormal_static = np.concatenate([[False], is_abnormal_static])  # (T,)

    # 找连续异常静止片段
    mask = np.ones(T, dtype=bool)

    i = 0
    while i < T:
        if is_abnormal_static[i]:
            # 找到静止片段的结束
            j = i
            while j < T and is_abnormal_static[j]:
                j += 1

            static_length = j - i

            # 如果异常静止时间过长 标记为需要过滤
            if static_length >= min_static_frames:
                # 检查这段时间内gripper是否真的完全没变化
                gripper_range = np.ptp(ee_pose[i:j, 6])  # peak-to-peak

                if gripper_range < gripper_threshold * 2:  # 确认gripper没有显著变化
                    mask[i:j] = False
                    print(
                        f"    Found abnormal static: frames {i}-{j-1} "
                        f"(length={static_length}, gripper_range={gripper_range:.4f})"
                    )
                else:
                    print(
                        f"    Kept gripper action: frames {i}-{j-1} " f"(EE static but gripper moving, range={gripper_range:.4f})"
                    )

            i = j
        else:
            i += 1

    return mask


def analyze_episode_motion(ee_pose, window_size=5):
    """
    分析episode的运动模式 用于更智能的过滤

    Returns:
        motion_score: (T,) 运动分数 越高越动态
    """
    T = len(ee_pose)
    motion_score = np.zeros(T)

    for i in range(T):
        start = max(0, i - window_size)
        end = min(T, i + window_size + 1)

        # 计算窗口内的变化
        window = ee_pose[start:end]
        pos_var = np.var(window[:, :3], axis=0).sum()
        rot_var = np.var(window[:, 3:6], axis=0).sum()
        gripper_var = np.var(window[:, 6])

        # 综合运动分数
        motion_score[i] = pos_var + rot_var + gripper_var * 10  # gripper权重更高

    return motion_score


def detect_static_segments_advanced(
    ee_pose,
    gripper_threshold=0.0005,
    pos_threshold=0.001,
    rot_threshold=0.01,
    min_static_frames=10,
    motion_score_threshold=0.0001,
):
    """
    改进的检测算法: 考虑局部运动模式
    """
    T = len(ee_pose)
    if T == 0:
        return np.array([], dtype=bool)

    # 计算运动分数
    motion_score = analyze_episode_motion(ee_pose)

    # 基础判断
    pos_delta = np.linalg.norm(np.diff(ee_pose[:, :3], axis=0), axis=1)
    rot_delta = np.linalg.norm(np.diff(ee_pose[:, 3:6], axis=0), axis=1)
    gripper_delta = np.abs(np.diff(ee_pose[:, 6], axis=0))

    ee_static = (pos_delta < pos_threshold) & (rot_delta < rot_threshold)
    gripper_static = gripper_delta < gripper_threshold

    # 结合运动分数判断
    is_static = np.concatenate([[False], ee_static & gripper_static])
    low_motion = motion_score < motion_score_threshold

    is_abnormal = is_static & low_motion

    # 找连续片段
    mask = np.ones(T, dtype=bool)

    i = 0
    while i < T:
        if is_abnormal[i]:
            j = i
            while j < T and is_abnormal[j]:
                j += 1

            static_length = j - i

            if static_length >= min_static_frames:
                # 额外检查: 前后是否有运动
                has_motion_before = (i > 0) and (motion_score[i - 1] > motion_score_threshold * 2)
                has_motion_after = (j < T) and (motion_score[j] > motion_score_threshold * 2)

                # 如果前后都有明显运动 这段静止可能是必要的  (如稳定等待)
                if has_motion_before and has_motion_after and static_length < min_static_frames * 2:
                    print(f"    Kept potential transition: frames {i}-{j-1} (length={static_length})")
                else:
                    mask[i:j] = False
                    avg_motion = motion_score[i:j].mean()
                    print(
                        f"    Filtered abnormal static: frames {i}-{j-1} "
                        f"(length={static_length}, avg_motion={avg_motion:.6f})"
                    )

            i = j
        else:
            i += 1

    return mask


def filter_hdf5_file(
    input_path,
    output_path,
    gripper_threshold=0.0005,
    pos_threshold=0.001,
    rot_threshold=0.01,
    min_static_frames=10,
    min_episode_length=20,
    use_advanced=True,
    fps=10.0,
):
    """过滤单个HDF5文件"""
    try:
        with h5py.File(input_path, "r") as f_in:
            ee_pose = f_in["observations"]["ee_pose"][:]
            original_length = len(ee_pose)

            # 检测并生成mask
            if use_advanced:
                mask = detect_static_segments_advanced(
                    ee_pose, gripper_threshold, pos_threshold, rot_threshold, min_static_frames
                )
            else:
                mask = detect_static_segments(ee_pose, gripper_threshold, pos_threshold, rot_threshold, min_static_frames)

            filtered_length = np.sum(mask)

            if filtered_length < min_episode_length:
                print(f"    [SKIP] Episode too short: {filtered_length} < {min_episode_length}")
                return False, original_length, 0

            # 创建输出文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with h5py.File(output_path, "w") as f_out:
                # 1. 重新生成连续的timestamp
                new_timestamp = np.arange(filtered_length, dtype=np.float32) / fps
                f_out.create_dataset("timestamp", data=new_timestamp)

                # 2. 过滤其他top-level数据
                for key in ["stage", "joint_action"]:
                    if key in f_in:
                        data = f_in[key][:]
                        f_out.create_dataset(key, data=data[mask])

                # 3. 过滤observations组
                obs_group = f_out.create_group("observations")
                f_in_obs = f_in["observations"]

                for key in f_in_obs.keys():
                    if key == "images":
                        img_group = obs_group.create_group("images")
                        for img_key in f_in_obs["images"].keys():
                            img_data = f_in_obs["images"][img_key][:]
                            img_group.create_dataset(img_key, data=img_data[mask])
                    elif key == "robot_base_pose_in_world":
                        data = f_in_obs[key][:]
                        obs_group.create_dataset(key, data=data[mask])
                    else:
                        data = f_in_obs[key][:]
                        obs_group.create_dataset(key, data=data[mask])

            print(f"    [FILTERED] {original_length} -> {filtered_length} frames")
            return True, original_length, filtered_length

    except Exception as e:
        print(f"    [ERROR] {str(e)}")
        return False, 0, 0


def clean_hdf5_dataset(
    input_root,
    output_root,
    gripper_threshold=0.0005,
    pos_threshold=0.001,
    rot_threshold=0.01,
    min_static_frames=10,
    min_episode_length=20,
    use_advanced=True,
):
    """批量清洗HDF5数据集"""
    patterns = ["**/*.h5", "**/*.hdf5"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_root, p), recursive=True))
    files = sorted(files)

    print(f"\n{'='*80}")
    print(f"HDF5 Data Cleaning (Smart Gripper-Aware)")
    print(f"{'='*80}")
    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Found {len(files)} HDF5 files")
    print(f"\nFiltering parameters:")
    print(f"  Gripper threshold:      {gripper_threshold} m")
    print(f"  Position threshold:     {pos_threshold} m")
    print(f"  Rotation threshold:     {rot_threshold} rad")
    print(f"  Min static frames:      {min_static_frames}")
    print(f"  Min episode length:     {min_episode_length}")
    print(f"  Advanced algorithm:     {use_advanced}")
    print(f"{'='*80}\n")

    stats = {
        "total": len(files),
        "success": 0,
        "skipped": 0,
        "error": 0,
        "original_frames": 0,
        "filtered_frames": 0,
    }

    for i, input_path in enumerate(tqdm(files, desc="Processing")):
        rel_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\n[{i+1}/{len(files)}] {os.path.basename(input_path)}")

        success, orig_len, filt_len = filter_hdf5_file(
            input_path,
            output_path,
            gripper_threshold=gripper_threshold,
            pos_threshold=pos_threshold,
            rot_threshold=rot_threshold,
            min_static_frames=min_static_frames,
            min_episode_length=min_episode_length,
            use_advanced=use_advanced,
        )

        if success:
            if filt_len > 0:
                stats["success"] += 1
                stats["original_frames"] += orig_len
                stats["filtered_frames"] += filt_len
            else:
                stats["skipped"] += 1
        else:
            stats["error"] += 1

    print(f"\n{'='*80}")
    print("Cleaning Summary")
    print(f"{'='*80}")
    print(f"Total files:          {stats['total']}")
    print(f"Successfully cleaned: {stats['success']}")
    print(f"Skipped (too short):  {stats['skipped']}")
    print(f"Errors:               {stats['error']}")
    print(f"\nFrame statistics:")
    print(f"  Original frames:  {stats['original_frames']}")
    print(f"  Filtered frames:  {stats['filtered_frames']}")
    if stats["original_frames"] > 0:
        kept_ratio = stats["filtered_frames"] / stats["original_frames"] * 100
        removed = stats["original_frames"] - stats["filtered_frames"]
        print(f"  Removed frames:   {removed}")
        print(f"  Kept ratio:       {kept_ratio:.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # TODO: here
    # MISSIONS = ["empty_book", "empty_empty", "empty_line", "tea_book", "tea_empty", "tea_line"]
    MISSIONS = ["empty_book", "empty_empty", "empty_line", "tea_empty"]

    # 全局清洗参数
    GRIPPER_THRESHOLD = 0.0005
    POS_THRESHOLD = 0.001
    ROT_THRESHOLD = 0.01
    MIN_STATIC_FRAMES = 1
    MIN_EPISODE_LENGTH = 20
    USE_ADVANCED = True

    # def make_paths(mission: str):
    #     input_root = f"/mnt/bn/mllm-all-datasets/anno/datasets/SafeTrajectory/hdf5/align/general-1105raw/{mission}"
    #     output_root = f"/mnt/bn/mllm-all-datasets/anno/datasets/SafeTrajectory/hdf5/align/general-1105clean/{mission}"
    #     return input_root, output_root

    # for mission in MISSIONS:
    #     input_root, output_root = make_paths(mission)
    #     print(f"[INFO] Cleaning mission={mission}")
    #     print(f"       input = {input_root}")
    #     print(f"       output= {output_root}")
    #     clean_hdf5_dataset(
    #         input_root=input_root,
    #         output_root=output_root,
    #         gripper_threshold=GRIPPER_THRESHOLD,
    #         pos_threshold=POS_THRESHOLD,
    #         rot_threshold=ROT_THRESHOLD,
    #         min_static_frames=MIN_STATIC_FRAMES,
    #         min_episode_length=MIN_EPISODE_LENGTH,
    #         use_advanced=USE_ADVANCED,
    #     )

    input_root = "/home/showlab/gendp/data/pick_n_place_mug_raw/"
    output_root = "/home/showlab/gendp/data/pick_n_place_mug_clean/"

    print(f"       input = {input_root}")
    print(f"       output= {output_root}")
    clean_hdf5_dataset(
        input_root=input_root,
        output_root=output_root,
        gripper_threshold=GRIPPER_THRESHOLD,
        pos_threshold=POS_THRESHOLD,
        rot_threshold=ROT_THRESHOLD,
        min_static_frames=MIN_STATIC_FRAMES,
        min_episode_length=MIN_EPISODE_LENGTH,
        use_advanced=USE_ADVANCED,
    )
    print("[DONE] All missions processed.")