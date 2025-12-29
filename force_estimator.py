#!/usr/bin/env python3
"""
Force estimation from tactile sensors and vision-based contact points.

This module provides force estimator classes that compute the net wrench
from tactile sensors and estimate contact forces at detected contact points.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Construct the skew-symmetric matrix of a 3D vector.
    
    Args:
        v: (3,) vector [x, y, z]
        
    Returns:
        (3, 3) skew-symmetric matrix [[v]_×]
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def estimate_tool_normals(tool_pcd_np, camera_location=np.array([0, 0, 0]), inward=True):
    """
    Estimates and orients normals for a raw point cloud.
    
    Args:
        tool_pcd_np (np.ndarray): (N, 3) array of tool points.
        camera_location (np.ndarray): The position of the primary camera in gripper frame.
        inward (bool): If True, flips normals to point into the tool.
        
    Returns:
        np.ndarray: (N, 3) oriented normal vectors.
    """
    # 1. Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tool_pcd_np)
    
    # 2. Estimate Normals using local PCA
    # radius: search radius; max_nn: max neighbors to consider
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30)
    )
    
    # 3. Orient normals towards the camera (Standard O3D behavior is outward)
    pcd.orient_normals_towards_camera_location(camera_location)
    
    normals = np.asarray(pcd.normals)
    
    # 4. Flip if inward normals are required for the solver (f_i = w_i * n_i)
    if inward:
        normals = -normals
        
    return normals

def smooth_normals(tool_pcd_np, normals, iterations=2):
    """
    Applies simple Laplacian smoothing to normals to reduce RealSense noise.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tool_pcd_np)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Using Open3D's filter_smooth_laplacian requires a mesh, 
    # for pure PCD we can use a simpler neighbor-averaging approach:
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    new_normals = np.copy(normals)
    
    for _ in range(iterations):
        for i in range(len(tool_pcd_np)):
            [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 10)
            neighbor_normals = normals[idx, :]
            avg_normal = np.mean(neighbor_normals, axis=0)
            new_normals[i] = avg_normal / np.linalg.norm(avg_normal)
        normals = np.copy(new_normals)
        
    return new_normals


class ForceEstimator(ABC):
    """
    Abstract base class for force estimators.
    
    Provides a unified interface for different force estimation methods.
    """
    
    @abstractmethod
    def compute_net_wrench(self,
                          tactile_force_left: np.ndarray,
                          tactile_force_right: np.ndarray,
                          tactile_coord_left: np.ndarray,
                          tactile_coord_right: np.ndarray) -> np.ndarray:
        """
        Calculate the net wrench from tactile sensor data.
        
        Args:
            tactile_force_left: (7, 9, 3) left tactile force field [fx, fy, fz]
            tactile_force_right: (7, 9, 3) right tactile force field [fx, fy, fz]
            tactile_coord_left: (7, 9, 3) left tactile marker coordinates in gripper frame
            tactile_coord_right: (7, 9, 3) right tactile marker coordinates in gripper frame
            
        Returns:
            (6,) wrench vector [Fx, Fy, Fz, Mx, My, Mz]
        """
        pass
    
    @abstractmethod
    def estimate_contact_forces(self,
                                wrench: np.ndarray,
                                contact_points: np.ndarray,
                                **kwargs) -> np.ndarray:
        """
        Estimate forces at contact points given the net wrench.
        
        Args:
            wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            contact_points: (N, 3) array of contact point coordinates
            **kwargs: Additional method-specific arguments
            
        Returns:
            (N, 3) array of estimated forces at each contact point
        """
        pass
    
    def compute_contact_forces_from_tactile(self,
                                           tactile_force_left: np.ndarray,
                                           tactile_force_right: np.ndarray,
                                           tactile_coord_left: np.ndarray,
                                           tactile_coord_right: np.ndarray,
                                           contact_points: np.ndarray,
                                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: compute wrench from tactile, then estimate contact forces.
        
        Args:
            tactile_force_left: (7, 9, 3) left tactile force field
            tactile_force_right: (7, 9, 3) right tactile force field
            tactile_coord_left: (7, 9, 3) left tactile marker coordinates
            tactile_coord_right: (7, 9, 3) right tactile marker coordinates
            contact_points: (N, 3) array of contact point coordinates
            **kwargs: Additional method-specific arguments
            
        Returns:
            Tuple of:
            - wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            - forces: (N, 3) estimated forces at each contact point
        """
        # Compute net wrench from tactile sensors
        wrench = self.compute_net_wrench(
            tactile_force_left,
            tactile_force_right,
            tactile_coord_left,
            tactile_coord_right
        )
        
        # Estimate forces at contact points
        forces = self.estimate_contact_forces(wrench, contact_points, **kwargs)
        
        return wrench, forces


class SimpleForceEstimator(ForceEstimator):
    """
    Estimates external forces acting on a tool held by a parallel gripper.
    
    Uses tactile sensor data (7×9 arrays on left and right grippers) and
    vision-based contact points to compute net wrench and contact forces.
    """
    
    def __init__(self):
        """Initialize the force estimator."""
        pass
    
    def compute_net_wrench(self,
                          tactile_force_left: np.ndarray,
                          tactile_force_right: np.ndarray,
                          tactile_coord_left: np.ndarray,
                          tactile_coord_right: np.ndarray) -> np.ndarray:
        """
        Calculate the net wrench from tactile sensor data.
        
        Args:
            tactile_force_left: (7, 9, 3) left tactile force field [fx, fy, fz]
            tactile_force_right: (7, 9, 3) right tactile force field [fx, fy, fz]
            tactile_coord_left: (7, 9, 3) left tactile marker coordinates in gripper frame
            tactile_coord_right: (7, 9, 3) right tactile marker coordinates in gripper frame
            
        Returns:
            (6,) wrench vector [Fx, Fy, Fz, Mx, My, Mz]
        """
        # Flatten the tactile arrays
        forces_left = tactile_force_left.reshape(-1, 3)  # (63, 3)
        forces_right = tactile_force_right.reshape(-1, 3)  # (63, 3)
        coords_left = tactile_coord_left.reshape(-1, 3)  # (63, 3)
        coords_right = tactile_coord_right.reshape(-1, 3)  # (63, 3)
        
        # Combine left and right
        all_forces = np.vstack([forces_left, forces_right])  # (126, 3)
        all_coords = np.vstack([coords_left, coords_right])  # (126, 3)
        
        # Calculate total force: F_total = sum(f_jk)
        F_total = np.sum(all_forces, axis=0)  # (3,)
        
        # Calculate total moment: M_total = sum(p_jk × f_jk)
        M_total = np.zeros(3)
        for p, f in zip(all_coords, all_forces):
            M_total += np.cross(p, f)
        
        # Return 6D wrench [F, M]
        wrench = np.concatenate([F_total, M_total])
        return wrench
    
    def construct_grasp_matrix(self, contact_points: np.ndarray) -> np.ndarray:
        """
        Construct the grasp matrix A for contact points.
        
        For N contact points, constructs a (6, 3N) matrix where each contact
        contributes a (6, 3) block: [I_3x3; [c_i]_×]
        
        Args:
            contact_points: (N, 3) array of contact point coordinates
            
        Returns:
            (6, 3N) grasp matrix
        """
        N = len(contact_points)
        A = np.zeros((6, 3 * N))
        
        for i, c_i in enumerate(contact_points):
            # Block for contact i
            G_i = np.zeros((6, 3))
            G_i[:3, :] = np.eye(3)  # Identity for force part
            G_i[3:, :] = skew_symmetric(c_i)  # Skew-symmetric for moment part
            
            # Place in grasp matrix
            A[:, 3*i:3*(i+1)] = G_i
        
        return A
    
    def estimate_contact_forces(self,
                                wrench: np.ndarray,
                                contact_points: np.ndarray) -> np.ndarray:
        """
        Estimate forces at contact points given the net wrench.
        
        Args:
            wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            contact_points: (N, 3) array of contact point coordinates
            
        Returns:
            (N, 3) array of estimated forces at each contact point
        """
        N = len(contact_points)
        
        if N == 0:
            return np.zeros((0, 3))
        
        # Construct grasp matrix
        A = self.construct_grasp_matrix(contact_points)  # (6, 3N)
        
        if N == 1:
            # Single contact: overdetermined system G_1 @ f_1 = W
            # Use least squares to solve
            G_1 = A  # (6, 3)
            f_1, residuals, rank, s = np.linalg.lstsq(G_1, wrench, rcond=None)
            return f_1.reshape(1, 3)
        else:
            # Multiple contacts: underdetermined system A @ x = W
            # Use Moore-Penrose pseudo-inverse for minimum-norm solution
            A_pinv = np.linalg.pinv(A)  # (3N, 6)
            x = A_pinv @ wrench  # (3N,)
            
            # Reshape to (N, 3)
            forces = x.reshape(N, 3)
            return forces


class AnalyticalForceEstimator(ForceEstimator):
    """
    Analytical force estimator that solves for force distribution on tool point cloud.
    
    Uses convex optimization (cvxpy) to find scalar force magnitudes at each voxel point
    that best explain the measured tactile wrench, subject to non-penetration constraints.
    """
    
    def __init__(self, lambda_reg: float = 0.01, epsilon: float = 1e-6):
        """
        Initialize the analytical force estimator.
        
        Args:
            lambda_reg: Regularization weight for the objective function (default: 0.01)
            epsilon: Small constant to avoid division by zero (default: 1e-6)
        """
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        
        # Try to import cvxpy
        try:
            import cvxpy as cp
            self.cp = cp
        except ImportError:
            raise ImportError(
                "cvxpy is required for AnalyticalForceEstimator. "
                "Install it with: pip install cvxpy"
            )
    
    def compute_net_wrench(self,
                          tactile_force_left: np.ndarray,
                          tactile_force_right: np.ndarray,
                          tactile_coord_left: np.ndarray,
                          tactile_coord_right: np.ndarray) -> np.ndarray:
        """
        Calculate the net wrench from tactile sensor data.
        
        Args:
            tactile_force_left: (7, 9, 3) left tactile force field [fx, fy, fz]
            tactile_force_right: (7, 9, 3) right tactile force field [fx, fy, fz]
            tactile_coord_left: (7, 9, 3) left tactile marker coordinates in gripper frame
            tactile_coord_right: (7, 9, 3) right tactile marker coordinates in gripper frame
            
        Returns:
            (6,) wrench vector [Fx, Fy, Fz, Mx, My, Mz]
        """
        # Flatten the tactile arrays
        forces_left = tactile_force_left.reshape(-1, 3)  # (63, 3)
        forces_right = tactile_force_right.reshape(-1, 3)  # (63, 3)
        coords_left = tactile_coord_left.reshape(-1, 3)  # (63, 3)
        coords_right = tactile_coord_right.reshape(-1, 3)  # (63, 3)
        
        # Combine left and right
        all_forces = np.vstack([forces_left, forces_right])  # (126, 3)
        all_coords = np.vstack([coords_left, coords_right])  # (126, 3)
        
        # Calculate total force: F_total = sum(f_jk)
        F_total = np.sum(all_forces, axis=0)  # (3,)
        
        # Calculate total moment: M_total = sum(p_jk × f_jk)
        M_total = np.zeros(3)
        for p, f in zip(all_coords, all_forces):
            M_total += np.cross(p, f)
        
        # Return 6D wrench [F, M]
        wrench = np.concatenate([F_total, M_total])
        return wrench
    
    def construct_grasp_matrix(self, 
                              voxel_pcd: np.ndarray, 
                              normals: np.ndarray) -> np.ndarray:
        """
        Construct the grasp matrix G for voxel point cloud.
        
        For N voxel points, constructs a (6, N) matrix where each column i represents
        the 6D wrench contribution of a unit force at point i acting along normal n_i:
        col_i = [n_i; c_i × n_i]
        
        Args:
            voxel_pcd: (N, 3) array of voxel point coordinates (in gripper frame)
            normals: (N, 3) array of inward-pointing surface normals
            
        Returns:
            (6, N) grasp matrix
        """
        N = len(voxel_pcd)
        G = np.zeros((6, N))
        
        for i in range(N):
            c_i = voxel_pcd[i]  # Position
            n_i = normals[i]    # Normal direction
            
            # Force contribution
            G[:3, i] = n_i
            
            # Moment contribution: c_i × n_i
            G[3:, i] = np.cross(c_i, n_i)
        
        return G
    
    def estimate_contact_forces(self,
                                wrench: np.ndarray,
                                contact_points: np.ndarray,
                                normals: Optional[np.ndarray] = None,
                                prob_weights: Optional[np.ndarray] = None,
                                **kwargs) -> np.ndarray:
        """
        Estimate forces at contact points using convex optimization.
        
        Solves:
            min_w ||G w - wrench||_2^2 + λ Σ(w_i^2 / (prob_weights_i + ε))
            subject to: w_i ≥ 0
        
        Args:
            wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            contact_points: (N, 3) array of contact point coordinates
            normals: (N, 3) array of inward-pointing surface normals
                     If None, assumes upward normals [0, 0, 1]
            prob_weights: (N,) array of contact probabilities
                         If None, uses uniform weights of 1.0
            **kwargs: Additional arguments (unused)
            
        Returns:
            (N, 3) array of estimated forces at each contact point (f_i = w_i * n_i)
        """
        N = len(contact_points)
        
        if N == 0:
            return np.zeros((0, 3))
        
        # Default normals to upward direction if not provided
        if normals is None:
            normals = np.tile([0.0, 0.0, 1.0], (N, 1))
        
        # Default prob_weights to uniform if not provided
        if prob_weights is None:
            prob_weights = np.ones(N)
        
        # Construct grasp matrix
        G = self.construct_grasp_matrix(contact_points, normals)  # (6, N)
        
        # Define optimization variable: scalar force magnitudes
        w = self.cp.Variable(N)
        
        # Wrench error term
        wrench_error = self.cp.sum_squares(G @ w - wrench)
        
        # Regularization term: weighted L2 penalty
        # weight_i = 1 / (prob_weights_i + epsilon)
        reg_weights = 1.0 / (prob_weights + self.epsilon)
        regularization = self.cp.sum(self.cp.multiply(reg_weights, self.cp.square(w)))
        
        # Objective: minimize wrench error + regularization
        objective = self.cp.Minimize(wrench_error + self.lambda_reg * regularization)
        
        # Constraints: non-penetration (w_i >= 0)
        constraints = [w >= 0]
        
        # Formulate and solve problem
        problem = self.cp.Problem(objective, constraints)
        
        try:
            # Try OSQP first (fast for QP)
            problem.solve(solver=self.cp.OSQP, verbose=False)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                # Fallback to ECOS
                problem.solve(solver=self.cp.ECOS, verbose=False)
                
        except Exception as e:
            print(f"Warning: Optimization failed with error: {e}")
            # Return zero forces as fallback
            return np.zeros((N, 3))
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Warning: Optimization did not converge (status: {problem.status})")
            return np.zeros((N, 3))
        
        # Extract solution
        w_opt = w.value
        
        if w_opt is None:
            return np.zeros((N, 3))
        
        # Compute force vectors: f_i = w_i * n_i
        forces = w_opt.reshape(-1, 1) * normals  # (N, 3)
        
        return forces
    
    
def create_force_estimator(method: str = 'simple', **kwargs) -> ForceEstimator:
    """
    Factory function to create force estimators.
    
    Args:
        method: Force estimation method ('simple' or 'analytical')
        **kwargs: Method-specific arguments
        
    Returns:
        ForceEstimator instance
    """
    if method == 'simple':
        return SimpleForceEstimator()
    elif method == 'analytical':
        lambda_reg = kwargs.get('lambda_reg', 0.01)
        epsilon = kwargs.get('epsilon', 1e-6)
        return AnalyticalForceEstimator(lambda_reg=lambda_reg, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown force estimation method: {method}")

