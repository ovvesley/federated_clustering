# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range, load_data
from nvflare.app_common.app_constant import AppConstants

from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus
from dfa_lib_python.dependency import Dependency

from time import perf_counter
import datetime

dataflow_tag = "nvidiaflare-df"


def ensure_serializable(obj):
    """Recursively convert numpy types to native Python types for serialization."""
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class DBSCANLearner(Learner):
    def __init__(
        self,
        data_path: str,
        client_id: int = 0,
        hash_trial: str = "unknown_trial",
        valid_frac: float = 1,
        eps: float = 0.1,  # Maximum distance between samples to be considered neighbors
        min_samples: int = 5,  # Minimum samples in neighborhood for core point
        random_state: int = None,
        max_core_points: int = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.valid_frac = valid_frac
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.client_id = int(client_id)
        self.train_data = None
        self.valid_data = None
        # number of training samples (used by NVFlare SKLearnExecutor)
        self.n_samples = None
        self.hash_trial = hash_trial
        self.max_core_points = int(max_core_points) if max_core_points else 0

    def _sanitize_features(self, x: np.ndarray, fl_ctx: FLContext, stage: str) -> np.ndarray:
        x_array = np.asarray(x, dtype=np.float32)
        if x_array.size == 0:
            return x_array

        finite_mask = np.isfinite(x_array)
        if finite_mask.all():
            return x_array

        sanitized = x_array.copy()
        invalid_count = int(sanitized.size - np.count_nonzero(finite_mask))
        sanitized[~finite_mask] = np.nan

        col_means = np.nanmean(sanitized, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
        nan_rows, nan_cols = np.where(np.isnan(sanitized))
        sanitized[nan_rows, nan_cols] = col_means[nan_cols]

        self.log_info(
            fl_ctx,
            f"Sanitized {invalid_count} non-finite values in {stage} data",
        )
        return sanitized

    def _allocate_cluster_budgets(self, cluster_sizes: np.ndarray, budget: int) -> np.ndarray:
        n_clusters = len(cluster_sizes)
        allocation = np.zeros(n_clusters, dtype=np.int32)

        if budget <= 0 or n_clusters == 0:
            return allocation

        sorted_indices = np.argsort(-cluster_sizes)
        if budget < n_clusters:
            allocation[sorted_indices[:budget]] = 1
            return allocation

        allocation = np.minimum(cluster_sizes, 1).astype(np.int32)
        remaining_budget = budget - int(allocation.sum())

        desired_min = np.minimum(cluster_sizes, 2).astype(np.int32)
        for cluster_index in sorted_indices:
            if remaining_budget <= 0:
                break
            extra_needed = desired_min[cluster_index] - allocation[cluster_index]
            if extra_needed <= 0:
                continue
            extra = min(int(extra_needed), remaining_budget)
            allocation[cluster_index] += extra
            remaining_budget -= extra

        remaining_capacity = np.maximum(cluster_sizes - allocation, 0)
        if remaining_budget > 0 and remaining_capacity.sum() > 0:
            proportional = remaining_budget * (remaining_capacity / remaining_capacity.sum())
            extra_allocation = np.minimum(
                remaining_capacity,
                np.floor(proportional).astype(np.int32),
            )
            allocation += extra_allocation
            remaining_budget -= int(extra_allocation.sum())
            remaining_capacity = np.maximum(cluster_sizes - allocation, 0)

        if remaining_budget > 0:
            ranked_indices = np.argsort(-remaining_capacity)
            for cluster_index in ranked_indices:
                if remaining_budget <= 0:
                    break
                if remaining_capacity[cluster_index] <= 0:
                    continue
                allocation[cluster_index] += 1
                remaining_capacity[cluster_index] -= 1
                remaining_budget -= 1

        return allocation

    def _farthest_point_sample_indices(self, points: np.ndarray, sample_size: int) -> np.ndarray:
        if sample_size >= len(points):
            return np.arange(len(points), dtype=np.int64)
        if sample_size <= 0 or len(points) == 0:
            return np.array([], dtype=np.int64)
        if sample_size == 1:
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            return np.array([int(np.argmax(distances))], dtype=np.int64)

        centroid = np.mean(points, axis=0)
        first_index = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
        selected_indices = [first_index]
        min_distances = np.linalg.norm(points - points[first_index], axis=1)

        while len(selected_indices) < sample_size:
            next_index = int(np.argmax(min_distances))
            if next_index in selected_indices:
                break
            selected_indices.append(next_index)
            next_distances = np.linalg.norm(points - points[next_index], axis=1)
            min_distances = np.minimum(min_distances, next_distances)

        return np.sort(np.asarray(selected_indices, dtype=np.int64))

    def _limit_core_points(
        self, core_points: np.ndarray, core_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.max_core_points <= 0 or len(core_points) <= self.max_core_points:
            return core_points, core_labels

        labels, inverse_indices, counts = np.unique(
            core_labels, return_inverse=True, return_counts=True
        )
        cluster_budget = self._allocate_cluster_budgets(counts, self.max_core_points)
        selected_indices = []

        for cluster_position, cluster_label in enumerate(labels):
            sample_size = int(cluster_budget[cluster_position])
            if sample_size <= 0:
                continue

            cluster_indices = np.flatnonzero(inverse_indices == cluster_position)
            if sample_size >= len(cluster_indices):
                selected_indices.extend(int(index) for index in cluster_indices)
                continue

            cluster_points = core_points[cluster_indices]
            sampled_local_indices = self._farthest_point_sample_indices(cluster_points, sample_size)
            selected_indices.extend(int(cluster_indices[index]) for index in sampled_local_indices)

        selected_indices = np.sort(np.asarray(selected_indices, dtype=np.int64))
        return core_points[selected_indices], core_labels[selected_indices]

    def load_data(self) -> dict:
        t3 = Task(3, dataflow_tag, "LoadData")
        t3.begin()
        start = perf_counter()

        train_data = load_data(self.data_path, require_header=True)
        data_size = train_data[-1]
        valid_size = int(round(data_size * self.valid_frac))

        indices = {
            "valid": {"start": 0, "end": valid_size},
        }
        
        train_data = load_data(self.data_path)
        valid_data = load_data_for_range(
            self.data_path,  
            indices["valid"]["start"],
            indices["valid"]["end"],
        )   

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t3_input = DataSet("iLoadData", [Element(to_dfanalyzer)])
        t3.add_dataset(t3_input)
        t3_output = DataSet("oLoadData", [Element([])])
        t3.add_dataset(t3_output)
        t3.end()

        return {"train": train_data, "valid": valid_data}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]

        t4 = Task(4, dataflow_tag, "InitializeClient",
                  dependency=Task(3, dataflow_tag, "LoadData"))
        t4.begin()
        start = perf_counter()
        duration = perf_counter() - start

        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t4_input = DataSet("iInitializeClient", [Element(to_dfanalyzer)])
        t4.add_dataset(t4_input)
        t4_output = DataSet("oInitializeClient", [Element([])])
        t4.add_dataset(t4_output)
        t4.end()

        # set number of samples for compatibility with SKLearnExecutor
        # load_data returns a tuple like (x, y, n_samples)
        try:
            self.n_samples = data["train"][-1]
        except Exception:
            # fallback: compute from x_train if available
            try:
                x_train = self.train_data[0]
                self.n_samples = len(x_train)
            except Exception:
                self.n_samples = 0

    def train(
        self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext
    ) -> Tuple[dict, dict]:
        t5 = Task(5 + 4 * (curr_round), dataflow_tag, "ClientTraining")
        if curr_round == 0:
            t5.add_dependency(
                Dependency(
                    ["InitializeClient", "ClientValidation"],
                    ["4", "0"],
                )
            )
        else:
            t5.add_dependency(
                Dependency(
                    ["InitializeClient", "ClientValidation"],
                    ["4", str(8 + 4 * (curr_round - 2))],
                )
            )

        t5.begin()
        start = perf_counter()
        timestamp = datetime.datetime.now()
        
        # Update hyperparameters from global if provided
        if global_param:
            self.eps = global_param.get("eps", self.eps)
            self.min_samples = global_param.get("min_samples", self.min_samples)

        to_dfanalyzer = [
            self.hash_trial,
            self.client_id,
            curr_round,
            self.eps,
            self.min_samples,
            timestamp
        ]

        t5_input = DataSet("iClientTraining", [Element(to_dfanalyzer)])
        t5.add_dataset(t5_input)

        # Get training data and perform local DBSCAN
        (x_train, y_train, train_size) = self.train_data
        x_train = self._sanitize_features(x_train, fl_ctx, "train")
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
        )
        dbscan.fit(x_train)

        # Extract core points and their labels
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        
        core_points = np.asarray(x_train[core_mask], dtype=np.float32)
        core_labels = np.asarray(dbscan.labels_[core_mask], dtype=np.int32)
        original_core_point_count = len(core_points)
        core_points, core_labels = self._limit_core_points(core_points, core_labels)
        if len(core_points) != original_core_point_count:
            self.log_info(
                fl_ctx,
                f"Limited core points from {original_core_point_count} to {len(core_points)}",
            )

        saved_path = None
        try:
            filename = f"dbscan_client_{self.client_id}_r{curr_round}.npz"

            # Ensure numpy arrays for saving
            cp_arr = np.array(core_points)
            cl_arr = np.array(core_labels)
            np.savez_compressed(filename, core_points=cp_arr, core_labels=cl_arr)
            saved_path = filename
        except Exception as e:
            try:
                self.log_error(fl_ctx, f"dbscan learner: failed to save artifact: {e}")
            except Exception:
                pass
        # Prepare return parameters - only core points and their labels
        # Ensure all numpy types are converted to Python native types for serialization
        params = {
            "core_points": core_points.tolist() if isinstance(core_points, np.ndarray) else list(core_points),
            "core_labels": [int(x) for x in core_labels.tolist()] if isinstance(core_labels, np.ndarray) else [int(x) for x in core_labels],
            "eps": float(self.eps),
            "min_samples": int(self.min_samples),
            "n_clusters": int(len(set(core_labels)) - (1 if -1 in core_labels else 0))
        }

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()

        to_dfanalyzer = [
            self.hash_trial,
            self.client_id,
            curr_round,
            saved_path,
            params["n_clusters"],
            len(core_points),
            duration,
            timestamp
        ]

        t5_output = DataSet("oClientTraining", [Element(to_dfanalyzer)])
        t5.add_dataset(t5_output)
        t5.end()

        # Explicitly clean up the DBSCAN model object to prevent serialization issues
        del dbscan

        # Ensure all data is serializable before returning
        params = ensure_serializable(params)
        
        # Return None for the model object to avoid serialization issues with DBSCAN's internal state
        return params, None

    def validate(
        self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext
    ) -> Tuple[dict, dict]:
        t7 = Task(
            8 + 4 * (curr_round-1),
            dataflow_tag,
            "ClientValidation",
            dependency=Task(7 + 4 * (curr_round-1), dataflow_tag, "Assemble"),
        )
        t7.begin()
        start = perf_counter()
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, curr_round, timestamp]
        t7_input = DataSet("iClientValidation", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_input)

        # Get validation data
        (x_valid, y_valid, valid_size) = self.valid_data
        x_valid = self._sanitize_features(x_valid, fl_ctx, "valid")

        # Use global parameters for validation
        if global_param and "core_points" in global_param and len(global_param["core_points"]) > 0:
            core_points = global_param["core_points"]
            core_labels = global_param["core_labels"]
            
            # Convert lists back to numpy arrays if needed
            if isinstance(core_points, list):
                core_points = np.asarray(core_points, dtype=np.float32)
            else:
                core_points = np.asarray(core_points, dtype=np.float32)
            core_points = self._sanitize_features(core_points, fl_ctx, "global_core")
            
            # Assign points to nearest core points within eps
            nn = NearestNeighbors(radius=self.eps)
            nn.fit(core_points)
            
            # Find neighbors within eps
            distances, indices = nn.radius_neighbors(x_valid)
            
            # Assign labels based on nearest core points
            y_pred = np.full(len(x_valid), -1)  # Initialize all as noise
            for i, neighbor_idx in enumerate(indices):
                if len(neighbor_idx) > 0:
                    # Assign the label of the nearest core point
                    nearest_idx = neighbor_idx[np.argmin(distances[i])]
                    y_pred[i] = core_labels[nearest_idx]
            
            # Calculate validation metrics
            if len(set(y_pred)) > 1:  # More than one cluster
                silhouette = silhouette_score(x_valid, y_pred)
                calinski = calinski_harabasz_score(x_valid, y_pred)
                self.log_info(fl_ctx, f"Silhouette Score {silhouette:.4f}")
                self.log_info(fl_ctx, f"Calinski-Harabasz Score {calinski:.4f}")
                metrics = {
                    "Silhouette Score": silhouette,
                    "Calinski-Harabasz Score": calinski
                }
            else:
                metrics = {
                    "Silhouette Score": 0.0,
                    "Calinski-Harabasz Score": 0.0
                }
                silhouette = 0.0
        else:
            metrics = {
                "Silhouette Score": 0.0,
                "Calinski-Harabasz Score": 0.0
            }
            silhouette = 0.0

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()

        to_dfanalyzer = [self.hash_trial, self.client_id, curr_round, silhouette, duration, timestamp]
        
        t7_output = DataSet("oClientValidation", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_output)
        t7.end()
        
        return metrics, None

    def finalize(self, fl_ctx: FLContext) -> None:
        curr_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        timestamp = datetime.datetime.now()

        t9 = Task(
            9 + 4 * (curr_round),
            dataflow_tag,
            "FinalizeClient",
            dependency=Task(8 + 4 * (curr_round-1), dataflow_tag, "ClientValidation"),
        )
        t9.begin()
        start = perf_counter()
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")

        duration = perf_counter() - start
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t9_output = DataSet("oFinalizeClient", [Element(to_dfanalyzer)])
        t9.add_dataset(t9_output)
        t9.end()