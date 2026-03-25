import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from typing import List

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.assembler import Assembler
# keep imports if other code expects them; we will not call make_model_learnable
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
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


class DBSCANAssembler(Assembler):
    def __init__(
        self,
        hash_trial: str = "unknown_trial",
        eps: float = 0.5,
        min_samples: int = 5,
    ):
        # Assembler expects a data_kind; use WEIGHTS similar to KMeans implementation
        super().__init__(data_kind=DataKind.WEIGHTS)
        self.eps = eps
        self.min_samples = min_samples
        self.hash_trial = hash_trial
        # Will store the global clustering state
        self.global_core_points = None
        self.global_core_labels = None
        self.current_round = 0

    def _merge_clusters(self, core_points_list, core_labels_list):
        """Merge clusters from different clients based on core point connectivity."""
        # Convert lists to numpy arrays if needed
        core_points_array_list = []
        for cp in core_points_list:
            if isinstance(cp, list):
                core_points_array_list.append(np.asarray(cp, dtype=np.float32))
            else:
                core_points_array_list.append(np.asarray(cp, dtype=np.float32))

        if not core_points_array_list:
            return np.array([]), np.array([])

        # Concatenate all core points
        all_core_points = np.vstack(core_points_array_list)

        # Build global connectivity matrix directly using NearestNeighbors
        nn = NearestNeighbors(radius=self.eps)
        nn.fit(all_core_points)
        global_connectivity = nn.radius_neighbors_graph(mode="distance")

        # Find connected components to determine global clusters
        n_components, labels = connected_components(
            csgraph=global_connectivity, directed=False, return_labels=True
        )

        # Preserve original client labels in metadata for traceability
        # (core_labels_list now preserved in returned metadata)
        self.client_labels_list = core_labels_list
        
        return all_core_points, labels

    def get_model_params(self, dxo: DXO) -> dict:
        """Extract model parameters from a client's DXO for collection."""
        t6 = Task(
            6 + 4 * (self.current_round),
            dataflow_tag,
            "GetModelParams",
        )
        t6.begin()
        data = dxo.data

        # Save client data to file and store only the path in the analyzer
        saved_path = None
        try:
            filename = f"dbscan_client_r{self.current_round}.npz"

            core_points = data.get("core_points", [])
            core_labels = data.get("core_labels", [])
            
            cp_arr = np.array(core_points)
            cl_arr = np.array(core_labels)
            np.savez_compressed(filename, core_points=cp_arr, core_labels=cl_arr)
            saved_path = filename
        except Exception as e:
            try:
                self.log_error(None, f"dbscan assembler: failed to save client artifact: {e}")
            except Exception:
                pass

        # Send trial_id, artifact_path (as "center"), and count to match schema
        to_dfanalyzer = ensure_serializable([self.hash_trial, saved_path, str(len(data.get("core_points", []))), data.get("n_clusters", 0)])
        t6_input = DataSet("iGetModelParams", [Element(to_dfanalyzer)])
        t6.add_dataset(t6_input)
        t6_output = DataSet("oGetModelParams", [Element([])])
        t6.add_dataset(t6_output)
        t6.end()

        return {
            "core_points": data.get("core_points", []),
            "core_labels": data.get("core_labels", []),
            "eps": data.get("eps", self.eps),
            "min_samples": data.get("min_samples", self.min_samples),
            "n_clusters": data.get("n_clusters", 0),
        }

    def assemble(self, data: dict, fl_ctx: FLContext) -> DXO:
        """
        Assemble core points from different clients and merge clusters.

        NOTE: 'data' is expected to be a dict mapping client_name -> client_payload_dict
        where client_payload_dict contains keys "core_points" and "core_labels".
        We do NOT call client_data.get_model() to avoid pulling in non-serializable objects.
        """
        t6 = Task(7 + 4 * (self.current_round), dataflow_tag, "Assemble")
        if self.current_round == 0:
            t6.add_dependency(
                Dependency(
                    ["ClientTraining"],
                    [str(5 + 4 * (self.current_round))],
                )
            )
        else:
            t6.add_dependency(
                Dependency(
                    ["ClientTraining", "ClientValidation"],
                    [str(5 + 4 * self.current_round), str(8 + 4 * (self.current_round - 1))],
                )
            )

        t6.begin()
        start = perf_counter()
        timestamp = datetime.datetime.now()

        to_dfanalyzer = ensure_serializable([self.hash_trial, self.current_round, timestamp])

        t6_input = DataSet("iAssemble", [Element(to_dfanalyzer)])
        t6.add_dataset(t6_input)

        # Extract core points from each client payload (expect client payloads are plain dicts)
        core_points_list = []
        core_labels_list = []

        # 'data' should be a dict of client_name -> payload_dict. If it's a list, handle accordingly.
        self.log_info(fl_ctx, f"Assembling {len(data)} client updates")

        # First round: just initialize and return params (no merging)
        if self.current_round == 0:
            params = {
                "eps": float(self.eps),
                "min_samples": int(self.min_samples),
            }
            self.current_round += 1

            duration = perf_counter() - start
            timestamp = datetime.datetime.now()
            to_dfanalyzer = ensure_serializable(
                [self.hash_trial, self.current_round, None, None, duration, timestamp]
            )
            t6_output = DataSet("oAssemble", [Element(to_dfanalyzer)])
            t6.add_dataset(t6_output)
            t6.end()

            # Return plain serializable dict — do NOT embed fl_ctx or create ModelLearnable
            dxo = DXO(data_kind=self.expected_data_kind, data=ensure_serializable(params))
            return dxo

        # Support both dict and list shapes for 'data' (robustness)
        if isinstance(data, dict):
            iterable = data.items()
        else:
            # If NVFlare provided a list of client wrappers, attempt to extract .get_model() safely:
            iterable = []
            for item in data:
                # If it's a plain dict already, use it
                if isinstance(item, dict):
                    iterable.append((None, item))
                else:
                    # attempt to access model() in a safe way, but avoid passing fl_ctx
                    try:
                        client_payload = item.get_model()
                        if isinstance(client_payload, dict):
                            iterable.append((None, client_payload))
                    except Exception:
                        # ignore items we can't safely introspect
                        continue

        # Populate core_points_list from payloads (each payload is expected to be a dict)
        for _, client_payload in iterable:
            if not isinstance(client_payload, dict):
                continue
            core_points = client_payload.get("core_points", None)
            core_labels = client_payload.get("core_labels", None)
            if core_points is not None and len(core_points) > 0:
                core_points_list.append(np.asarray(core_points, dtype=np.float32))
                # If client didn't provide labels, create placeholders
                if core_labels is None:
                    core_labels_list.append(np.zeros(len(core_points), dtype=np.int32))
                else:
                    core_labels_list.append(np.asarray(core_labels, dtype=np.int32))

        # Merge clusters from all clients
        all_core_points, global_labels = self._merge_clusters(core_points_list, core_labels_list)

        # Convert to serializable lists
        serial_core_points = (
            all_core_points.tolist()
            if isinstance(all_core_points, np.ndarray)
            else ensure_serializable(all_core_points)
        )
        serial_global_labels = (
            [int(x) for x in global_labels.tolist()]
            if isinstance(global_labels, np.ndarray)
            else ensure_serializable(global_labels)
        )

        # Update assembler state
        self.global_core_points = serial_core_points
        self.global_core_labels = serial_global_labels

        # Adapt eps based on global core-point density, if we have enough cores
        if isinstance(all_core_points, np.ndarray) and len(all_core_points) >= max(
            10, self.min_samples
        ):
            try:
                # Use k-distance (k = min_samples) on global core points
                k = int(max(2, self.min_samples))
                k = min(k, len(all_core_points))
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(all_core_points)
                dists, _ = nn.kneighbors(all_core_points)
                kth = dists[:, -1]

                # Pick a robust upper-quantile as global eps candidate
                new_eps = float(np.percentile(kth, 90))

                # Avoid degenerate values; keep within a reasonable band
                if np.isfinite(new_eps) and new_eps > 0.0:
                    # Blend slightly with previous eps for stability
                    self.eps = 0.5 * float(self.eps) + 0.5 * new_eps
            except Exception:
                # On any failure, keep existing eps
                self.log_warning(fl_ctx, "DBSCAN assembler: failed to adapt eps based on global core points; keeping previous eps")
                pass

        # Prepare params for next round
        params = {
            "core_points": self.global_core_points,
            "core_labels": self.global_core_labels,
            "eps": float(self.eps),
            "min_samples": int(self.min_samples),
        }

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()

        # Save core points and labels to a small artifact file and include the path
        # in the analyzer payload instead of embedding large arrays.
        saved_path = None
        try:
            filename = f"dbscan_r{self.current_round}.npz"

            # numpy can save lists/arrays; convert to arrays for consistency
            cp_arr = np.array(serial_core_points)
            gl_arr = np.array(serial_global_labels)
            np.savez_compressed(filename, core_points=cp_arr, core_labels=gl_arr)
            saved_path = filename
        except Exception as e:
            try:
                self.log_error(fl_ctx, f"dbscan learner: failed to save artifact: {e}")
            except Exception:
                pass

        # Keep payload small: provide artifact path (or None) plus timing info
        to_dfanalyzer = ensure_serializable([self.hash_trial, self.current_round, saved_path, self.eps, self.min_samples, duration, timestamp])
        t6_output = DataSet("oAssemble", [Element(to_dfanalyzer)])
        t6.add_dataset(t6_output)
        t6.end()

        self.current_round += 1

        # Ensure full serializability and return a plain dict in DXO (no ModelLearnable, no fl_ctx)
        dxo = DXO(data_kind=self.expected_data_kind, data=ensure_serializable(params))
        return dxo