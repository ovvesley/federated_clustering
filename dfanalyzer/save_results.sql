-- Prepare Data Input Parameters (KMeans)
COPY (
    SELECT
        trial_id,
        dataset_name,
        randomize,
        out_path,
        duration
    FROM iPrepareData
)
INTO 'results/iPrepareData.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- FL Job Configuration (KMeans)
COPY (
    SELECT
        trial_id,
        task_name,
        data_path,
        site_num,
        site_name_prefix,
        data_size,
        valid_frac,
        num_rounds
    FROM iJobConfig
)
INTO 'results/iJobConfig.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Data Loading Input Parameters (KMeans)
COPY (
    SELECT
        trial_id,
        client_id,
        loading_time,
        timestamp
    FROM iLoadData
)
INTO 'results/iLoadData.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Initialization Input Parameters (KMeans)
COPY (
    SELECT
        trial_id,
        client_id,
        n_samples,
        duration,
        timestamp
    FROM iInitializeClient
)
INTO 'results/iInitializeClient.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Training Input Parameters (KMeans)
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        n_clusters,
        n_samples,
        max_iter,
        n_init,
        reassignment_ratio,
        random_state,
        timestamp
    FROM iClientTraining
)
INTO 'results/iClientTraining.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Training Output (local results and timing) (KMeans)
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        center_local,
        count_local,
        center_global,
        training_time,
        timestamp
    FROM oClientTraining
)
INTO 'results/oClientTraining.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Training Input Parameters (DBSCAN)
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        eps,
        min_samples,
        timestamp
    FROM iClientTraining
)
INTO 'results/iClientTraining_dbscan.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Training Output (local results and timing) (DBSCAN)
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        core_points_path,
        n_clusters,
        count_core_points,
        training_time,
        timestamp
    FROM oClientTraining
)
INTO 'results/oClientTraining_dbscan.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Model Parameters Aggregation Input (KMeans)
COPY (
    SELECT
        trial_id,
        center,
        count
    FROM iGetModelParams
)
INTO 'results/iGetModelParams.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Model Parameters Aggregation Input (DBSCAN)
COPY (
    SELECT
        trial_id,
        core_points_path,
        count_core_points,
        n_clusters
    FROM iGetModelParams
)
INTO 'results/iGetModelParams_dbscan.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';
-- Aggregated Global Parameters (KMeans)
COPY (
    SELECT
        trial_id,
        center,
        count,
        assembling_time,
        minibatch_kmeans_time,
        timestamp
    FROM oAssemble
)
INTO 'results/oAssemble.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Aggregated Global Parameters (DBSCAN)
COPY (
    SELECT
        trial_id,
        current_round,
        core_points_path,
        eps,
        min_samples,
        dbscan_time,
        timestamp
    FROM oAssemble
)
INTO 'results/oAssemble_dbscan.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Aggregation Input Parameters (KMeans)
COPY (
    SELECT
        trial_id,
        current_round,
        n_feature,
        n_cluster,
        timestamp
    FROM iAssemble
)
INTO 'results/iAssemble.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Aggregation Input Parameters (DBSCAN)
COPY (
    SELECT
        trial_id,
        current_round,
        timestamp
    FROM iAssemble
)
INTO 'results/iAssemble_dbscan.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Validation Metrics
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        silhouette_score,
        validation_time,
        timestamp
    FROM oClientValidation
)
INTO 'results/oClientValidation.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Validation Input Parameters (KMeans)
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        timestamp
    FROM iClientValidation
)
INTO 'results/iClientValidation.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Finalization Metadata
COPY (
    SELECT
        trial_id,
        client_id,
        duration,
        timestamp
    FROM oFinalizeClient
)
INTO 'results/oFinalizeClient.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';