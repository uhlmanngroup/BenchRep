# Reductions and clustering

Dimensionality reduction gives a lower-dimensional view of the representation. Clustering assigns observations to groups. BenchRep stores both alongside the original representation, so you can inspect them together without replacing the matrix being evaluated.

## Choose a representation deliberately

The methods do not all consume the same input automatically:

| Method | Implementation | Input | Default output |
| --- | --- | --- | --- |
| PCA | scikit-learn | `adata.X` | `adata.obsm["X_pca"]` |
| UMAP | Scanpy | `use_rep` / Scanpy representation selection | `adata.obsm["X_umap"]` |
| t-SNE | Scanpy | `use_rep` / Scanpy representation selection | `adata.obsm["X_tsne"]` |
| KMeans | scikit-learn | `adata.X` | `adata.obs["kmeans"]` |
| Leiden | Scanpy | Neighbor graph constructed from `use_rep` / Scanpy representation selection | `adata.obs["leiden"]` |
| HDBSCAN | scikit-learn | `adata.X` | `adata.obs["hdbscan"]` |

For UMAP, t-SNE, and Leiden, `use_rep: X` explicitly selects the original matrix; a named `obsm` entry selects that representation. When `use_rep` is omitted, Scanpy chooses a representation according to its own rules, potentially using or computing PCA. `n_pcs` controls how many principal components are used where applicable.

<strong class="highlight">Computing PCA or UMAP does not redirect every subsequent analysis to that reduction.</strong> KMeans and HDBSCAN still cluster `X`, and internal clustering metrics still measure geometry in `X`, even if Leiden used another representation. To change the common representation for all analyses, supply it in `X`; see [AnnData as the representation interface](inputs.md#anndata-as-the-representation-interface).

## Configure reductions and clusters

For example, this config computes a UMAP directly from the saved embedding and fits ten KMeans clusters to the same embedding:

```yaml
stage: evaluation
source:
  anndata_path: outputs/embeddings.h5ad
reductions:
  umap:
    enabled: true
    params:
      use_rep: X
clustering:
  kmeans:
    enabled: true
    params:
      n_clusters: 10
```

PCA is enabled automatically when AnnData is available; its default number of components is the smaller of 30, the observation count, and the feature count. UMAP, t-SNE, and every clustering method require explicit activation. KMeans additionally requires `n_clusters`.

KMeans asks for a fixed number of clusters. Leiden finds communities in a neighbor graph, with its resolution controlling the partition's granularity. HDBSCAN groups observations by density and can leave observations unassigned as noise. These methods answer different grouping questions; agreement between a plot and a clustering is not guaranteed, particularly when they use different representations.

Configuration exposes method parameters, output keys, seeds, and overwrite behavior. Full accepted settings are in the [evaluation schema reference](../../reference/schemas/evaluation.md). Preprocessing such as scaling features should be chosen before evaluation; BenchRep does not automatically standardize the matrix for these methods.

???+ warning "UMAP and Leiden own separate neighbor graphs"
    Both methods construct a graph and default to the neighbor key `neighbors`. If both are enabled, give them distinct `params.neighbors_key` values, such as `umap_neighbors` and `leiden_neighbors`, to preserve both graphs. Otherwise, the resolver rejects their shared key unless Leiden explicitly permits overwriting it. Matching graph parameters do not make the two steps share a computed graph.

## Interpret and retain the outputs

Reductions add coordinates to `obsm`; clustering adds categorical labels to `obs`. Method parameters and diagnostic metadata are recorded under `adata.uns["benchrep"]`. PCA includes explained-variance information, KMeans includes inertia, and HDBSCAN also writes membership strengths to `<key_added>_probability`.

HDBSCAN's label `-1` means noise. BenchRep's clustering metrics exclude these observations when the cluster metadata identifies the result as HDBSCAN. A good score on assigned observations should therefore be considered together with how many observations were excluded.

Existing output keys are preserved unless that step's `params.overwrite` allows replacement. An output collision is a recoverable step failure; it does not silently rename the result. Choose distinct `key_added` values when retaining several analyses in the same artifact.

## Implementation

[`embeddings/reductions.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/embeddings/reductions.py) and [`embeddings/clustering.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/embeddings/clustering.py) adapt the underlying libraries to AnnData: validate inputs and output locations, invoke the estimator, and attach results and provenance. Their returned AnnData is committed by the [pipeline step](inputs.md#implementation) only on success.

The pipeline runs PCA, UMAP, and t-SNE before KMeans, Leiden, and HDBSCAN. Clustering metric steps depend on their corresponding clustering result. Thus a failed clustering can skip its metrics without preventing embedding statistics or a later predictability target from running. Figures read the resulting coordinates and labels afterward; plotting does not determine the clusters.
