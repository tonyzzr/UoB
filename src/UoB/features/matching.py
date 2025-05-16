"""Functions for computing similarity and correspondences between feature maps."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def compute_similarity_matrix(
    source_feats: torch.Tensor,
    query_feats: torch.Tensor,
    normalize: bool = True,
    temperature: Optional[float] = None,
    apply_softmax: bool = False
) -> torch.Tensor:
    """
    Computes a dense similarity matrix between source and query feature maps.

    Args:
        source_feats: Source feature map, expected shape [C, H_s, W_s].
        query_feats: Query feature map, expected shape [C, H_q, W_q].
                     Feature dimension C must match source_feats.
        normalize: If True, L2-normalize feature vectors before computing similarity
                   (results in cosine similarity).
        temperature: Optional temperature scaling factor applied before softmax.
                     Ignored if apply_softmax is False.
        apply_softmax: If True, apply softmax along the query dimension (dim=1).

    Returns:
        similarity_matrix: Tensor of shape [H_s * W_s, H_q * W_q] or
                           [H_s, W_s, H_q, W_q] if reshaped (currently returns flat).
                           Contains pairwise similarities (or softmax probabilities).
    """
    # Validate shapes
    if source_feats.ndim != 3 or query_feats.ndim != 3:
        raise ValueError(f"Expected 3D features [C, H, W], got {source_feats.ndim}D and {query_feats.ndim}D")
    if source_feats.shape[0] != query_feats.shape[0]:
        raise ValueError(f"Feature dimension C must match, got {source_feats.shape[0]} and {query_feats.shape[0]}")

    C = source_feats.shape[0]
    H_s, W_s = source_feats.shape[1], source_feats.shape[2]
    H_q, W_q = query_feats.shape[1], query_feats.shape[2]

    # Reshape features to [N, C] where N = H * W
    source_feats_flat = source_feats.reshape(C, -1).transpose(0, 1)  # [H_s*W_s, C]
    query_feats_flat = query_feats.reshape(C, -1).transpose(0, 1)    # [H_q*W_q, C]

    # Normalize features if requested
    if normalize:
        source_feats_flat = F.normalize(source_feats_flat, p=2, dim=1)
        query_feats_flat = F.normalize(query_feats_flat, p=2, dim=1)

    # Compute pairwise similarity (dot product)
    # Result shape: [H_s*W_s, H_q*W_q]
    similarity = torch.mm(source_feats_flat, query_feats_flat.transpose(0, 1))

    # Apply temperature scaling and softmax if requested
    if apply_softmax:
        if temperature is not None:
            similarity = similarity / temperature
        similarity = F.softmax(similarity, dim=1) # Softmax along query dimension

    # Note: The legacy function reshaped back to [H_s, W_s, H_q, W_q]
    # Keeping it flat [H_s*W_s, H_q*W_q] might be more common for downstream matching algos
    # Can add reshaping later if needed.

    return similarity

def find_nearest_neighbors(similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the nearest neighbor in the query set for each source feature.

    Args:
        similarity_matrix: Tensor of shape [N_s, N_q] containing pairwise
                           similarities (higher is better).

    Returns:
        Tuple containing:
            - nn_scores: Tensor of shape [N_s] containing the similarity score
                         of the nearest neighbor for each source feature.
            - nn_indices: Tensor of shape [N_s] containing the index (in the query set)
                          of the nearest neighbor for each source feature.
    """
    if similarity_matrix.ndim != 2:
        raise ValueError(f"Expected 2D similarity matrix [N_s, N_q], got {similarity_matrix.ndim}D")

    # Find the max similarity and corresponding index along the query dimension (dim=1)
    nn_scores, nn_indices = torch.max(similarity_matrix, dim=1)

    return nn_scores, nn_indices

def find_mutual_nearest_neighbors(similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find mutual nearest neighbors between source and query features.

    Args:
        similarity_matrix: Tensor of shape [N_s, N_q] containing pairwise
                           similarities (higher is better).

    Returns:
        Tuple containing:
            - mnn_source_indices: Tensor containing the indices of source features
                                   that are part of a mutual match.
            - mnn_query_indices: Tensor containing the indices of the corresponding
                                  matched query features.
    """
    if similarity_matrix.ndim != 2:
        raise ValueError(f"Expected 2D similarity matrix [N_s, N_q], got {similarity_matrix.ndim}D")

    N_s, N_q = similarity_matrix.shape

    # 1. Find nearest neighbor from source to query (S -> Q)
    _, nn_s_to_q_indices = find_nearest_neighbors(similarity_matrix)
    # nn_s_to_q_indices[i] = index of the NN query feature for source feature i

    # 2. Find nearest neighbor from query to source (Q -> S)
    # We use the transpose of the similarity matrix for this
    _, nn_q_to_s_indices = find_nearest_neighbors(similarity_matrix.T)
    # nn_q_to_s_indices[j] = index of the NN source feature for query feature j

    # 3. Find mutual matches
    # Create indices for source features (0 to N_s-1)
    source_indices = torch.arange(N_s, device=similarity_matrix.device)

    # Check the MNN condition:
    # For a source feature i, its NN query feature is j = nn_s_to_q_indices[i].
    # We need to check if the NN source feature for j is also i.
    # i.e., check if nn_q_to_s_indices[j] == i
    # which is equivalent to: nn_q_to_s_indices[nn_s_to_q_indices[i]] == i
    mutual_match_mask = nn_q_to_s_indices[nn_s_to_q_indices] == source_indices

    # Get the indices where the mask is True
    mnn_source_indices = source_indices[mutual_match_mask]
    # Get the corresponding query indices using the S->Q NN mapping
    mnn_query_indices = nn_s_to_q_indices[mutual_match_mask]

    return mnn_source_indices, mnn_query_indices

def find_k_nearest_neighbors(
    similarity_matrix: torch.Tensor,
    k: int,
    source_indices: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the top k nearest neighbors in the query set for each specified source feature.

    Args:
        similarity_matrix: Tensor of shape [N_s, N_q] containing pairwise
                           similarities (higher is better).
        k: The number of nearest neighbors to find.
        source_indices: Optional tensor of shape [M] containing the flat indices
                        of the source features for which to find k-NNs.
                        If None, finds k-NNs for all source features.

    Returns:
        Tuple containing:
            - knn_scores: Tensor of shape [M, k] or [N_s, k] containing the similarity
                          scores of the top k neighbors.
            - knn_indices: Tensor of shape [M, k] or [N_s, k] containing the flat indices
                           (in the query set) of the top k neighbors.
    """
    if similarity_matrix.ndim != 2:
        raise ValueError(f"Expected 2D similarity matrix [N_s, N_q], got {similarity_matrix.ndim}D")
    N_s, N_q = similarity_matrix.shape
    if k > N_q:
        raise ValueError(f"k ({k}) cannot be greater than the number of query features ({N_q})")

    if source_indices is None:
        # Find k-NN for all source features
        target_sim_matrix = similarity_matrix
    else:
        if source_indices.ndim != 1:
            raise ValueError(f"source_indices must be a 1D tensor, got {source_indices.ndim}D")
        if torch.max(source_indices) >= N_s or torch.min(source_indices) < 0:
             raise ValueError("source_indices contains out-of-bounds values.")
        # Select rows corresponding to the source indices
        target_sim_matrix = similarity_matrix[source_indices]

    # Find the top k scores and indices along the query dimension (dim=1)
    # torch.topk returns (values, indices)
    knn_scores, knn_indices = torch.topk(target_sim_matrix, k=k, dim=1)

    return knn_scores, knn_indices

# TODO: Add functions for sparse matching using the similarity matrix
# TODO: Consider reshaping similarity matrix optionally
