import pytest
import torch

# Ensure src package is findable if running pytest from project root
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.UoB.features.matching import compute_similarity_matrix, find_nearest_neighbors, find_mutual_nearest_neighbors, find_k_nearest_neighbors


def test_compute_similarity_matrix_basic():
    """Tests basic functionality, shape, and normalization."""
    C, H, W = 4, 3, 2
    N = H * W # 6
    source_feats = torch.randn(C, H, W)
    query_feats = torch.randn(C, H, W)

    # Test without normalization or softmax
    sim = compute_similarity_matrix(source_feats, query_feats, normalize=False, apply_softmax=False)
    assert sim.shape == (N, N), f"Expected shape {(N, N)}, got {sim.shape}"
    assert sim.dtype == torch.float32

    # Test with normalization (cosine similarity)
    sim_norm = compute_similarity_matrix(source_feats, query_feats, normalize=True, apply_softmax=False)
    assert sim_norm.shape == (N, N)
    assert sim_norm.dtype == torch.float32
    # Cosine similarity should be between -1 and 1 (allow for small float errors)
    assert torch.all(sim_norm >= -1.001) and torch.all(sim_norm <= 1.001)

    # Manually check one cosine similarity value
    sf_flat = source_feats.reshape(C, -1).t() # N, C
    qf_flat = query_feats.reshape(C, -1).t()  # N, C
    sf_norm = torch.nn.functional.normalize(sf_flat, p=2, dim=1)
    qf_norm = torch.nn.functional.normalize(qf_flat, p=2, dim=1)
    expected_sim_00 = torch.dot(sf_norm[0], qf_norm[0])
    assert torch.allclose(sim_norm[0, 0], expected_sim_00)

def test_compute_similarity_matrix_softmax():
    """Tests softmax application."""
    C, H, W = 4, 3, 2
    N = H * W # 6
    source_feats = torch.randn(C, H, W)
    query_feats = torch.randn(C, H, W)

    # Test with softmax (optional temperature)
    sim_softmax = compute_similarity_matrix(source_feats, query_feats, normalize=True, apply_softmax=True)
    assert sim_softmax.shape == (N, N)
    # Check if rows sum to 1 (within tolerance)
    assert torch.allclose(sim_softmax.sum(dim=1), torch.ones(N))

    # Test with temperature
    temp = 0.1
    sim_softmax_temp = compute_similarity_matrix(source_feats, query_feats, normalize=True, apply_softmax=True, temperature=temp)
    assert sim_softmax_temp.shape == (N, N)
    assert torch.allclose(sim_softmax_temp.sum(dim=1), torch.ones(N))
    # Lower temperature should make distributions sharper (higher max value)
    assert torch.max(sim_softmax_temp) > torch.max(sim_softmax)

def test_compute_similarity_matrix_diff_shapes():
    """Tests with different H, W for source and query."""
    C = 4
    H_s, W_s = 3, 2
    H_q, W_q = 2, 3
    N_s = H_s * W_s # 6
    N_q = H_q * W_q # 6
    source_feats = torch.randn(C, H_s, W_s)
    query_feats = torch.randn(C, H_q, W_q)

    sim = compute_similarity_matrix(source_feats, query_feats, normalize=True)
    assert sim.shape == (N_s, N_q), f"Expected shape {(N_s, N_q)}, got {sim.shape}"

def test_compute_similarity_matrix_errors():
    """Tests expected error conditions."""
    C, H, W = 4, 3, 2
    source_feats = torch.randn(C, H, W)
    query_feats = torch.randn(C, H, W)

    # Incorrect dimensions
    with pytest.raises(ValueError, match="Expected 3D features"):
        compute_similarity_matrix(torch.randn(C, H), query_feats)
    with pytest.raises(ValueError, match="Expected 3D features"):
        compute_similarity_matrix(source_feats, torch.randn(C, H, W, 1))

    # Mismatched feature dimension C
    with pytest.raises(ValueError, match="Feature dimension C must match"):
        compute_similarity_matrix(source_feats, torch.randn(C + 1, H, W))

# --- Tests for find_nearest_neighbors --- 

def test_find_nearest_neighbors():
    """Tests finding nearest neighbors."""
    # Simple case
    # Source 0 -> Query 1 (max)
    # Source 1 -> Query 0 (max)
    # Source 2 -> Query 1 (max)
    sim = torch.tensor([
        [0.1, 0.9, 0.3],
        [0.8, 0.2, 0.5],
        [0.4, 0.7, 0.6]
    ])
    N_s, N_q = sim.shape
    expected_scores = torch.tensor([0.9, 0.8, 0.7])
    expected_indices = torch.tensor([1, 0, 1])

    scores, indices = find_nearest_neighbors(sim)
    assert scores.shape == (N_s,)
    assert indices.shape == (N_s,)
    assert torch.allclose(scores, expected_scores)
    assert torch.equal(indices, expected_indices)

    # Test with identity matrix (each source matches corresponding query)
    N = 5
    sim_eye = torch.eye(N)
    scores_eye, indices_eye = find_nearest_neighbors(sim_eye)
    assert torch.allclose(scores_eye, torch.ones(N))
    assert torch.equal(indices_eye, torch.arange(N))

def test_find_nearest_neighbors_errors():
    """Tests error handling for find_nearest_neighbors."""
    # Incorrect dimensions
    with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
        find_nearest_neighbors(torch.randn(3, 4, 5))
    with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
        find_nearest_neighbors(torch.randn(3))

# --- Tests for find_mutual_nearest_neighbors --- 

def test_find_mutual_nearest_neighbors():
    """Tests finding mutual nearest neighbors."""
    # Case 1: Simple MNN
    # S0 <-> Q1 (0.9 is max in row 0, 0.9 is max in col 1)
    # S1 -> Q0 (0.8 max), Q0 -> S1 (0.8 max) => S1 <-> Q0
    # S2 -> Q1 (0.7 max), Q1 -> S0 (0.9 max) => Not mutual
    sim = torch.tensor([
        [0.1, 0.9, 0.3],
        [0.8, 0.2, 0.5],
        [0.4, 0.7, 0.6]
    ])
    expected_mnn_s = torch.tensor([0, 1]) # Source indices 0 and 1 are MNN
    expected_mnn_q = torch.tensor([1, 0]) # Query indices 1 and 0 are corresponding MNN

    mnn_s, mnn_q = find_mutual_nearest_neighbors(sim)
    # Sort results for comparison as order might not be guaranteed
    s_sort_idx = torch.argsort(mnn_s)
    q_sort_idx = torch.argsort(expected_mnn_s)
    assert torch.equal(mnn_s[s_sort_idx], expected_mnn_s[q_sort_idx])
    assert torch.equal(mnn_q[s_sort_idx], expected_mnn_q[q_sort_idx])

    # Case 2: Identity matrix (all are MNN)
    N = 5
    sim_eye = torch.eye(N)
    expected_mnn_s_eye = torch.arange(N)
    expected_mnn_q_eye = torch.arange(N)
    mnn_s_eye, mnn_q_eye = find_mutual_nearest_neighbors(sim_eye)
    assert torch.equal(mnn_s_eye, expected_mnn_s_eye)
    assert torch.equal(mnn_q_eye, expected_mnn_q_eye)

    # Case 3: Test with the specific failing example
    # S0 -> Q1 (0.9 max), Q1 -> S0 (0.9 max) => S0 <-> Q1 (Mutual)
    # S1 -> Q1 (0.8 max), Q1 -> S0 (0.9 max) => Not Mutual
    sim_case3 = torch.tensor([
        [0.1, 0.9],
        [0.2, 0.8]
    ])
    expected_mnn_s_case3 = torch.tensor([0]) # Only source 0 is MNN
    expected_mnn_q_case3 = torch.tensor([1]) # Corresponding query is 1
    mnn_s_case3, mnn_q_case3 = find_mutual_nearest_neighbors(sim_case3)
    assert torch.equal(mnn_s_case3, expected_mnn_s_case3)
    assert torch.equal(mnn_q_case3, expected_mnn_q_case3)

    # Case 4: Rectangular matrix
    # S0 -> Q1 (0.9 max), Q1 -> S0 (0.9 max) => S0 <-> Q1
    # S1 -> Q0 (0.8 max), Q0 -> S1 (0.8 max) => S1 <-> Q0
    sim_rect = torch.tensor([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.4, 0.7] # S2 -> Q1, Q1 -> S0
    ])
    expected_mnn_s_rect = torch.tensor([0, 1])
    expected_mnn_q_rect = torch.tensor([1, 0])
    mnn_s_rect, mnn_q_rect = find_mutual_nearest_neighbors(sim_rect)
    s_sort_idx = torch.argsort(mnn_s_rect)
    q_sort_idx = torch.argsort(expected_mnn_s_rect)
    assert torch.equal(mnn_s_rect[s_sort_idx], expected_mnn_s_rect[q_sort_idx])
    assert torch.equal(mnn_q_rect[s_sort_idx], expected_mnn_q_rect[q_sort_idx])

def test_find_mutual_nearest_neighbors_errors():
    """Tests error handling for find_mutual_nearest_neighbors."""
    # Incorrect dimensions
    with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
        find_mutual_nearest_neighbors(torch.randn(3, 4, 5))
    with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
        find_mutual_nearest_neighbors(torch.randn(3))

# --- Tests for find_k_nearest_neighbors --- 

def test_find_k_nearest_neighbors():
    """Tests finding k-nearest neighbors."""
    sim = torch.tensor([
        [0.1, 0.9, 0.3, 0.8], # NN order: 1, 3, 2, 0
        [0.8, 0.2, 0.5, 0.6], # NN order: 0, 3, 2, 1
        [0.4, 0.7, 0.6, 0.5]  # NN order: 1, 2, 3, 0
    ])
    N_s, N_q = sim.shape

    # Test k=1 (should match find_nearest_neighbors)
    k1_scores, k1_indices = find_k_nearest_neighbors(sim, k=1)
    exp_k1_scores = torch.tensor([[0.9], [0.8], [0.7]])
    exp_k1_indices = torch.tensor([[1], [0], [1]])
    assert k1_scores.shape == (N_s, 1)
    assert k1_indices.shape == (N_s, 1)
    assert torch.allclose(k1_scores, exp_k1_scores)
    assert torch.equal(k1_indices, exp_k1_indices)

    # Test k=3
    k3_scores, k3_indices = find_k_nearest_neighbors(sim, k=3)
    exp_k3_scores = torch.tensor([
        [0.9, 0.8, 0.3],
        [0.8, 0.6, 0.5],
        [0.7, 0.6, 0.5]
    ])
    exp_k3_indices = torch.tensor([
        [1, 3, 2],
        [0, 3, 2],
        [1, 2, 3]
    ])
    assert k3_scores.shape == (N_s, 3)
    assert k3_indices.shape == (N_s, 3)
    assert torch.allclose(k3_scores, exp_k3_scores)
    assert torch.equal(k3_indices, exp_k3_indices)

    # Test with specific source_indices
    source_indices = torch.tensor([0, 2])
    k2_scores_subset, k2_indices_subset = find_k_nearest_neighbors(sim, k=2, source_indices=source_indices)
    exp_k2_scores_subset = torch.tensor([
        [0.9, 0.8],
        [0.7, 0.6]
    ])
    exp_k2_indices_subset = torch.tensor([
        [1, 3],
        [1, 2]
    ])
    assert k2_scores_subset.shape == (len(source_indices), 2)
    assert k2_indices_subset.shape == (len(source_indices), 2)
    assert torch.allclose(k2_scores_subset, exp_k2_scores_subset)
    assert torch.equal(k2_indices_subset, exp_k2_indices_subset)

def test_find_k_nearest_neighbors_errors():
    """Tests error handling for find_k_nearest_neighbors."""
    sim = torch.randn(3, 4)
    
    # Incorrect dimensions
    with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
        find_k_nearest_neighbors(torch.randn(3, 4, 5), k=2)
        
    # k > N_q
    with pytest.raises(ValueError, match="k .* cannot be greater than"):
        find_k_nearest_neighbors(sim, k=5)
        
    # Invalid source_indices dimensions
    with pytest.raises(ValueError, match="source_indices must be a 1D tensor"):
        find_k_nearest_neighbors(sim, k=2, source_indices=torch.tensor([[0],[1]]))
        
    # Out-of-bounds source_indices
    with pytest.raises(ValueError, match="source_indices contains out-of-bounds"):
        find_k_nearest_neighbors(sim, k=2, source_indices=torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="source_indices contains out-of-bounds"):
        find_k_nearest_neighbors(sim, k=2, source_indices=torch.tensor([-1, 1])) 