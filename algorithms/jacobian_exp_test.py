import numpy as np

def hat(v):
    """벡터를 반대칭 행렬(Skew-symmetric matrix)로 변환"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def so3_exp(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        return np.eye(3) + hat(phi)
    
    u = phi / theta
    K = hat(u)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

def right_jacobian(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        return np.eye(3)
    
    u = phi / theta
    K = hat(u)
    return np.eye(3) - ((1 - np.cos(theta)) / theta) * K + \
           ((theta - np.sin(theta)) / theta) * np.dot(K, K)

# --- 수식 검증 ---
phi = np.array([0.1, 0.2, 0.3])
delta_phi = np.array([0.01, -0.01, 0.02])

# 좌변: Exp(phi + delta_phi)
LHS = so3_exp(phi + delta_phi)

# 우변: Exp(phi) * Exp(Jr(phi) * delta_phi)
Jr = right_jacobian(phi)
R_phi = so3_exp(phi)
R_delta = so3_exp(Jr @ delta_phi)
RHS = R_phi @ R_delta

print("Left Hand Side (Exp(phi + d_phi)):\n", LHS)
print("\nRight Hand Side (Exp(phi) * Exp(Jr * d_phi)):\n", RHS)
print("\nDifference (L2 Norm):", np.linalg.norm(LHS - RHS))