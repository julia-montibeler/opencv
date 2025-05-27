import cv2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# 1. Carregar imagem
img = cv2.imread("imagem.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_small = cv2.resize(img_rgb, (100, 100))  # use tamanho pequeno para testes

# 2. Criar vetor [R, G, B, x, y]
h, w = img_small.shape[:2]
X_rgb = img_small.reshape(-1, 3).astype(np.float64)

# Coordenadas normalizadas
X_coords = np.indices((h, w)).transpose(1, 2, 0).reshape(-1, 2).astype(np.float64)
X_coords[:, 0] /= h  # y
X_coords[:, 1] /= w  # x

# Vetor final de entrada: [R, G, B, x, y]
X = np.hstack((X_rgb, X_coords))  # shape: (n_pixels, 5)
n, d = X.shape

# 3. Inicializar GMM
K = 3
np.random.seed(0)

pi = np.full(K, 1/K)
mu = X[np.random.choice(n, K, replace=False)]
sigma = np.array([np.cov(X.T) for _ in range(K)])

def compute_log_likelihood(X, pi, mu, sigma):
    ll = 0
    for k in range(K):
        ll += pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
    return np.sum(np.log(ll))

# 4. Executar EM
max_iter = 100
tol = 1e-4
log_likelihoods = []

for iteration in range(max_iter):
    # E-step
    gamma = np.zeros((n, K))
    for k in range(K):
        gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    # M-step
    Nk = gamma.sum(axis=0)
    for k in range(K):
        pi[k] = Nk[k] / n
        mu[k] = (gamma[:, k].reshape(-1, 1) * X).sum(axis=0) / Nk[k]
        diff = X - mu[k]
        sigma[k] = (gamma[:, k].reshape(-1, 1) * diff).T @ diff / Nk[k]

    # Verificar convergência
    ll = compute_log_likelihood(X, pi, mu, sigma)
    log_likelihoods.append(ll)
    if iteration > 0 and abs(ll - log_likelihoods[-2]) < tol:
        print(f"Convergiu na iteração {iteration}")
        break

# 5. Resultado: clusterização
labels = np.argmax(gamma, axis=1)

# 6. Montar imagem segmentada (usando apenas o RGB das médias)
mu_rgb = mu[:, :3]  # pegar só as cores
segmented = mu_rgb[labels].reshape(img_small.shape).astype(np.uint8)

# 7. Mostrar imagem
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img_small)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmentado (EM com cor + posição)")
plt.imshow(segmented)
plt.axis("off")
plt.tight_layout()
plt.show()
