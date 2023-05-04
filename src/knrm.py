from typing import Dict, List

import torch
import torch.nn.functional as F


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))


class KNRM(torch.nn.Module):
    def __init__(self,
                 embedding_matrix: torch.FloatTensor,
                 mlp_state: Dict,
                 kernel_num: int = 11,
                 sigma: float = 0.1,
                 exact_sigma: float = 0.001,
                 out_layers: List[int] = [],
                 ):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0,
        )
        self.embed_weights = self.embeddings.state_dict()['weight']

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(mlp_state)
        self.out_activation = torch.nn.Sigmoid()

    def _kernal_mus(self, n_kernels):
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            mu = round(l_mu[i] - bin_size, 5)
            l_mu.append(mu)

        return list(reversed(l_mu))

    def _kernel_sigmas(self, n_kernels, sigma, exact_sigma, lamb=None):
        l_sigma = [exact_sigma]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        # use different sigmas for kernels
        if lamb:
            bin_size = 2.0 / (n_kernels - 1)
            l_sigma += [bin_size * lamb] * (n_kernels - 1)
        else:
            for i in range(1, n_kernels):
                l_sigma.append(sigma)

        return list(reversed(l_sigma))

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        mus = self._kernal_mus(self.kernel_num)
        sigmas = self._kernel_sigmas(self.kernel_num, self.sigma, self.exact_sigma)
        kernels = []
        for mu, sigma in zip(mus, sigmas):
            kernels.append(GaussianKernel(mu, sigma))

        kernels = torch.nn.ModuleList(kernels)
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        if len(self.out_layers) == 0:
            return torch.nn.Sequential(torch.nn.Linear(self.kernel_num, 1))

        dims = [self.kernel_num, *self.out_layers, 1]
        layers = []
        for i in range(1, len(dims)):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(dims[i - 1], dims[i]))

        return torch.nn.Sequential(*layers)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)
        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)

        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # Broadcast to match query and doc lenght
        query = self.embeddings(query).unsqueeze(2)
        doc = self.embeddings(doc).unsqueeze(1)

        return F.cosine_similarity(query, doc, dim=-1)

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left, Embedding], [Batch, Right, Embedding]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)

        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out
