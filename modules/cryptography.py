import ipdb
import numpy as np
from itertools import product

import torch
import torch.nn as nn


def modulo_add(a, b, mod):
    return (a + b) % mod


def modulo_mul(a, b, mod):
    return torch.matmul(a, b) % mod


class LinearBFVEncryption(nn.Module):
    def __init__(self, msg_len, pt_mod, ct_mod, n1, n2, err_k, device):
        super().__init__()
        self.msg_len = msg_len
        self.ct_mod = ct_mod
        self.pt_mod = pt_mod
        self.n1 = n1  # 192
        self.n2 = n2  # 192
        self.err_k = err_k
        self.device = device
        self.pt_scale = ct_mod // pt_mod

        self.n_ct_bits = int(np.ceil(np.log2(ct_mod)))
        self.ct_dict = torch.tensor(list(product([0, 1], repeat=self.n_ct_bits)), dtype=torch.float).unsqueeze(0).to(device)
        self.ct_bins = torch.arange(2**self.n_ct_bits, device=device).to(torch.float)

        self.pk, self.sk = self._keygen(msg_len, ct_mod)

    def _noise_gen(self, size):
        u = (torch.randn((self.n1, 1), device=self.device) * self.err_k).round()
        e2 = (torch.randn((self.n2, 1), device=self.device) * self.err_k).round()
        e1 = (torch.randn((size, 1), device=self.device).cuda() * self.err_k).round()
        return u, e1, e2

    def _keygen(self, size, modulus):
        """Generate a public and secret keys.
        Args:
            size: size of the vectors for the public and secret keys.
            modulus: field modulus.
        Returns:
            Public and secret key.
        """
        e = (torch.randn(size, self.n1).cuda() * self.err_k).round()

        sk = (torch.randn(size, self.n2).cuda() * self.err_k).round()
        a = torch.randint(0, modulus, size=(self.n2, self.n1), device=self.device, dtype=torch.float)
        b = modulo_add(
            modulo_mul(sk, a, modulus), e, modulus)
        pk = (-b, a)
        sk = sk
        return pk, sk

    def _update_noise(self, size):
        u, e1, e2 = self._noise_gen(size)

        ct1 = modulo_add(
            modulo_mul(self.pk[1], u, self.ct_mod), e2, self.ct_mod)

        self.encryptor = modulo_add(
            modulo_mul(self.pk[0], u, self.ct_mod), e1, self.ct_mod
        ).transpose(1, 0).unsqueeze(0)

        self.decryptor = modulo_mul(self.sk, ct1, self.ct_mod).transpose(1, 0).unsqueeze(0)

    def ct2bin(self, ct):
        B, N, _ = ct.size()
        ct_bin = self.ct_dict.repeat_interleave(B, 0)
        ct_bin = ct_bin.gather(1, ct.view(B, -1, 1).repeat_interleave(self.n_ct_bits, 2).to(torch.int64))
        coded = ct_bin.view(B, N, -1)
        return coded

    def get_soft_ct(self, llr):
        soft_likelihood = torch.matmul(llr, 2*self.ct_dict.squeeze(0).transpose(1, 0)-1)
        soft_likelihood = torch.softmax(5*soft_likelihood, dim=-1)
        soft_ct = torch.matmul(soft_likelihood, self.ct_bins)
        return soft_ct

    def encrypt(self, pt):
        """Encrypt an integer vector.
        Args:
            pt: integer message vector to be encrypted.
        Returns:
            Tuple representing a ciphertext.
        """
        self._update_noise(self.msg_len)

        m = pt.view(pt.size(0), 1, -1) * self.pt_scale
        ct0 = modulo_add(self.encryptor, m, self.ct_mod)

        # ct0_bin = self.ct2bin(ct0)
        return ct0

    def decrypt(self, ct, attack=False):
        """Decrypt a ciphertext
        Args:
            ct: ciphertext.
        Returns:
            Scaled integer vector representing the plaintext.
        """
        soft_ct = self.get_soft_ct(ct)

        if attack:
            decrypted = soft_ct
        else:
            decrypted = modulo_add(soft_ct, self.decryptor, self.ct_mod).contiguous()

        decrypted = decrypted.view(ct.shape[:-1]) / self.pt_scale
        return decrypted

    def __str__(self):
        return f'LBFV({self.pt_mod},{self.ct_mod},({self.n1},{self.n2}),{self.err_k})'
