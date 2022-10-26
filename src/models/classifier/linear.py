import torch.nn as nn


class FeedForward(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.2):
        super(FeedForward, self).__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )


class DeepBiAffine(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_p: float = 0.2,
        feat_embed_dim: int = 0,
    ):
        super(DeepBiAffine, self).__init__()
        self.W_left = FeedForward(input_dim, hidden_dim, hidden_dim, dropout_p)
        self.W_right = FeedForward(input_dim, hidden_dim, hidden_dim, dropout_p)

        self.W_s = nn.Bilinear(hidden_dim, hidden_dim, output_dim)
        self.V_left = nn.Linear(hidden_dim, output_dim)
        self.V_right = nn.Linear(hidden_dim, output_dim)

        self.disable_feat = feat_embed_dim == 0
        if not self.disable_feat:
            self.W_feat = FeedForward(feat_embed_dim, 100, output_dim)

    def forward(self, h_ik, h_kj, feat=None):
        h_ik = self.W_left(h_ik)
        h_kj = self.W_right(h_kj)
        y = self.W_s(h_ik, h_kj) + self.V_left(h_ik) + self.V_right(h_kj)

        if not self.disable_feat:
            y_f = self.W_feat(feat)
            y = y + y_f

        return y
