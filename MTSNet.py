import torch
import torch.nn as nn
import random
from einops import rearrange, repeat
from timm.models.layers import PatchEmbed, Mlp, DropPath


class Multi_Scale_Convformer(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()

        self.att2conv = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.att2conv2 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=int(kernal_length * 0.5), padding=int(kernal_length * 0.5) // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.att2conv3 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=int(kernal_length * 0.5 * 0.5), padding=int(kernal_length * 0.5 * 0.5) // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.att2conv(x) + self.att2conv2(x) + self.att2conv3(x)
        return out


class Convformer(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()

        self.att2conv = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.att2conv(x)
        return out


class Multi_Level_Convformer(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()

        self.att2conv1 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=int(kernal_length * 0.5), padding=int(kernal_length * 0.5) // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.att2conv2 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x1 = self.att2conv1(x)
        x2 = self.att2conv2(x1)
        out = x1 + x2
        return out


class Block_fusion(nn.Module):
    def __init__(self, dim, token_num, kernal_length = 31, dropout = 0.5 , mlp_ratio= 2., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Multi_Scale_Convformer(token_num, dim, kernal_length, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_fre(nn.Module):
    def __init__(self, dim, token_num, kernal_length = 31, dropout = 0.5, mlp_ratio= 2., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Convformer(token_num, dim, kernal_length, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_time(nn.Module):
    def __init__(self, dim, token_num, kernal_length = 31, dropout = 0.5, mlp_ratio= 2., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Multi_Level_Convformer(token_num, dim, kernal_length, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT_MTFNet(nn.Module):
    def __init__(self, depth_L, depth_M, time_points, frequency_components, attention_kernal_length, chs_num, class_num, dropout):
        super().__init__()
        self.token_num = chs_num * 2
        self.frequency_dim = frequency_components
        self.time_dim = time_points

        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(chs_num, self.token_num, 1, padding=1 // 2, groups=1),
            nn.LayerNorm(self.time_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.to_patch_embedding_fre = nn.Sequential(
            nn.Conv1d(chs_num, self.token_num, 1, padding=1 // 2, groups=1),
            nn.LayerNorm(self.frequency_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.transformer = nn.Sequential(*[
            Block_time(dim = self.time_dim,
                  token_num = self.token_num,
                  kernal_length = attention_kernal_length,
                dropout = dropout
            ) for i in range(depth_L)])

        self.transformer_fre = nn.Sequential(*[
            Block_fre(dim = self.frequency_dim,
                  token_num=self.token_num,
                  kernal_length=attention_kernal_length,
                  dropout=dropout
                  ) for i in range(depth_L)])

        self.transformer_fusion = nn.Sequential(*[
            Block_fusion(dim=(self.time_dim + self.frequency_dim),
                  token_num=self.token_num,
                  kernal_length=attention_kernal_length,
                  dropout=dropout
                  ) for i in range(depth_M)])

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear((self.time_dim + self.frequency_dim) * self.token_num, class_num * 6),
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        )

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)


    def forward(self, x, x_fre):

        x = self.to_patch_embedding(x)
        x_fre = self.to_patch_embedding_fre(x_fre)
        x = self.transformer(x)
        x_fre = self.transformer_fre(x_fre)
        xy_cat = torch.cat((x, x_fre), axis=2)
        xy = self.transformer_fusion(xy_cat)
        out = self.mlp_head(xy)

        return out


if __name__ == "__main__":

    X = torch.randn(64, 9, 250)
    Y = torch.randn(64, 9, 560)
    model = ViT_MTFNet(depth_L = 2, depth_M = 2, time_points = 250, frequency_components = 560, attention_kernal_length = 31, chs_num = 9, class_num = 40, dropout = 0.5)
    output = model(X, Y)
    print(output)

