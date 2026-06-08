import torch
import torch.nn as nn
from einops import rearrange

class TransformerBlock(nn.Module):
    """Transformer Block for global context modeling"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x

class FlareAwareAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.component_channels = channels // 2

        self.flare_detector = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, self.component_channels, 1)
        )

        self.background_detector = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, self.component_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        global_feat = self.global_pool(x)

        flare_att = self.flare_detector(global_feat)
        background_att = self.background_detector(global_feat)

        return flare_att, background_att

class FeatureDisentanglement(nn.Module):
    """Separate flare and background features."""
    def __init__(self, in_channels):
        super().__init__()

        self.flare_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.background_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def orthogonal_loss(self, flare_feat, background_feat):
        """Penalize correlation between flare and background features."""

        flare_flat = flare_feat.view(flare_feat.size(0), flare_feat.size(1), -1)
        background_flat = background_feat.view(background_feat.size(0), background_feat.size(1), -1)

        flare_mean = torch.mean(flare_flat, dim=2, keepdim=True)
        background_mean = torch.mean(background_flat, dim=2, keepdim=True)

        flare_centered = flare_flat - flare_mean
        background_centered = background_flat - background_mean

        cross_corr = torch.bmm(flare_centered, background_centered.transpose(1, 2))

        orthogonal_loss = torch.mean(torch.abs(cross_corr))

        return orthogonal_loss

    def forward(self, x):

        flare_feat = self.flare_branch(x)
        background_feat = self.background_branch(x)

        orth_loss = self.orthogonal_loss(flare_feat, background_feat)

        return flare_feat, background_feat, orth_loss

class UNetEncoder(nn.Module):
    """UNet encoder with dilated convolution blocks."""
    def __init__(self, in_channels, base_channels=64):
        super().__init__()

        self.enc1 = self._make_encoder_block_dilated(in_channels, base_channels, dilation=1)
        self.enc2 = self._make_encoder_block_dilated(base_channels, base_channels * 2, dilation=2)
        self.enc3 = self._make_encoder_block_dilated(base_channels * 2, base_channels * 4, dilation=4)
        self.enc4 = self._make_encoder_block_dilated(base_channels * 4, base_channels * 8, dilation=2)

        self.pool = nn.MaxPool2d(2)

    def _make_encoder_block_dilated(self, in_channels, out_channels, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        enc1 = self.enc1(x)

        enc2_input = self.pool(enc1)
        enc2 = self.enc2(enc2_input)

        enc3_input = self.pool(enc2)
        enc3 = self.enc3(enc3_input)

        enc4_input = self.pool(enc3)
        enc4 = self.enc4(enc4_input)

        return [enc1, enc2, enc3, enc4]

class UNetDecoder(nn.Module):
    """UNet decoder with transposed-convolution upsampling."""
    def __init__(self, base_channels=64):
        super().__init__()

        self.upconv4 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )

        self.dec4 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        self.dec3 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_decoder_block(base_channels * 2, base_channels)

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, bottleneck, encoder_features):

        enc1, enc2, enc3 = encoder_features

        up4 = self.upconv4(bottleneck)
        up4 = torch.cat([up4, enc3], dim=1)
        dec4 = self.dec4(up4)

        up3 = self.upconv3(dec4)
        up3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.dec3(up3)

        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, enc1], dim=1)
        dec2 = self.dec2(up2)

        return dec2

class TransformerBottleneck(nn.Module):
    """Transformer bottleneck with 2D positional embeddings."""
    def __init__(self, channels, num_heads=8, num_layers=4, max_height=128, max_width=128):
        super().__init__()
        self.channels = channels
        self.max_height = max_height
        self.max_width = max_width

        self.to_patch_embedding = nn.Conv2d(channels, channels, 1)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(channels, num_heads)
            for _ in range(num_layers)
        ])

        self.pos_embedding_h = nn.Parameter(torch.randn(1, max_height, 1, channels))
        self.pos_embedding_w = nn.Parameter(torch.randn(1, 1, max_width, channels))

        self.output_projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        if H > self.max_height or W > self.max_width:
            raise ValueError(
                f"Feature map size {H}x{W} exceeds maximum supported size "
                f"{self.max_height}x{self.max_width}"
            )

        x = self.to_patch_embedding(x)

        x_seq = rearrange(x, 'b c h w -> b (h w) c')

        pos_h = self.pos_embedding_h[:, :H, :, :]
        pos_w = self.pos_embedding_w[:, :, :W, :]

        pos_2d = pos_h + pos_w
        pos_seq = rearrange(pos_2d, '1 h w c -> 1 (h w) c')

        x_seq = x_seq + pos_seq

        for transformer_layer in self.transformer_layers:
            x_seq = transformer_layer(x_seq)

        x = rearrange(x_seq, 'b (h w) c -> b c h w', h=H, w=W)

        x = self.output_projection(x)

        return x

class FlareDisentanglementNetwork(nn.Module):
    """Flare disentanglement network used by LUCID."""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.encoder = UNetEncoder(in_channels, base_channels)

        self.transformer_bottleneck = TransformerBottleneck(
            base_channels * 8, num_heads=8, num_layers=4
        )

        self.decoder = UNetDecoder(base_channels)

        self.feature_disentangle = FeatureDisentanglement(base_channels)

        self.flare_attention = FlareAwareAttention(base_channels // 2 * 2)

        self.background_head = self._make_output_head(base_channels // 2, 3, 'background')
        self.flare_head = self._make_output_head(base_channels // 2, 3, 'flare')

        self.final_fusion = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, 3, padding=1)
        )

    def _make_output_head(self, in_channels, out_channels, component_type):
        """Create an output head for one disentangled component."""
        if component_type == 'flare':

            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU()
            )
        else:

            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, out_channels, 1),
                nn.Sigmoid()
            )

    def extract_features(self, lq_image):

        ENCODER_WEIGHTS = [0.1, 0.2, 0.3, 0.4]
        COMPONENT_WEIGHTS = {
            'flare_feat': 1.0,
            'background_feat': 2.0,
            'flare_att': 0.5,
            'background_att': 0.5,
            'refined_feat': 1.5
        }

        encoder_features = self.encoder(lq_image)

        bottleneck = encoder_features[-1]
        enhanced_bottleneck = self.transformer_bottleneck(bottleneck)

        decoded_features = self.decoder(enhanced_bottleneck, encoder_features[:-1])

        flare_feat, background_feat, orth_loss = self.feature_disentangle(decoded_features)

        combined_feat = torch.cat([flare_feat, background_feat], dim=1)

        flare_att, background_att = self.flare_attention(combined_feat)

        flare_feat_attended = flare_feat * flare_att
        background_feat_attended = background_feat * background_att

        fused_feat = torch.cat([flare_feat_attended, background_feat_attended], dim=1)
        refined_feat = self.final_fusion(fused_feat)

        return {
            'encoder_features': encoder_features,
            'encoder_weights': ENCODER_WEIGHTS,
            'flare_feat': flare_feat,
            'background_feat': background_feat,
            'flare_att': flare_att,
            'background_att': background_att,
            'refined_feat': refined_feat,
            'component_weights': COMPONENT_WEIGHTS
        }

    def _background_prior(self, lq_image):
        background_prior = lq_image.amax(dim=1, keepdim=True)
        return torch.clamp(background_prior, min=1e-6, max=1.0)

    def forward(self, lq_image):

        background_prior = self._background_prior(lq_image)

        encoder_features = self.encoder(lq_image)

        bottleneck = encoder_features[-1]
        enhanced_bottleneck = self.transformer_bottleneck(bottleneck)

        decoded_features = self.decoder(enhanced_bottleneck, encoder_features[:-1])

        flare_feat, background_feat, orth_loss = self.feature_disentangle(decoded_features)

        combined_feat = torch.cat([flare_feat, background_feat], dim=1)

        flare_att, background_att = self.flare_attention(combined_feat)

        flare_feat_attended = flare_feat * flare_att
        background_feat_attended = background_feat * background_att

        fused_feat = torch.cat([flare_feat_attended, background_feat_attended], dim=1)
        refined_feat = self.final_fusion(fused_feat)

        refined_flare, refined_background = torch.split(
            refined_feat, refined_feat.size(1) // 2, dim=1
        )

        background_residual = self.background_head(refined_background)
        background_prior_3ch = background_prior.repeat(1, 3, 1, 1)
        background = torch.clamp(background_residual + background_prior_3ch, 0, 1)
        flare = self.flare_head(refined_flare)

        return background_prior, background, flare, orth_loss

def create_model(model_config=None):
    """Create a flare disentanglement model."""
    if model_config is None:
        model_config = {
            'in_channels': 3,
            'base_channels': 64
        }

    model = FlareDisentanglementNetwork(**model_config)
    return model
