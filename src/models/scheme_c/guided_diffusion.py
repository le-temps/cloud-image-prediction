"""
分割引导的条件扩散模型
用云分割 mask 作为条件信息引导扩散模型生成未来云图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


def get_beta_schedule(schedule_type, timesteps, beta_start=1e-4, beta_end=0.02):
    """生成噪声调度"""
    if schedule_type == 'linear':
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule_type == 'cosine':
        steps = timesteps + 1
        s = 0.008
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


class TimeEmbedding(nn.Module):
    """Sinusoidal 时间步嵌入"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class ConditionalResBlock(nn.Module):
    """条件残差块 with 时间嵌入和条件注入"""

    def __init__(self, in_ch, out_ch, time_dim, cond_dim=0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # 条件注入
        if cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Conv2d(cond_dim, out_ch, 1),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 1),
            )
        else:
            self.cond_proj = None

    def forward(self, x, t_emb, cond=None):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        if self.cond_proj is not None and cond is not None:
            h = h + self.cond_proj(cond)
        h = self.conv2(h)
        return h + self.skip(x)


class TemporalAttention(nn.Module):
    """时间维度注意力"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x, T):
        """
        Args:
            x: [B*T, D, H, W]
            T: 时间步数
        """
        BT, D, H, W = x.shape
        B = BT // T
        # Reshape for temporal attention
        x_res = x
        x = self.norm(x)
        x = rearrange(x, '(b t) d h w -> (b h w) t d', b=B, t=T)
        x, _ = self.attn(x, x, x)
        x = rearrange(x, '(b h w) t d -> (b t) d h w', b=B, h=H, w=W)
        return x + x_res


class SpatialAttention(nn.Module):
    """空间自注意力"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        B, D, H, W = x.shape
        x_res = x
        x = self.norm(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x, _ = self.attn(x, x, x)
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        return x + x_res


class DiffusionUNet(nn.Module):
    """
    条件时空 U-Net (噪声预测网络)
    条件: 过去帧 + 云mask序列
    """

    def __init__(self, in_channels=3, out_channels=3,
                 channels=[64, 128, 256, 512],
                 cond_channels=64, time_dim=256,
                 num_frames=10, attn_resolutions=[16, 8]):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.num_frames = num_frames
        self.attn_resolutions = attn_resolutions

        # 条件编码器: 将过去帧+mask编码为条件特征
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, cond_channels, 3, padding=1),  # +1 for mask
            nn.GELU(),
            nn.Conv2d(cond_channels, cond_channels, 3, padding=1),
        )

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = in_channels * num_frames  # 展平时间维度作为通道

        for i, ch in enumerate(channels):
            self.enc_blocks.append(
                ConditionalResBlock(prev_ch, ch, time_dim, cond_channels)
            )
            if i < len(channels) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev_ch = ch

        # Bottleneck
        self.mid_block = ConditionalResBlock(channels[-1], channels[-1], time_dim, cond_channels)
        self.mid_attn = SpatialAttention(channels[-1])

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_samples.append(nn.ConvTranspose2d(channels[i], channels[i], 2, stride=2))
            self.dec_blocks.append(
                ConditionalResBlock(channels[i] + channels[i - 1], channels[i - 1],
                                    time_dim, cond_channels)
            )

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels * num_frames, 1),
        )

    def forward(self, x_noisy, t, cond_features):
        """
        Args:
            x_noisy: [B, T*C, H, W] 加噪的目标序列 (时间展平为通道)
            t: [B] 扩散时间步
            cond_features: [B, D_cond, H, W] 条件特征
        Returns:
            noise_pred: [B, T*C, H, W] 预测噪声
        """
        t_emb = self.time_embed(t)

        # Encoder
        skips = []
        h = x_noisy
        for i, block in enumerate(self.enc_blocks):
            h = block(h, t_emb, cond_features)
            skips.append(h)
            if i < len(self.down_samples):
                h = self.down_samples[i](h)
                # 下采样条件特征
                cond_features = F.interpolate(cond_features, size=h.shape[-2:], mode='bilinear')

        # Bottleneck
        h = self.mid_block(h, t_emb, cond_features)
        h = self.mid_attn(h)

        # Decoder
        for i, (up, block) in enumerate(zip(self.up_samples, self.dec_blocks)):
            h = up(h)
            skip = skips[-(i + 2)]
            # 处理尺寸不匹配
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear')
            h = torch.cat([h, skip], dim=1)
            cond_up = F.interpolate(cond_features, size=h.shape[-2:], mode='bilinear')
            h = block(h, t_emb, cond_up)

        return self.out_conv(h)


class GuidedDiffusion(nn.Module):
    """
    云分割引导的扩散模型
    训练: 对目标序列加噪，学习预测噪声
    推理: 从纯噪声开始，在条件引导下逐步去噪
    """

    def __init__(self, in_channels=3, img_size=128, num_frames=10,
                 channels=[64, 128, 256, 512], cond_channels=64,
                 timesteps=1000, beta_schedule='cosine',
                 inference_steps=50):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_frames = num_frames
        self.timesteps = timesteps
        self.inference_steps = inference_steps

        # 噪声调度
        betas = get_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        # 条件编码器
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, cond_channels, 3, padding=1),  # img + mask
            nn.GELU(),
            nn.Conv2d(cond_channels, cond_channels, 3, padding=1),
        )

        # 噪声预测网络
        self.unet = DiffusionUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            cond_channels=cond_channels,
            num_frames=num_frames,
        )

    def q_sample(self, x_start, t, noise=None):
        """前向扩散: 给 x_0 加噪得到 x_t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def forward(self, input_frames, input_masks, target_frames):
        """
        训练前向: 预测噪声
        Args:
            input_frames: [B, T_in, C, H, W] 条件输入
            input_masks: [B, T_in, 1, H, W] 输入云mask
            target_frames: [B, T_out, C, H, W] 目标未来帧
        Returns:
            loss: 噪声预测 MSE loss
        """
        B = input_frames.shape[0]

        # 编码条件: 取最后一帧作为主条件
        last_frame = input_frames[:, -1]  # [B, C, H, W]
        last_mask = input_masks[:, -1]    # [B, 1, H, W]
        cond = self.cond_encoder(torch.cat([last_frame, last_mask], dim=1))

        # 展平目标序列
        x_start = rearrange(target_frames, 'b t c h w -> b (t c) h w')

        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device)

        # 加噪
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # 预测噪声
        noise_pred = self.unet(x_noisy, t, cond)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, input_frames, input_masks, num_samples=1):
        """
        推理: DDIM 采样生成未来帧
        Args:
            input_frames: [B, T_in, C, H, W]
            input_masks: [B, T_in, 1, H, W]
            num_samples: 采样次数 (用于概率预测)
        Returns:
            samples: [num_samples, B, T_out, C, H, W]
        """
        B, T_in, C, H, W = input_frames.shape
        device = input_frames.device

        # 编码条件
        last_frame = input_frames[:, -1]
        last_mask = input_masks[:, -1]
        cond = self.cond_encoder(torch.cat([last_frame, last_mask], dim=1))

        all_samples = []
        for _ in range(num_samples):
            # 从纯噪声开始
            x = torch.randn(B, self.num_frames * C, H, W, device=device)

            # DDIM 采样步
            step_size = self.timesteps // self.inference_steps
            timesteps = list(range(0, self.timesteps, step_size))[::-1]

            for i, t_val in enumerate(timesteps):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                noise_pred = self.unet(x, t, cond)

                # DDIM update
                alpha_t = self.alphas_cumprod[t_val]
                if i + 1 < len(timesteps):
                    alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
                else:
                    alpha_prev = torch.tensor(1.0, device=device)

                pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                x = torch.sqrt(alpha_prev) * pred_x0 + \
                    torch.sqrt(1 - alpha_prev) * noise_pred

            # 恢复序列形状
            x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.num_frames, c=C)
            x = torch.clamp((x + 1) / 2, 0, 1)  # [-1,1] → [0,1]
            all_samples.append(x)

        return torch.stack(all_samples, dim=0)  # [N, B, T_out, C, H, W]
