import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


# ======================= 基础组件保持不变 =======================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ESAM_Type1(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.reduced_channels = max(4, in_channels // reduction)
        self.branch1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        )
        self.fusion = nn.Sequential(
            nn.BatchNorm2d(3 * self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * self.reduced_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        b1, b2, b3 = self.branch1(x), self.branch2(x), self.branch3(x)
        return self.fusion(torch.cat([b1, b2, b3], dim=1))


class ESAM_Type2(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.reduced_channels = max(4, in_channels // reduction)
        self.branch1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=5, padding=2)
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        )
        self.fusion = nn.Sequential(
            nn.BatchNorm2d(3 * self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * self.reduced_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        b1, b2 = self.branch1(x), self.branch2(x)
        b3 = self.branch3(x).expand(-1, -1, x.shape[2], x.shape[3])  # 使用expand替代interpolate更高效
        return self.fusion(torch.cat([b1, b2, b3], dim=1))


# ======================= 新增：自适应通道门控融合模块 =======================

class AdaptiveChannelGateFusion(nn.Module):
    """自适应通道门控融合模块 - 替换原来的静态fusion卷积"""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels

        # 生成门控权重的网络（通道注意力机制）
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化获取全局信息 [B, C, 1, 1]
            nn.Conv2d(in_channels, max(4, in_channels // reduction), kernel_size=1),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, in_channels // reduction), in_channels, kernel_size=1),  # 恢复维度
            nn.Sigmoid()  # 输出0-1的权重 [B, C, 1, 1]
        )

        # 融合卷积（用于进一步精炼特征）
        self.refine = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, feat1, feat2):
        """
        x: 原始输入特征 [B, C, H, W]
        feat1: ESAM_Type1输出 [B, C/2, H, W]
        feat2: ESAM_Type2输出 [B, C/2, H, W]
        返回: 动态融合后的特征 [B, C, H, W]
        """
        # 拼接两个特征
        combined = torch.cat([feat1, feat2], dim=1)  # [B, C, H, W]

        # 基于原始输入x生成动态门控权重
        # 注意：这里使用x而不是combined，因为x包含原始信息
        gate_weights = self.gate_net(x)  # [B, C, 1, 1]

        # 使用门控权重动态融合（广播机制：权重自动扩展到H×W）
        gated_combined = combined * gate_weights

        # 进一步精炼特征
        refined = self.refine(gated_combined)

        # 残差连接
        return refined + x


# ======================= 其他模块保持不变 =======================

class LightweightCrossAtt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        batch_size, C, H, W = x1.size()
        proj_query = self.query_conv(x1).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x2).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x2).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        return self.gamma * out + x1, x2


class EdgeAwareDAF_Module(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.edge_detector = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Tanh()
        )
        self.fusion_network = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, opt, sar):
        opt_edge = torch.abs(self.edge_detector(opt))
        sar_edge = torch.abs(self.edge_detector(sar))
        fused = self.fusion_network(torch.cat([opt + opt_edge, sar + sar_edge], dim=1))
        return fused + (opt + sar) * 0.5


class SP_Block(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SP_Block, self).__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat = self.bn(self.conv_1x1(torch.cat((x_h, x_w), dim=3)))
        x_relu = F.relu(x_cat)
        s_h = torch.sigmoid(self.F_h(x_relu[:, :, :, :h])).permute(0, 1, 3, 2).expand(-1, -1, h, w)
        s_w = torch.sigmoid(self.F_w(x_relu[:, :, :, h:])).expand(-1, -1, h, w)
        return x * s_h * s_w + x


class CrossTransformer(nn.Module):
    def __init__(self, dropout, d_model=512, n_head=4):
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def cross(self, q, k_v):
        attn_out, _ = self.attention(q, k_v, k_v)
        q = self.norm1(q + attn_out)
        return self.norm2(q + self.linear(q))

    def forward(self, input1, input2):
        dif = input2 - input1
        return self.cross(input1, dif), self.cross(input2, dif)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class FrequencyPhaseAlignment_256(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        # 学习一个可变的混合系数，让模型决定对齐的强度
        self.mix_factor = nn.Parameter(torch.tensor(0.5))

        # 对齐后的特征精炼
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_opt, x_sar):
        # 1. 快速傅里叶变换 (RFFT 处理实数输入更高效)
        # 结果形状: [B, C, H, W//2 + 1] 且为复数类型
        fft_opt = torch.fft.rfft2(x_opt, norm='ortho')
        fft_sar = torch.fft.rfft2(x_sar, norm='ortho')

        # 2. 提取幅值（代表风格/纹理）和相位（代表结构/几何）
        amp_opt, phase_opt = fft_opt.abs(), fft_opt.angle()
        amp_sar, phase_sar = fft_sar.abs(), fft_sar.angle()

        # 3. 相位混合 (对齐结构)
        # 我们希望 SAR 在结构上向 Optical 靠拢，反之亦然
        alpha = torch.sigmoid(self.mix_factor)

        # 构造混合相位：让两者的相位互相渗透
        mixed_phase_opt = phase_opt * (1 - alpha * 0.5) + phase_sar * (alpha * 0.5)
        mixed_phase_sar = phase_sar * (1 - alpha * 0.5) + phase_opt * (alpha * 0.5)

        # 4. 逆变换回空域
        # 使用 torch.polar 根据幅值和混合后的相位重构复数
        recon_opt = torch.polar(amp_opt, mixed_phase_opt)
        recon_sar = torch.polar(amp_sar, mixed_phase_sar)

        # 逆傅里叶变换
        out_opt = torch.fft.irfft2(recon_opt, s=x_opt.shape[-2:], norm='ortho')
        out_sar = torch.fft.irfft2(recon_sar, s=x_sar.shape[-2:], norm='ortho')

        # 5. 残差连接与精炼 (保证泛化稳定性)
        opt_final = x_opt + self.refine(out_opt)
        sar_final = x_sar + self.refine(out_sar)

        return opt_final, sar_final


# ======================= 主模型 HRSICD (已集成动态门控) =======================

class HRSICD(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=64, bilinear=True):
        super(HRSICD, self).__init__()

        # Encoder 基础结构
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # ESAM 模块 (各层级) - 保持不变
        self.esam64_1, self.esam64_2 = ESAM_Type1(32), ESAM_Type2(32)
        self.esam128_1, self.esam128_2 = ESAM_Type1(64), ESAM_Type2(64)
        self.esam256_1, self.esam256_2 = ESAM_Type1(128), ESAM_Type2(128)
        self.esam512_1, self.esam512_2 = ESAM_Type1(256), ESAM_Type2(256)

        # 自适应通道门控融合模块 - 替换原来的静态fusion卷积
        self.gate_fusion64 = AdaptiveChannelGateFusion(64, reduction=8)
        self.gate_fusion128 = AdaptiveChannelGateFusion(128, reduction=8)
        self.gate_fusion256 = AdaptiveChannelGateFusion(256, reduction=8)
        self.gate_fusion512 = AdaptiveChannelGateFusion(512, reduction=8)

        # 空间注意力 SPM
        self.SPM1, self.SPM2, self.SPM3, self.SPM4 = SP_Block(64), SP_Block(128), SP_Block(256), SP_Block(512)

        # 跨模态交互
        self.cross_att_64 = LightweightCrossAtt(64)
        self.edge_aware_daf_128 = EdgeAwareDAF_Module(128)

        # 256层频域对齐
        self.freq_align_256 = FrequencyPhaseAlignment_256(256)

        self.SFM = CrossTransformer(dropout=0.1, d_model=512)

        # Decoder 融合与上采样
        self.fusion1, self.fusion2, self.fusion3, self.fusion4 = \
            DoubleConv(128, 64), DoubleConv(256, 128), DoubleConv(512, 256), DoubleConv(1024, 512)

        self.up1 = Up(512 + 256, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def esam_layer(self, x, esam1, esam2, gate_module):
        """
        使用自适应门控融合的esam_layer
        参数:
            x: 输入特征 [B, C, H, W]
            esam1: ESAM_Type1模块
            esam2: ESAM_Type2模块
            gate_module: 自适应门控融合模块
        返回:
            动态融合后的特征 [B, C, H, W]
        """
        # 将输入特征分割成两部分
        x1, x2 = torch.chunk(x, 2, dim=1)  # 各为 [B, C/2, H, W]

        # 分别通过两个ESAM模块
        feat1 = esam1(x1)  # ESAM_Type1处理结果
        feat2 = esam2(x2)  # ESAM_Type2处理结果

        # 使用自适应门控融合模块动态融合两个特征
        return gate_module(x, feat1, feat2)

    def encoder_stream(self, x):
        """编码器流 - 使用动态门控融合"""
        # 第一层: 64通道
        x1 = self.esam_layer(self.inc(x), self.esam64_1, self.esam64_2, self.gate_fusion64)
        x1_sp = self.SPM1(x1)

        # 第二层: 128通道
        x2 = self.esam_layer(self.down1(x1_sp), self.esam128_1, self.esam128_2, self.gate_fusion128)
        x2_sp = self.SPM2(x2)

        # 第三层: 256通道
        x3 = self.esam_layer(self.down2(x2_sp), self.esam256_1, self.esam256_2, self.gate_fusion256)
        x3_sp = self.SPM3(x3)

        # 第四层: 512通道
        x4 = self.esam_layer(self.down3(x3_sp), self.esam512_1, self.esam512_2, self.gate_fusion512)
        x4_sp = self.SPM4(x4)

        return [x1_sp, x2_sp, x3_sp, x4_sp]

    def forward(self, x1, x2):
        """前向传播"""
        # 编码器提取特征
        opt_feats = self.encoder_stream(x1)
        sar_feats = self.encoder_stream(x2)

        # 64层交互
        opt_feats[0], sar_feats[0] = self.cross_att_64(opt_feats[0], sar_feats[0])

        # 128层边缘对齐
        fused_128 = self.edge_aware_daf_128(opt_feats[1], sar_feats[1])
        opt_feats[1] = fused_128  # 更新特征用于skip connection

        # 256层频域相位对齐
        opt_feats[2], sar_feats[2] = self.freq_align_256(opt_feats[2], sar_feats[2])

        # 512层 Transformer 交互
        B, C, H, W = opt_feats[3].shape
        opt_f = opt_feats[3].view(B, C, -1).permute(0, 2, 1)
        sar_f = sar_feats[3].view(B, C, -1).permute(0, 2, 1)
        o_out, s_out = self.SFM(opt_f, sar_f)
        opt_feats[3] = o_out.permute(0, 2, 1).view(B, C, H, W)
        sar_feats[3] = s_out.permute(0, 2, 1).view(B, C, H, W)

        # 特征拼接与通道融合
        en_out = [
            self.fusion1(torch.cat([opt_feats[0], sar_feats[0]], 1)),
            self.fusion2(torch.cat([opt_feats[1], sar_feats[1]], 1)),
            self.fusion3(torch.cat([opt_feats[2], sar_feats[2]], 1)),
            self.fusion4(torch.cat([opt_feats[3], sar_feats[3]], 1))
        ]

        # Decoder
        de_out = self.up1(en_out[3], en_out[2])
        de_out = self.up2(de_out, en_out[1])
        de_out = self.up3(de_out, en_out[0])

        return torch.sigmoid(self.out(de_out))


# ======================= 测试脚本 =======================
if __name__ == "__main__":
    model = HRSICD(img_size=64)
    x1, x2 = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)

    print("开始测试带自适应通道门控融合的HRSICD...")
    print("模型结构:")
    print(f"  - 自适应通道门控融合模块: {model.gate_fusion64}")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    output = model(x1, x2)
    print(f"\n输出形状: {output.shape}")

    # 计算自适应门控模块的参数量
    gate_params = sum(p.numel() for p in model.gate_fusion64.parameters())
    gate_params += sum(p.numel() for p in model.gate_fusion128.parameters())
    gate_params += sum(p.numel() for p in model.gate_fusion256.parameters())
    gate_params += sum(p.numel() for p in model.gate_fusion512.parameters())
    print(f"自适应门控模块总参数量: {gate_params}")

    # 验证梯度
    output.mean().backward()
    print("梯度反向传播完成，自适应门控路径通畅。")

    # 测试动态性：验证不同输入产生不同的门控权重
    print("\n测试自适应门控的动态性...")
    with torch.no_grad():
        # 创建两个不同的输入
        test_input1 = torch.randn(1, 64, 32, 32)
        test_input2 = torch.randn(1, 64, 32, 32) * 2  # 不同的分布

        # 获取门控权重
        gate_weights1 = model.gate_fusion64.gate_net(test_input1)
        gate_weights2 = model.gate_fusion64.gate_net(test_input2)

        # 计算权重差异
        weight_diff = torch.abs(gate_weights1 - gate_weights2).mean().item()
        print(f"不同输入的门控权重平均差异: {weight_diff:.6f}")

        if weight_diff > 0.001:
            print("✓ 自适应门控机制正常工作：不同输入产生不同的权重")
        else:
            print("⚠ 注意：门控权重可能没有充分自适应")