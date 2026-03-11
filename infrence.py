import os
import math
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torchvision.transforms as transforms
from skimage import morphology, filters, segmentation
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn

# -------------------- 全局设置 --------------------
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像预处理：直接转 Tensor
to_tensor = transforms.Compose([transforms.ToTensor()])


# -------------------- 图像处理工具 --------------------
def pad_image_to_multiple(img: Image.Image, multiple=64):
    """将图像填充到指定倍数的尺寸"""
    w, h = img.size
    w_pad = math.ceil(w / multiple) * multiple
    h_pad = math.ceil(h / multiple) * multiple
    delta_w = w_pad - w
    delta_h = h_pad - h
    return ImageOps.expand(img, border=(0, 0, delta_w, delta_h), fill=0), w, h


def read_and_preprocess(image_path: str, target_size=64):
    """读取并预处理图像"""
    try:
        with rasterio.open(image_path) as ds:
            if ds.count >= 3:
                arr = ds.read([1, 2, 3])
                arr = np.transpose(np.uint8(arr), (1, 2, 0))
                img = Image.fromarray(arr)
            else:
                band = np.uint8(np.squeeze(ds.read(1)))
                img = Image.fromarray(band).convert('RGB')
        return img
    except Exception as e:
        print(f"读取图像 {image_path} 时出错: {e}")
        img = Image.open(image_path).convert('RGB')
        return img


def safe_filename(name):
    """生成安全的文件名"""
    import re
    name = re.sub(r'[<>:"/\\|?*.]', '_', name)
    return name[:100]


def preprocess_for_model(tensor, target_size=64):
    """预处理张量以适应模型尺寸要求"""
    _, _, h, w = tensor.shape

    # 强制调整到模型期望的尺寸
    if h != target_size or w != target_size:
        tensor = F.interpolate(tensor, size=(target_size, target_size),
                               mode='bilinear', align_corners=False)
        print(f"调整输入尺寸至: {target_size}x{target_size}")

    return tensor


# -------------------- 调试函数 --------------------
def debug_model_structure(model):
    """调试函数：打印模型的所有层名"""
    print("=" * 60)
    print("模型层结构调试信息:")
    print("=" * 60)

    # 打印所有层
    all_layers = []
    for name, module in model.named_modules():
        if name:  # 只打印有名称的模块
            all_layers.append(name)
            print(f"  {name}: {type(module).__name__}")

    # 特别检查EfficientMSA相关层
    print("\nEfficientMSA相关层:")
    emsa_layers = [name for name in all_layers if 'emsa' in name.lower()]
    for layer in emsa_layers:
        print(f"  ✓ {layer}")

    # 检查是否还有PCM残留
    pcm_layers = [name for name in all_layers if 'pcm' in name.lower()]
    if pcm_layers:
        print("\n警告: 发现PCM相关层 (可能来自旧权重):")
        for layer in pcm_layers:
            print(f"  ⚠️ {layer}")
    else:
        print("\n✓ 未发现PCM相关层")

    print("=" * 60)
    return all_layers


# -------------------- 完整特征可视化函数 --------------------
def complete_feature_visualization(img1_path, img2_path, model_path, save_dir):
    """完整的特征可视化 - 适配你的新模型结构（使用EfficientMSA）"""

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "encoder_features"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "decoder_features"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "attention_maps"), exist_ok=True)

    # 加载模型
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    model.to(device).eval()

    # 调试模型结构
    all_layers = debug_model_structure(model)

    # 读取图像
    imgA = read_and_preprocess(img1_path)
    imgB = read_and_preprocess(img2_path)

    print(f"原始图像尺寸 - A: {imgA.size}, B: {imgB.size}")

    # 选择分析区域（中心区域或完整图像）
    w, h = imgA.size
    if w <= 512 and h <= 512:
        patchA = imgA
        patchB = imgB
        print("使用完整图像进行分析")
    else:
        # 选择中心区域
        center_x, center_y = w // 2, h // 2
        crop_size = min(512, w, h)
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(w, left + crop_size)
        bottom = min(h, top + crop_size)

        patchA = imgA.crop((left, top, right, bottom))
        patchB = imgB.crop((left, top, right, bottom))
        print(f"使用中心区域进行分析: {crop_size}x{crop_size}")

    # 保存原始patch用于可视化
    original_patchA = patchA.copy()
    original_patchB = patchB.copy()

    # 调整到模型兼容的尺寸
    patchA = patchA.resize((64, 64), Image.BILINEAR)
    patchB = patchB.resize((64, 64), Image.BILINEAR)

    # 转换为Tensor
    tA = to_tensor(patchA).unsqueeze(0).to(device)
    tB = to_tensor(patchB).unsqueeze(0).to(device)

    print("开始完整特征分析...")

    # 存储所有中间特征
    feature_maps = {}
    hooks = []

    def register_hook(module, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                feature_maps[name] = output[0].detach()
            else:
                feature_maps[name] = output.detach()

        return hook

    # 注册所有重要层的钩子 - 使用你的新模型结构
    layers_to_monitor = [
        # 编码器部分
        'inc',  # 输入卷积
        'emsa1', 'emsa2', 'emsa3', 'emsa4',  # 替换的EfficientMSA模块
        'FE1', 'FE2', 'FE3',  # 特征提取层

        # 空间注意力模块
        'SPM1', 'SPM2', 'SPM3', 'SPM4',

        # 跨模态Transformer
        'SFM',  # CrossTransformer

        # 特征融合
        'fusion1', 'fusion2', 'fusion3', 'fusion4',

        # 解码器部分
        'double2single1', 'double2single2', 'double2single3',
        'sce1', 'sce2', 'sce3',

        # 输出层
        'out'
    ]

    registered_count = 0
    for name, module in model.named_modules():
        if name in layers_to_monitor:  # 精确匹配，避免部分匹配问题
            hook = module.register_forward_hook(register_hook(module, name))
            hooks.append(hook)
            registered_count += 1
            print(f"✓ 注册层: {name}")

    print(f"总共注册了 {registered_count} 个层进行特征监控")

    try:
        # 前向传播
        with torch.no_grad():
            start_time = time.time()
            output = model(tA, tB)
            inference_time = time.time() - start_time
            print(f"✓ 前向传播完成，耗时: {inference_time:.2f}秒")

        # 可视化最终输出
        visualize_final_output(output, original_patchA, original_patchB, save_dir)

        # 可视化所有中间特征
        visualize_all_features(feature_maps, original_patchA, save_dir)

        # 生成特征分析报告
        generate_feature_report(feature_maps, save_dir)

        print("✓ 完整特征分析完成!")

    except Exception as e:
        print(f"✗ 特征分析失败: {e}")
        import traceback
        traceback.print_exc()

        # 如果完整分析失败，尝试简化分析
        print("尝试简化特征分析...")
        simplified_feature_analysis(model, original_patchA, original_patchB, save_dir)

    finally:
        # 移除钩子
        for hook in hooks:
            hook.remove()


def simplified_feature_analysis(model, patchA, patchB, save_dir):
    """简化特征分析 - 绕过可能出问题的模块"""
    try:
        # 调整到64x64
        patchA_small = patchA.resize((64, 64), Image.BILINEAR)
        patchB_small = patchB.resize((64, 64), Image.BILINEAR)

        tA = to_tensor(patchA_small).unsqueeze(0).to(device)
        tB = to_tensor(patchB_small).unsqueeze(0).to(device)

        with torch.no_grad():
            # 只进行编码器前向传播
            opt_features = model.encoder(tA)
            sar_features = model.encoder(tB)

            # 可视化编码器特征
            for i, (opt_feat, sar_feat) in enumerate(zip(opt_features, sar_features)):
                # 可视化光学特征
                if opt_feat.dim() == 4:
                    feat_map = torch.mean(opt_feat[0], dim=0).cpu().numpy()
                    if np.ptp(feat_map) > 1e-8:
                        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

                        # 上采样到原始尺寸
                        original_size = patchA.size[::-1]  # (height, width)
                        feat_map_upscaled = F.interpolate(
                            torch.from_numpy(feat_map).unsqueeze(0).unsqueeze(0),
                            size=original_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().numpy()

                        create_simple_feature_visualization(
                            np.array(patchA), feat_map_upscaled,
                            f"encoder_opt_{i}", save_dir, i
                        )

            print("✓ 简化特征分析完成")

    except Exception as e:
        print(f"✗ 简化特征分析也失败: {e}")


def visualize_final_output(output, patchA, patchB, save_dir):
    """可视化最终输出"""
    print("可视化最终输出...")

    if output.dim() == 4:
        # 取第一个batch，第一个通道
        final_feature = output[0, 0].cpu().numpy()

        # 归一化
        final_feature = (final_feature - final_feature.min()) / (final_feature.max() - final_feature.min() + 1e-8)

        # 上采样到原始patch尺寸
        original_size = patchA.size[::-1]  # (height, width)
        final_feature_upscaled = F.interpolate(
            torch.from_numpy(final_feature).unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # 创建高质量可视化
        create_comprehensive_visualization(
            np.array(patchA), np.array(patchB), final_feature_upscaled,
            save_dir, "final_output"
        )

        print("✓ 最终输出可视化完成")


def visualize_all_features(feature_maps, original_img, save_dir):
    """可视化所有中间特征"""
    print("可视化中间特征...")

    original_array = np.array(original_img)
    original_size = original_img.size[::-1]  # (height, width)

    for i, (layer_name, feature) in enumerate(feature_maps.items()):
        try:
            if feature.dim() == 4 and feature.size(2) > 1 and feature.size(3) > 1:
                # 处理4D特征图
                feature = feature[0]  # 第一个batch

                if feature.size(0) > 0:
                    # 计算通道平均
                    if feature.size(0) <= 3:
                        # 如果通道数<=3，直接显示
                        if feature.size(0) == 1:
                            heatmap = feature[0].cpu().numpy()
                        else:
                            heatmap = torch.mean(feature, dim=0).cpu().numpy()
                    else:
                        # 多通道特征图，取平均
                        heatmap = torch.mean(feature, dim=0).cpu().numpy()

                    if np.ptp(heatmap) > 1e-8:
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                        # 上采样到原始尺寸
                        heatmap_upscaled = F.interpolate(
                            torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0),
                            size=original_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().numpy()

                        # 创建特征可视化
                        create_feature_visualization(
                            original_array, heatmap_upscaled, layer_name,
                            os.path.join(save_dir, "encoder_features"), i
                        )

                        if i % 5 == 0:  # 每5层打印一次进度
                            print(f"✓ 已处理 {i + 1}/{len(feature_maps)} 个特征图")

        except Exception as e:
            print(f"✗ 处理层 {layer_name} 时出错: {e}")

    print("✓ 所有中间特征可视化完成")


def create_comprehensive_visualization(opt_img, sar_img, attention_map, save_dir, prefix):
    """创建全面的可视化"""
    dpi = 300

    # 主分析图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=dpi)

    # 第一行：基础可视化
    axes[0, 0].imshow(opt_img)
    axes[0, 0].set_title('光学图像', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sar_img, cmap='gray')
    axes[0, 1].set_title('SAR图像', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    im1 = axes[0, 2].imshow(attention_map, cmap='jet', interpolation='lanczos')
    axes[0, 2].set_title('关注度热图', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 第二行：分析图
    # 叠加效果
    axes[1, 0].imshow(opt_img)
    axes[1, 0].imshow(attention_map, cmap='jet', alpha=0.6, interpolation='lanczos')
    axes[1, 0].set_title('光学+关注度叠加', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # 阈值分析
    binary_mask = attention_map > 0.5
    coverage = np.sum(binary_mask) / binary_mask.size * 100
    axes[1, 1].imshow(opt_img)
    axes[1, 1].imshow(binary_mask, cmap='Reds', alpha=0.6)
    axes[1, 1].set_title(f'阈值分割 (0.5)\n覆盖度: {coverage:.1f}%', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # 统计分布
    axes[1, 2].hist(attention_map.flatten(), bins=50, alpha=0.7, color='steelblue',
                    edgecolor='black', linewidth=0.5)
    axes[1, 2].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='阈值 0.5')
    axes[1, 2].set_xlabel('关注度值', fontsize=10)
    axes[1, 2].set_ylabel('频率', fontsize=10)
    axes[1, 2].set_title('值分布统计', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_comprehensive.png"),
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    # 创建详细的多阈值分析
    create_multi_threshold_analysis(opt_img, sar_img, attention_map, save_dir, prefix)


def create_feature_visualization(original_img, feature_map, layer_name, save_dir, index):
    """创建单个特征图的可视化"""
    dpi = 300

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=dpi)

    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('输入图像', fontsize=10)
    axes[0].axis('off')

    # 特征热图
    im = axes[1].imshow(feature_map, cmap='jet', interpolation='lanczos')
    axes[1].set_title(f'特征热图', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 叠加显示
    axes[2].imshow(original_img)
    axes[2].imshow(feature_map, cmap='jet', alpha=0.7, interpolation='lanczos')
    axes[2].set_title('叠加显示', fontsize=10)
    axes[2].axis('off')

    # 统计信息
    axes[3].hist(feature_map.flatten(), bins=50, alpha=0.7, color='green')
    axes[3].set_xlabel('特征值', fontsize=8)
    axes[3].set_ylabel('频率', fontsize=8)
    axes[3].set_title('值分布', fontsize=9)
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f'层: {layer_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    safe_name = safe_filename(layer_name)
    plt.savefig(os.path.join(save_dir, f"feature_{index:03d}_{safe_name}.png"),
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_simple_feature_visualization(original_img, feature_map, layer_name, save_dir, index):
    """创建简化特征可视化"""
    dpi = 300

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('输入图像', fontsize=10)
    axes[0].axis('off')

    # 特征热图
    im = axes[1].imshow(feature_map, cmap='jet', interpolation='lanczos')
    axes[1].set_title(f'特征热图', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 叠加显示
    axes[2].imshow(original_img)
    axes[2].imshow(feature_map, cmap='jet', alpha=0.7, interpolation='lanczos')
    axes[2].set_title('叠加显示', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f'层: {layer_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    safe_name = safe_filename(layer_name)
    plt.savefig(os.path.join(save_dir, f"simple_feature_{index:03d}_{safe_name}.png"),
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_multi_threshold_analysis(opt_img, sar_img, attention_map, save_dir, prefix):
    """创建多阈值分析"""
    dpi = 300
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=dpi)

    thresholds = [0.3, 0.5, 0.7]
    colors = ['yellow', 'orange', 'red']

    for i, (threshold, color) in enumerate(zip(thresholds, colors)):
        binary_mask = attention_map > threshold
        coverage = np.sum(binary_mask) / binary_mask.size * 100

        # 光学图像叠加
        axes[0, i].imshow(opt_img)
        axes[0, i].imshow(binary_mask, cmap=plt.cm.colors.ListedColormap([color]),
                          alpha=0.6, interpolation='nearest')
        axes[0, i].set_title(f'阈值={threshold}\n覆盖度: {coverage:.1f}%', fontsize=10)
        axes[0, i].axis('off')

        # SAR图像叠加
        axes[1, i].imshow(sar_img, cmap='gray')
        axes[1, i].imshow(binary_mask, cmap=plt.cm.colors.ListedColormap([color]),
                          alpha=0.6, interpolation='nearest')
        axes[1, i].set_title(f'阈值={threshold}\n覆盖度: {coverage:.1f}%', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_threshold_analysis.png"),
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_feature_report(feature_maps, save_dir):
    """生成特征分析报告"""
    report_path = os.path.join(save_dir, "feature_analysis_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("HRSICD模型特征分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"监控层数量: {len(feature_maps)}\n\n")

        f.write("各层特征统计:\n")
        f.write("-" * 50 + "\n")

        for layer_name, feature in feature_maps.items():
            f.write(f"\n层名称: {layer_name}\n")
            f.write(f"  形状: {tuple(feature.shape)}\n")
            f.write(f"  数据类型: {feature.dtype}\n")
            if feature.numel() > 0:
                f.write(f"  值范围: [{feature.min():.4f}, {feature.max():.4f}]\n")
                f.write(f"  平均值: {feature.mean():.4f}\n")
                f.write(f"  标准差: {feature.std():.4f}\n")
            else:
                f.write(f"  空特征图\n")

        f.write(f"\n报告生成完成。\n")

    print(f"✓ 特征分析报告已保存: {report_path}")


# -------------------- 主函数 --------------------
def main():
    # 参数设置
    model_file = r"result/64+e/improved_iou_epoch99_model.pth"
    image1_path = r"data/keshihua/1.bmp"
    image2_path = r"data/keshihua/2.bmp"
    result_dir = r"result/CompleteFeatureAnalysis"

    print("=" * 60)
    print("开始完整特征分析")
    print("=" * 60)

    # 执行完整特征分析
    complete_feature_visualization(
        image1_path, image2_path, model_file, result_dir
    )

    print("=" * 60)
    print("特征分析完成!")
    print(f"结果保存在: {result_dir}")
    print("\n生成的内容包括:")
    print("✓ 最终输出关注度热图")
    print("✓ 所有中间层特征可视化")
    print("✓ 多阈值分析")
    print("✓ 特征统计报告")
    print("✓ 所有图像均为高清质量 (300 DPI)")
    print("=" * 60)


if __name__ == "__main__":
    main()