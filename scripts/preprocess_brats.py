#!/usr/bin/env python3
"""
BraTS2020/2021 数据预处理脚本
用于SAM-Med3D训练

作者: Leo (基于SAM-Med3D要求)
日期: 2026-02
"""

import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import argparse


def resample_volume(volume, target_spacing=(1.5, 1.5, 1.5)):
    """
    重采样到目标spacing
    
    Args:
        volume: SimpleITK图像
        target_spacing: 目标spacing (z, y, x)
    
    Returns:
        重采样后的SimpleITK图像
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    
    new_size = [
        int(round(osz * ospc / nspc)) 
        for osz, ospc, nspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkLinear)
    
    return resample.Execute(volume)


def resample_mask(mask, target_spacing=(1.5, 1.5, 1.5)):
    """
    重采样mask（使用最近邻插值保持标签完整性）
    
    Args:
        mask: SimpleITK mask图像
        target_spacing: 目标spacing
    
    Returns:
        重采样后的SimpleITK mask图像
    """
    original_spacing = mask.GetSpacing()
    original_size = mask.GetSize()
    
    new_size = [
        int(round(osz * ospc / nspc)) 
        for osz, ospc, nspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(mask.GetDirection())
    resample.SetOutputOrigin(mask.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    
    return resample.Execute(mask)


def normalize_intensity(img_array, lower_percentile=0.5, upper_percentile=99.5):
    """
    强度归一化到[0,255]
    
    使用百分位数截断，避免离群值影响
    
    Args:
        img_array: numpy数组
        lower_percentile: 下百分位数
        upper_percentile: 上百分位数
    
    Returns:
        归一化后的uint8数组
    """
    # 只计算前景区域的百分位数
    foreground = img_array[img_array > 0]
    if len(foreground) == 0:
        return np.zeros_like(img_array, dtype=np.uint8)
    
    p_low = np.percentile(foreground, lower_percentile)
    p_high = np.percentile(foreground, upper_percentile)
    
    # 截断
    img_array = np.clip(img_array, p_low, p_high)
    
    # 归一化到[0, 255]
    img_array = (img_array - p_low) / (p_high - p_low + 1e-8) * 255
    
    return img_array.astype(np.uint8)


def convert_brats_to_binary(seg_array, label_type='WT'):
    """
    将BraTS多类分割转换为二值mask
    
    BraTS标签定义:
    - 0: 背景
    - 1: Necrotic/Non-enhancing Tumor (NCR/NET) - 坏死/非增强肿瘤
    - 2: Peritumoral Edema (ED) - 瘤周水肿
    - 4: GD-enhancing Tumor (ET) - 增强肿瘤
    
    标准分割区域:
    - Whole Tumor (WT): 包含所有肿瘤区域 (1+2+4)
    - Tumor Core (TC): 肿瘤核心 (1+4)
    - Enhancing Tumor (ET): 仅增强区域 (4)
    
    Args:
        seg_array: 分割numpy数组
        label_type: 'WT', 'TC', 或 'ET'
    
    Returns:
        二值mask数组
    """
    if label_type == 'WT':
        # Whole Tumor: 所有非零标签
        binary = (seg_array > 0).astype(np.uint8)
    elif label_type == 'TC':
        # Tumor Core: 标签1和4
        binary = ((seg_array == 1) | (seg_array == 4)).astype(np.uint8)
    elif label_type == 'ET':
        # Enhancing Tumor: 仅标签4
        binary = (seg_array == 4).astype(np.uint8)
    else:
        raise ValueError(f"未知的label_type: {label_type}，应为 'WT', 'TC', 或 'ET'")
    
    return binary


def process_single_case(case_path, images_dir, labels_dir, modality, label_type, target_spacing):
    """
    处理单个病例
    
    Args:
        case_path: 病例目录路径
        images_dir: 输出图像目录
        labels_dir: 输出标签目录
        modality: MRI模态
        label_type: 标签类型
        target_spacing: 目标spacing
    
    Returns:
        (success, message)
    """
    case_name = os.path.basename(case_path)
    
    # 构建文件路径
    img_path = os.path.join(case_path, f'{case_name}_{modality}.nii.gz')
    seg_path = os.path.join(case_path, f'{case_name}_seg.nii.gz')
    
    # 检查文件是否存在
    if not os.path.exists(img_path):
        return False, f"图像文件不存在: {img_path}"
    if not os.path.exists(seg_path):
        return False, f"分割文件不存在: {seg_path}"
    
    try:
        # 读取图像
        img_sitk = sitk.ReadImage(img_path)
        seg_sitk = sitk.ReadImage(seg_path)
        
        # 重采样
        img_resampled = resample_volume(img_sitk, target_spacing=target_spacing)
        seg_resampled = resample_mask(seg_sitk, target_spacing=target_spacing)
        
        # 获取numpy数组
        img_array = sitk.GetArrayFromImage(img_resampled)
        seg_array = sitk.GetArrayFromImage(seg_resampled)
        
        # 强度归一化
        img_normalized = normalize_intensity(img_array)
        
        # 转换为二值mask
        binary_mask = convert_brats_to_binary(seg_array, label_type)
        
        # 检查mask是否有效（至少100个前景体素）
        foreground_voxels = binary_mask.sum()
        if foreground_voxels < 100:
            return False, f"mask太小 ({foreground_voxels} voxels)"
        
        # 构建输出文件名
        output_name = f'{case_name}_{label_type}.nii.gz'
        
        # 保存图像
        img_out = sitk.GetImageFromArray(img_normalized.astype(np.float32))
        img_out.SetSpacing(target_spacing)
        sitk.WriteImage(img_out, os.path.join(images_dir, output_name))
        
        # 保存标签
        seg_out = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
        seg_out.SetSpacing(target_spacing)
        sitk.WriteImage(seg_out, os.path.join(labels_dir, output_name))
        
        return True, f"成功处理，前景体素: {foreground_voxels}"
        
    except Exception as e:
        return False, f"处理错误: {str(e)}"


def process_brats_dataset(input_dir, output_dir, modality='flair', label_type='WT', 
                          target_spacing=(1.5, 1.5, 1.5)):
    """
    处理整个BraTS数据集
    
    Args:
        input_dir: 原始数据目录
        output_dir: 输出目录
        modality: 使用的MRI模态 ('flair', 't1', 't1ce', 't2')
        label_type: 标签类型 ('WT', 'TC', 'ET')
        target_spacing: 目标spacing
    """
    print(f"\n{'='*60}")
    print(f"BraTS 数据预处理")
    print(f"{'='*60}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模态: {modality}")
    print(f"标签类型: {label_type}")
    print(f"目标spacing: {target_spacing}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    images_dir = os.path.join(output_dir, 'imagesTr')
    labels_dir = os.path.join(output_dir, 'labelsTr')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 查找所有病例（支持BraTS2020和BraTS2021命名格式）
    cases = []
    for pattern in ['BraTS20_*', 'BraTS21_*', 'BraTS2020_*', 'BraTS2021_*']:
        cases.extend(glob(os.path.join(input_dir, pattern)))
    cases = sorted(set(cases))
    
    if len(cases) == 0:
        print(f"错误: 未找到任何病例！请检查输入目录: {input_dir}")
        return
    
    print(f"找到 {len(cases)} 个病例\n")
    
    # 统计
    success_count = 0
    fail_count = 0
    
    # 处理每个病例
    for case_path in tqdm(cases, desc="处理病例"):
        success, message = process_single_case(
            case_path, images_dir, labels_dir, 
            modality, label_type, target_spacing
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            tqdm.write(f"  跳过 {os.path.basename(case_path)}: {message}")
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"成功: {success_count}")
    print(f"失败/跳过: {fail_count}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")


def split_dataset(data_dir, train_ratio=0.8, seed=42):
    """
    将数据划分为训练集和验证集
    
    Args:
        data_dir: 数据目录（包含imagesTr和labelsTr）
        train_ratio: 训练集比例
        seed: 随机种子
    """
    import random
    import shutil
    
    random.seed(seed)
    
    images_dir = os.path.join(data_dir, 'imagesTr')
    labels_dir = os.path.join(data_dir, 'labelsTr')
    
    # 获取所有文件
    images = sorted(glob(os.path.join(images_dir, '*.nii.gz')))
    
    if len(images) == 0:
        print("错误: 未找到任何图像文件!")
        return
    
    # 随机打乱
    random.shuffle(images)
    
    # 划分
    n_train = int(len(images) * train_ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]
    
    # 创建验证集目录
    val_images_dir = os.path.join(data_dir, 'imagesTs')
    val_labels_dir = os.path.join(data_dir, 'labelsTs')
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 移动验证集文件
    for img_path in val_images:
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, filename)
        
        if os.path.exists(label_path):
            shutil.move(img_path, os.path.join(val_images_dir, filename))
            shutil.move(label_path, os.path.join(val_labels_dir, filename))
    
    print(f"\n数据划分完成!")
    print(f"训练集: {len(train_images)} 个样本")
    print(f"验证集: {len(val_images)} 个样本")


def main():
    parser = argparse.ArgumentParser(
        description='BraTS数据预处理脚本 for SAM-Med3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理BraTS2020 Whole Tumor
  python preprocess_brats.py --input data/raw/BraTS2020 --output data/preprocessed/BraTS2020_WT --modality flair --label WT
  
  # 处理BraTS2020 Tumor Core
  python preprocess_brats.py --input data/raw/BraTS2020 --output data/preprocessed/BraTS2020_TC --modality t1ce --label TC
  
  # 划分数据集
  python preprocess_brats.py --split data/preprocessed/BraTS2020_WT --train_ratio 0.8
        """
    )
    
    # 预处理参数
    parser.add_argument('--input', type=str, help='输入数据目录')
    parser.add_argument('--output', type=str, help='输出数据目录')
    parser.add_argument('--modality', type=str, default='flair',
                        choices=['flair', 't1', 't1ce', 't2'],
                        help='MRI模态 (默认: flair)')
    parser.add_argument('--label', type=str, default='WT',
                        choices=['WT', 'TC', 'ET'],
                        help='标签类型: WT=Whole Tumor, TC=Tumor Core, ET=Enhancing Tumor (默认: WT)')
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.5, 1.5, 1.5],
                        help='目标spacing (默认: 1.5 1.5 1.5)')
    
    # 数据划分参数
    parser.add_argument('--split', type=str, help='要划分的数据目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 数据划分模式
    if args.split:
        split_dataset(args.split, args.train_ratio, args.seed)
        return
    
    # 预处理模式
    if not args.input or not args.output:
        parser.print_help()
        print("\n错误: 请提供 --input 和 --output 参数，或使用 --split 进行数据划分")
        return
    
    target_spacing = tuple(args.spacing)
    process_brats_dataset(
        args.input, 
        args.output, 
        args.modality, 
        args.label,
        target_spacing
    )
    
    # 询问是否划分数据集
    response = input("\n是否现在划分训练/验证集? (y/n): ").strip().lower()
    if response == 'y':
        split_dataset(args.output, args.train_ratio, args.seed)


if __name__ == '__main__':
    main()
