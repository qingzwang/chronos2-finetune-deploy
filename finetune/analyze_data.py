#!/usr/bin/env python3
"""
数据质量分析脚本

统计：
1. 缺失值 (NaN) 情况
2. power 连续为 0 的片段数量

使用方法:
    python finetune/analyze_data.py --min_zeros 48 --max_files 100
    python finetune/analyze_data.py --min_zeros 96 --split train
"""

import os
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path


def load_file_list(split_csv: str, data_dir: str, split: str = None) -> List[str]:
    """加载文件列表"""
    df = pd.read_csv(split_csv)

    if split:
        df = df[df["split"] == split]

    file_paths = [os.path.join(data_dir, f) for f in df["filename"].tolist()]
    existing = [f for f in file_paths if os.path.exists(f)]

    return existing


def count_consecutive_zeros(arr: np.ndarray, min_length: int) -> Tuple[int, List[int]]:
    """
    统计连续 0 的片段数量

    Args:
        arr: 数组
        min_length: 最小连续长度阈值

    Returns:
        (片段数量, 各片段长度列表)
    """
    segments = []
    count = 0
    in_zero_segment = False
    current_length = 0

    for val in arr:
        if val == 0:
            if not in_zero_segment:
                in_zero_segment = True
                current_length = 1
            else:
                current_length += 1
        else:
            if in_zero_segment:
                if current_length >= min_length:
                    segments.append(current_length)
                in_zero_segment = False
                current_length = 0

    # 处理末尾的连续 0
    if in_zero_segment and current_length >= min_length:
        segments.append(current_length)

    return len(segments), segments


def analyze_file(file_path: str, min_zeros: int) -> Dict:
    """分析单个文件"""
    try:
        df = pd.read_csv(file_path)

        result = {
            "file": Path(file_path).name,
            "total_rows": len(df),
            "power_missing": df["power"].isna().sum(),
            "temperature_missing": df["temperature"].isna().sum(),
            "irradiance_missing": df["irradiance"].isna().sum(),
            "power_zeros": (df["power"] == 0).sum(),
            "zero_segments": 0,
            "zero_segment_lengths": [],
            "max_zero_segment": 0,
        }

        # 统计连续 0 片段
        power = df["power"].fillna(0).values
        num_segments, segment_lengths = count_consecutive_zeros(power, min_zeros)

        result["zero_segments"] = num_segments
        result["zero_segment_lengths"] = segment_lengths
        result["max_zero_segment"] = max(segment_lengths) if segment_lengths else 0

        return result

    except Exception as e:
        return {
            "file": Path(file_path).name,
            "error": str(e),
        }


def print_summary(results: List[Dict], min_zeros: int):
    """打印汇总统计"""
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    print("\n" + "=" * 70)
    print("数据质量分析报告")
    print("=" * 70)

    print(f"\n【文件统计】")
    print(f"  分析文件数: {len(results)}")
    print(f"  成功解析:   {len(valid_results)}")
    print(f"  解析失败:   {len(error_results)}")

    if not valid_results:
        print("\n没有成功解析的文件!")
        return

    # 缺失值统计
    total_rows = sum(r["total_rows"] for r in valid_results)
    power_missing = sum(r["power_missing"] for r in valid_results)
    temp_missing = sum(r["temperature_missing"] for r in valid_results)
    irr_missing = sum(r["irradiance_missing"] for r in valid_results)

    print(f"\n【缺失值统计】")
    print(f"  总数据点:       {total_rows:,}")
    print(f"  power 缺失:     {power_missing:,} ({100*power_missing/total_rows:.2f}%)")
    print(f"  temperature 缺失: {temp_missing:,} ({100*temp_missing/total_rows:.2f}%)")
    print(f"  irradiance 缺失:  {irr_missing:,} ({100*irr_missing/total_rows:.2f}%)")

    # power = 0 统计
    total_zeros = sum(r["power_zeros"] for r in valid_results)
    print(f"\n【Power = 0 统计】")
    print(f"  power=0 数据点: {total_zeros:,} ({100*total_zeros/total_rows:.2f}%)")

    # 连续 0 片段统计
    total_segments = sum(r["zero_segments"] for r in valid_results)
    files_with_segments = sum(1 for r in valid_results if r["zero_segments"] > 0)
    all_segment_lengths = []
    for r in valid_results:
        all_segment_lengths.extend(r["zero_segment_lengths"])

    print(f"\n【连续 0 片段统计】(阈值: >= {min_zeros} 个连续 0)")
    print(f"  总片段数:       {total_segments:,}")
    print(f"  含片段的文件数: {files_with_segments:,} ({100*files_with_segments/len(valid_results):.1f}%)")

    if all_segment_lengths:
        print(f"  片段长度统计:")
        print(f"    最小: {min(all_segment_lengths)}")
        print(f"    最大: {max(all_segment_lengths)}")
        print(f"    平均: {np.mean(all_segment_lengths):.1f}")
        print(f"    中位数: {np.median(all_segment_lengths):.1f}")

        # 长度分布
        print(f"\n  片段长度分布:")
        bins = [min_zeros, 96, 192, 288, 480, 960, float('inf')]
        bin_labels = [
            f"{min_zeros}-95",
            "96-191 (2-4天)",
            "192-287 (4-6天)",
            "288-479 (6-10天)",
            "480-959 (10-20天)",
            "960+ (>20天)"
        ]

        for i in range(len(bins) - 1):
            count = sum(1 for l in all_segment_lengths if bins[i] <= l < bins[i+1])
            if count > 0:
                print(f"    {bin_labels[i]}: {count} 个片段")

    # 每个文件的详细统计 (只显示有问题的)
    files_with_issues = [r for r in valid_results
                        if r["power_missing"] > 0 or r["zero_segments"] > 0]

    if files_with_issues:
        print(f"\n【问题文件详情】(前 20 个)")
        print("-" * 70)
        print(f"{'文件名':<15} {'行数':>8} {'缺失':>8} {'0值':>8} {'片段数':>8} {'最长片段':>10}")
        print("-" * 70)

        # 按问题严重程度排序
        files_with_issues.sort(key=lambda x: (x["zero_segments"], x["max_zero_segment"]), reverse=True)

        for r in files_with_issues[:20]:
            print(f"{r['file']:<15} {r['total_rows']:>8} {r['power_missing']:>8} "
                  f"{r['power_zeros']:>8} {r['zero_segments']:>8} {r['max_zero_segment']:>10}")

    # 错误文件
    if error_results:
        print(f"\n【解析失败的文件】")
        for r in error_results[:10]:
            print(f"  {r['file']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="分析数据质量：缺失值和连续 0 片段",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/efs/to_customer/zendure/data/",
        help="数据目录",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default="/home/ubuntu/efs/to_customer/zendure/zero-shot/data_split.csv",
        help="数据划分文件",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="只分析指定的数据集划分",
    )
    parser.add_argument(
        "--min_zeros",
        type=int,
        default=48,
        help="连续 0 的最小长度阈值 (48 = 24小时 @ 30分钟间隔)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="最多分析的文件数 (用于快速测试)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出详细结果到 CSV 文件",
    )

    args = parser.parse_args()

    print(f"加载文件列表...")
    file_paths = load_file_list(args.split_csv, args.data_dir, args.split)

    if args.max_files:
        file_paths = file_paths[:args.max_files]

    print(f"共 {len(file_paths)} 个文件待分析")
    print(f"连续 0 阈值: >= {args.min_zeros} (约 {args.min_zeros * 0.5:.1f} 小时)")

    # 分析每个文件
    results = []
    for i, file_path in enumerate(file_paths):
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{len(file_paths)} 个文件...")

        result = analyze_file(file_path, args.min_zeros)
        results.append(result)

    # 打印汇总
    print_summary(results, args.min_zeros)

    # 输出到 CSV
    if args.output:
        valid_results = [r for r in results if "error" not in r]
        df = pd.DataFrame(valid_results)
        df = df.drop(columns=["zero_segment_lengths"])  # 列表不好存 CSV
        df.to_csv(args.output, index=False)
        print(f"\n详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
