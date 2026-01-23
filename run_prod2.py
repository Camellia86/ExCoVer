#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产版启动脚本 - 直接开始训练，无需交互
使用线上数据路径: /root/autodl-fs/stickers
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from learning_training_system import LearningTrainingSystem


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  🚀 语言化学习训练系统 - 一键启动 (生产版)")
    print("="*70 + "\n")

    # 配置参数 (生产环境)
    train_json = "train均匀.json"
    image_path = "/root/autodl-fs/stickers"  # 生产路径
    batch_size = 4  # ✨ 批处理大小：2表示累积2个错误样本后调用Optimizer

    # 检查训练数据是否存在
    if not os.path.exists(train_json):
        print(f"❌ 错误: 找不到 {train_json}")
        print(f"   请确保 {train_json} 文件存在")
        sys.exit(1)

    try:
        # 初始化系统
        print(f"📂 训练数据: {train_json}")
        print(f"🖼️  图片路径: {image_path}")
        print(f"\n⏳ 初始化系统...")
        system = LearningTrainingSystem(train_json, image_path, batch_size=batch_size)

        # 显示数据统计
        print(f"\n📊 系统状态:")
        print(f"  ✓ 已加载 {len(system.train_data)} 条训练数据")
        print(f"  ✓ 规则表: {'已有内容' if system.similar_intent_rules else '为空（初始化中）'}")

        # 显示API信息
        print(f"\n🔌 API配置:")
        print(f"  ✓ Learner & Optimizer: 火山引擎 (Doubao)")
        print(f"  ✓ Regularizer: OpenAI代理 (Gemini-3)")

        # 开始训练
        print(f"\n{'='*70}")
        print("🎓 开始训练...")
        print(f"{'='*70}\n")

        system.train(save_interval=20, resume_from =0)

        # 训练完成
        print(f"\n{'='*70}")
        print("✅ 训练完成！")
        print(f"{'='*70}")
        print(f"\n📈 最终统计:")
        print(f"  ✓ 总样本数: {system.training_stats['total_samples']}")
        print(f"  ✓ 正确预测: {system.training_stats['correct_count']}")
        print(f"  ✓ 错误预测: {system.training_stats['error_count']}")
        print(f"  ✓ Optimizer调用: {system.training_stats['optimizer_calls']} 次")
        print(f"  ✓ Regularizer调用: {system.training_stats['regularizer_calls']} 次")
        if system.training_stats['total_samples'] > 0:
            accuracy = system.training_stats['correct_count'] / system.training_stats['total_samples']
            print(f"  ✓ 准确率: {accuracy:.2%}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
