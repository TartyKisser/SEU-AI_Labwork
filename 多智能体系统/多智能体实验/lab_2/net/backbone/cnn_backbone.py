import torch
import torch.nn as nn

from net.component.basic_conv import BasicConv
from net.component.basic_res_block import BasicResBlock


class CNNBackbone(nn.Module):

    def __init__(self, channel):
        super(CNNBackbone, self).__init__()
        self.conv1_311 = BasicConv(channel, channel, 3, 1, 1)
        self.res_1 = nn.Sequential(*[BasicResBlock(channel) for _ in range(20)])
        self.conv2_312 = BasicConv(channel, 2*channel, 3, 1, 2)
        self.res_2 = nn.Sequential(*[BasicResBlock(2*channel) for _ in range(50)])
        self.up_2 = nn.Upsample(scale_factor=2)
        self.conv3_312 = BasicConv(2*channel, 4*channel, 3, 1, 2)
        self.res_3 = nn.Sequential(*[BasicResBlock(4*channel) for _ in range(90)])
        self.up_3 = nn.Upsample(scale_factor=4)
        self.conv4_311 = BasicConv(7*channel, channel, 3, 1, 1)

    import torch
    import torch.nn as nn

    def forward(self, x):
        print(f"=" * 50)
        print(f"CNNBackbone Forward Pass Debug")
        print(f"=" * 50)

        # 输入调试信息
        print(f"Input shape: {x.shape}")
        print(f"Input - Batch: {x.shape[0]}, Channels: {x.shape[1]}, Height: {x.shape[2]}, Width: {x.shape[3]}")

        # 第一阶段：初始卷积
        print(f"\n--- Stage 1: Initial Conv ---")
        x = self.conv1_311(x)
        print(f"After conv1_311: {x.shape}")

        # 第一分支：原尺度特征
        print(f"\n--- Branch 1: Original Scale ---")
        x_branch1 = x  # 保存分支点
        out1 = self.res_1(x_branch1)
        print(f"After res_1 (20 blocks): {out1.shape}")
        print(f"Branch 1 final (out1): {out1.shape}")

        # 第二分支：开始处理
        print(f"\n--- Branch 2: 2x Channels ---")
        print(f"Starting from conv1_311 output: {x.shape}")
        x = self.conv2_312(x)
        print(f"After conv2_312: {x.shape}")

        out2 = self.res_2(x)
        print(f"After res_2 (50 blocks): {out2.shape}")

        out2 = self.up_2(out2)
        print(f"After up_2 (2x upsample): {out2.shape}")
        print(f"Branch 2 final (out2): {out2.shape}")

        # 第三分支：继续处理
        print(f"\n--- Branch 3: 4x Channels ---")
        print(f"Starting from conv2_312 output: {x.shape}")
        x = self.conv3_312(x)
        print(f"After conv3_312: {x.shape}")

        out3 = self.res_3(x)
        print(f"After res_3 (90 blocks): {out3.shape}")

        out3 = self.up_3(out3)
        print(f"After up_3 (4x upsample): {out3.shape}")
        print(f"Branch 3 final (out3): {out3.shape}")

        # 特征融合
        print(f"\n--- Feature Fusion ---")
        print(f"Before concat:")
        print(f"  out1: {out1.shape}")
        print(f"  out2: {out2.shape}")
        print(f"  out3: {out3.shape}")

        # 检查是否可以拼接
        try:
            out = torch.cat([out1, out2, out3], dim=1)
            print(f"After concat: {out.shape}")
            print(f"Concat successful!")

            total_channels = out1.shape[1] + out2.shape[1] + out3.shape[1]
            print(f"Total channels: {out1.shape[1]} + {out2.shape[1]} + {out3.shape[1]} = {total_channels}")

        except RuntimeError as e:
            print(f"Concat failed: {e}")
            print(f"Dimension mismatch detected!")

            # 分析尺寸差异
            h1, w1 = out1.shape[2], out1.shape[3]
            h2, w2 = out2.shape[2], out2.shape[3]
            h3, w3 = out3.shape[2], out3.shape[3]

            print(f"Spatial dimensions:")
            print(f"  out1: {h1}×{w1}")
            print(f"  out2: {h2}×{w2}")
            print(f"  out3: {h3}×{w3}")

            # 建议修复方案
            if h2 != h1 or w2 != w1:
                print(f"\nSuggested fix for out2:")
                print(f"  Current: {h2}×{w2} → Target: {h1}×{w1}")
                if h2 > h1:
                    print(f"  Need downsample by factor: {h2 // h1}")
                else:
                    print(f"  Need upsample by factor: {h1 // h2}")

            if h3 != h1 or w3 != w1:
                print(f"\nSuggested fix for out3:")
                print(f"  Current: {h3}×{w3} → Target: {h1}×{w1}")
                if h3 > h1:
                    print(f"  Need downsample by factor: {h3 // h1}")
                else:
                    print(f"  Need upsample by factor: {h1 // h3}")

            # 尝试修复（示例）
            print(f"\n--- Attempting Auto-fix ---")
            if h2 > h1:  # out2需要下采样
                down_factor = h2 // h1
                out2_fixed = nn.functional.adaptive_avg_pool2d(out2, (h1, w1))
                print(f"out2 downsampled to: {out2_fixed.shape}")
            else:
                out2_fixed = out2

            if h3 > h1:  # out3需要下采样
                down_factor = h3 // h1
                out3_fixed = nn.functional.adaptive_avg_pool2d(out3, (h1, w1))
                print(f"out3 downsampled to: {out3_fixed.shape}")
            else:
                out3_fixed = out3

            # 重新尝试拼接
            try:
                out = torch.cat([out1, out2_fixed, out3_fixed], dim=1)
                print(f"After fixed concat: {out.shape}")
            except Exception as e2:
                print(f"Fixed concat still failed: {e2}")
                return None

        # 最终卷积
        print(f"\n--- Final Convolution ---")
        print(f"Before conv4_311: {out.shape}")
        out = self.conv4_311(out)
        print(f"After conv4_311: {out.shape}")

        print(f"\n--- Summary ---")
        print(f"Input:  {x.shape if 'x' in locals() else 'N/A'}")
        print(f"Output: {out.shape}")
        print(f"Network flow: Input → conv1 → [3 branches with residuals] → concat → conv4 → Output")

        return out

