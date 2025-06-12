"""
Cheap-Align: a 3x3 conv module that imitates the SD-XL VAE decoder
==================================================================

Input  : float tensor in [0, 1] range of shape (B, 3, H, W)
Output : float tensor in [0, 1] range of shape (B, 3, H, W)
Params : ~0.1 M (3-layer conv, 3 → 64 → 64 → 3)

During pre-training we minimise ‖CheapAlign(x) - VAE(x)‖₁.
"""

from typing import Optional, List
import torch
import torch.nn as nn

class CheapAlign(nn.Module):
        def __init__(
                self,
                hidden: int = 64,
                use_bn: bool = False,
                act: Optional[nn.Module] = nn.ReLU(inplace=True)
        ):
                """
                Args:
                    hidden : number of channels in the two hidden conv layers.
                    use_bn : insert BatchNorm2d after each hidden conv layer.
                    act    : activation to apply after hidden layers (None = linear).
                """
                
                super().__init__()
                
                def block(in_channels: int, out_channels: int) -> List[nn.Module]:
                        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
                        
                        if use_bn:
                                layers.append(nn.BatchNorm2d(out_channels))
                        if act is not None:
                                layers.append(act)
                        
                        return layers
                
                self.net = nn.Sequential(
                        *block(3, hidden),
                        *block(hidden, hidden),
                        nn.Conv2d(hidden, 3, kernel_size=3, padding=1)
                )
                self._init_weights()
                
        def _init_weights(self) -> None:
                """
                Kaiming-uniform init for conv weights with zero bias.
                """
                for m in self.modules():
                        if isinstance(m, nn.Conv2d):
                                nn.init.kaiming_uniform_(m.weight, a=0.0, mode='fan_out')
                                if m.bias is not None:
                                        nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.net(x)
                return y.clamp(0, 1)