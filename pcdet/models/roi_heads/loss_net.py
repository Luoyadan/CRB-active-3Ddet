import torch.nn as nn
import torch

class LossNet(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        
        self.model_cfg = model_cfg
        self.num_layer = self.model_cfg.LOSS_NET.SHARED_FC.__len__()

        for k in range(self.num_layer):
            end_channel = 1

            conv = nn.Conv1d(self.model_cfg.LOSS_NET.SHARED_FC[k], end_channel, kernel_size=1, bias=False)
            bn = nn.BatchNorm1d(end_channel)
            relu = nn.ReLU()
            setattr(self, 'conv_' + str(k), conv)
            setattr(self, 'bn_' + str(k), bn)
            setattr(self, 'relu_' + str(k), relu)
            

        
        total_dim = model_cfg.TARGET_CONFIG.ROI_PER_IMAGE * self.num_layer
        self.linear = nn.Linear(total_dim, 1)
    
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, features, batch_size=None):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        out_list = []
        for i in range(self.num_layer):
            out = getattr(self, 'conv_' + str(i))(features[i])
            out = getattr(self, 'bn_' + str(i))(out)
            out = getattr(self, 'relu_' + str(i))(out)
            out_list.append(out.view(batch_size, -1))
        output = self.linear(torch.cat(out_list, 1))

        return output
