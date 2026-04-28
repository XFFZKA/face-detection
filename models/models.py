import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceSwap(nn.Module):
    def __init__(self, use_gpu=True):
        super().__init__()
        self.swap_model = UNet()
        self.predict_model = BuildFaceSwap()
        # self.predict_model.load_state_dict(torch.load('/home/yl/shangbo/Train/checkpoints/weight/predict_model.pth'))
        # self.swap_model.mask.load_state_dict(torch.load('/home/yl/shangbo/Train/checkpoints/weight/mask_weights.pth'))
        # self.swap_model.final.load_state_dict(torch.load('/home/yl/shangbo/Train/checkpoints/weight/final_weights.pth'))

    def set_model_param(self, id_emb, id_feature_map):

        weights_encoder, weights_decoder, encode_mod, decode_mod = self.predict_model(id_emb, id_feature_map)

        for i in range(len(self.swap_model.Encoder)):
            self.swap_model.Encoder[i][0].set_weight(weights_encoder[i][0].unsqueeze(axis=1))
            self.swap_model.Encoder[i][1].set_weight(encode_mod[i])

        for i in range(len(self.swap_model.Decoder)):
            self.swap_model.Decoder[i][0].set_weight(weights_decoder[i][0].unsqueeze(axis=1))
            self.swap_model.Decoder[i][1].set_weight(decode_mod[i])

    def forward(self, att_img):
        img, mask = self.swap_model(att_img)

        return img, mask


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512]
        self.Encoder = nn.ModuleList()#用于管理子层的列表
        for i in range(len(self.Encoder_channel)-1):
            self.Encoder.append(nn.Sequential(*[
                Conv2dFunction(stride=2,padding=1,groups=self.Encoder_channel[i]),
                Conv2dFunction(stride=1,padding=0),
            ]))

        self.Decoder_inchannel = [512, 512, 256, 128]
        self.Decoder_outchannel = [256, 128, 64, 32]
        self.Decoder = nn.ModuleList()
        for i in range(len(self.Decoder_inchannel)):            
            self.Decoder.append(nn.Sequential(*[
                Conv2dFunction(stride=1,padding=1,groups=self.Decoder_inchannel[i]),
                Conv2dFunction(stride=1,padding=0),
            ]))

        self.relu = nn.LeakyReLU(0.1)
        self.up = nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear')


        self.final = nn.Sequential(*[
                nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear'), 
                nn.Conv2d(self.Decoder_outchannel[-1], self.Decoder_outchannel[-1] // 4, kernel_size=1),
                nn.BatchNorm2d(self.Decoder_outchannel[-1] // 4), 
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.Decoder_outchannel[-1] // 4, 3, 3, padding=1),
                nn.BatchNorm2d(3), 
                nn.LeakyReLU(0.1),
                nn.Conv2d(3, 3, 3, padding=1),
                nn.Tanh()
        ])


        mask_channel = [512, 128, 64, 32, 8, 2]
        mask = []
        for i in range(len(mask_channel)-1):
            mask.append(nn.Sequential(
                    nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear'), 
                    nn.Conv2d(mask_channel[i], mask_channel[i], kernel_size=3, stride=1, padding=1, groups=mask_channel[i]),
                    nn.Conv2d(mask_channel[i], mask_channel[i+1], kernel_size=1, stride=1),
                    nn.BatchNorm2d(mask_channel[i+1]), 
                    nn.LeakyReLU(0.1)
                ))
        mask.append(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1))
        mask.append(nn.Sigmoid())
        self.mask = nn.Sequential(*mask)

        weight_channel = [512,256,128,64]
        weight = []
        for i in range(len(weight_channel)-1):
            weight.append(nn.Sequential(
                    nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear'), 
                    Spatial_attention()
                ))
        self.weight = nn.Sequential(*weight)
        self.sig = nn.Sigmoid()


    def forward(self, data):
        x = (data - 0.5) / 0.5
        arr_x = []
        for i in range(len(self.Encoder)):
            x = self.relu(self.Encoder[i](x))
            arr_x.append(x)

        mask = x.detach()

        for i in range(len(self.mask)):
            mask = self.mask[i](mask)

        weight = x
        weights = []
        for i in range(len(self.weight)):
            weight = self.weight[i](weight)
            weights.append(self.sig(weight))

        y = arr_x[-1]
        for i in range(len(self.Decoder)):
            y = self.up(y)
            y = self.relu(self.Decoder[i](y))
            if i != len(self.Decoder) - 1:
                wei = arr_x[len(self.Decoder)-1-i] * weights[i]
                y = torch.cat((y, wei), 1)
        out = self.final(y)
        out = (1 + out) / 2.0
        out = out * mask + (1-mask) * data
        return out, mask


class BuildFaceSwap(nn.Module):
    def __init__(self, opt=None):
        super(BuildFaceSwap, self).__init__()
        encoder_scale = 2

        self.Encoder_channel = [3, 64//encoder_scale, 128//encoder_scale, 256//encoder_scale, 512//encoder_scale, 1024//encoder_scale]

        self.EncoderModulation = nn.ModuleList()
        for i in range(len(self.Encoder_channel)-1):
            self.EncoderModulation.append(Mod2Weight(self.Encoder_channel[i], self.Encoder_channel[i+1]))

        self.Decoder_inchannel = [1024//encoder_scale, 1024//encoder_scale, 512//encoder_scale, 256//encoder_scale]
        self.Decoder_outchannel = [512//encoder_scale, 256//encoder_scale, 128//encoder_scale, 64//encoder_scale]
        
        self.DecoderModulation = nn.ModuleList()
        for i in range(len(self.Decoder_inchannel)):
            self.DecoderModulation.append(Mod2Weight(self.Decoder_inchannel[i], self.Decoder_outchannel[i]))

        self.predictor = WeightPrediction(self.Encoder_channel[:-1], self.Decoder_inchannel)



    def forward(self, id_emb, id_feature_map):
        weights_encoder, weights_decoder = self.predictor(id_feature_map)

        encode_mod = []
        decode_mod = []
        for i in range(len(self.EncoderModulation)):
            encode_mod.append(self.EncoderModulation[i](id_emb))

        for i in range(len(self.DecoderModulation)):
            decode_mod.append(self.DecoderModulation[i](id_emb))

        return weights_encoder, weights_decoder, encode_mod, decode_mod


class WeightPrediction(nn.Module):#预测解码器编码器权重
    def __init__(self, encoder_channels, decoder_channels, style_dim=512):
        super().__init__()
        self.first = nn.Conv2d(style_dim, style_dim, kernel_size=4, stride=1)
        self.first_decoder = nn.Conv2d(style_dim, style_dim, kernel_size=2, stride=1)

        self.decoder_norm = nn.BatchNorm2d(style_dim)
        self.norm =nn.BatchNorm2d(style_dim)
        
        self.relu = nn.LeakyReLU(0.1)

        encoder_channels += [style_dim]
        encoder_channels = encoder_channels[::-1]
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels)-1):
            self.encoder.append(ConvBlock(encoder_channels[i], encoder_channels[i+1]))
        
        decoder_channels = [style_dim] + decoder_channels
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_channels)-1):
            self.decoder.append(ConvBlock(decoder_channels[i], decoder_channels[i+1]))

    def forward(self, z_id):
        encoder_weights = []
        decoder_weights = []
        z_id = self.first(z_id)
        z_id = self.relu(self.norm(z_id))
        x = z_id
        for i in range(len(self.encoder)):                
            x, weight = self.encoder[i](x)
            encoder_weights.append(weight)
        y = z_id
        y = self.relu(self.decoder_norm(self.first_decoder(y)))
        for i in range(len(self.decoder)):
            y, weight = self.decoder[i](y)
            decoder_weights.append(weight)
        return encoder_weights[::-1], decoder_weights


class Mod2Weight(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim=512):
        super().__init__()
        self.out_channel = out_channel

        self.kernel = 1
        self.stride = 1
        self.eps = 1e-16

        self.style = nn.Linear(style_dim, in_channel)
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, self.kernel, self.kernel).float(), requires_grad=True)#方便地创建可训练的参数变量，并将其添加到模型中。通常，神经网络中的权重和偏置是需要被训练的参数。


    def forward(self, style, b=1):
        style = self.style(style)
        scale_deta = style.unsqueeze(axis=1).unsqueeze(axis=-1).unsqueeze(axis=-1)

        weights = self.weight.unsqueeze(axis=0) * (scale_deta + 1) # * 是前后逐元素相乘

        d = torch.rsqrt((weights ** 2).sum(axis=(2, 3, 4), keepdim=True) + self.eps)  #rsqrt（x）计算1/x½
        weights = weights * d

        _, _, *ws = weights.shape
        
        weights = weights.reshape((b * self.out_channel, *ws))
        return weights


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, padding_mode='zeros'):
        super().__init__()
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)
        self.weight = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, padding_mode=padding_mode)

    def forward(self, x): 
        out = self.relu(self.norm(self.conv(x)))
        weight = self.weight(out)
        return out, weight


class Conv2dFunction(nn.Module):
    def __init__(self,stride,padding,groups=1):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.groups = groups

    def set_weight(self,weight):
        self.weight = weight

    def forward(self,x):

        out = F.conv2d(x,weight=self.weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups)

        return out
    

class Spatial_attention(nn.Module):
    def __init__(self,kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=1,bias=False)
    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return x

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.divide(input, norm)
    return output
