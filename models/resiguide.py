import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

from modules.transform import *
from modules.transform.context import ResiGuideModule
from modules.transform.quantization import LatentResidualPrediction

def ste_round(x):
    return torch.round(x) - x.detach() + x

class ResiGuide(CompressionModel):

    def __init__(self, N=192, M=320, S=10, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        self.S = S  
        self.M = M  
        

        self.g_a = AnalysisTransform(3, M)     
        self.g_s = SynthesisTransform(M, 3)     
        

        self.h_a = HyperAnalysis(M, N)          
        self.h_s = HyperSynthesis(N, M)         
        

        self.entropy_parameters = nn.ModuleList()
        

        self.lrp = nn.ModuleList()
        

        self.channel_context = nn.ModuleList()
        

        self.context_modules = nn.ModuleList()
        
        self.gaussian_conditional = GaussianConditional(None)
        
        for i in range(S):
            if i == 0:
                self.entropy_parameters.append(
                    nn.Sequential(
                        nn.Conv2d(M * 2, 640, 1),
                        nn.GELU(),
                        nn.Conv2d(640, 640, 1),
                        nn.GELU(),
                        nn.Conv2d(640, M // S * 2, 1)
                    )
                )
                self.lrp.append(
                    LatentResidualPrediction(M + M // S, M // S)
                )
            else:
                self.context_modules.append(
                    ResiGuideModule(M // S, M // S)
                )
                
                self.channel_context.append(
                    nn.Sequential(
                        nn.Conv2d(M // S * i, M // S, 3, padding=1),
                        nn.GELU(),
                        nn.Conv2d(M // S, M // S, 3, padding=1)
                    )
                )
                
                self.entropy_parameters.append(
                    nn.Sequential(
                        nn.Conv2d(M // S * 2 + M * 2, 640, 1),
                        nn.GELU(),
                        nn.Conv2d(640, 640, 1),
                        nn.GELU(),
                        nn.Conv2d(640, M // S * 2, 1)
                    )
                )
                
                self.lrp.append(
                    LatentResidualPrediction((i + 1) * M // S + M, M // S)
                )

    def forward(self, x):
        y = self.g_a(x)
        
        z = self.h_a(y)
        
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        
        y_slices = y.chunk(self.S, dim=1)  
        y_hat_slices = []  
        y_likelihoods = []  
        
        intermediates = {
            'latent_y': y.detach(),
            'latent_z': z.detach(), 
            'z_hat': z_hat.detach(),
            'hyper_params': hyper_params.detach(),
            'channel_ctx': [],
            'Phi_global': [],  
            'Phi_local': [],   
            'R': [],           
            'F_raw': [],       
            'F_filter_low': [], 
            'F_filter_high': [] 
        }
        
        for i, y_slice in enumerate(y_slices):
            if i == 0:
                params = self.entropy_parameters[i](hyper_params)
                scales, means = params.chunk(2, 1)
                y_slice_hat = ste_round(y_slice - means) + means
                
                lrp = self.lrp[i](torch.cat(([hyper_means] + y_hat_slices + [y_slice_hat]), dim=1))
                y_slice_hat = y_slice_hat + lrp
                
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales, means)
                y_hat_slices.append(y_slice_hat)
                y_likelihoods.append(y_slice_likelihoods)
            else:
                channel_ctx = self.channel_context[i-1](torch.cat(y_hat_slices, dim=1))
                intermediates['channel_ctx'].append(channel_ctx.detach())
                
                y_hat_prev = y_hat_slices[-1]  
                context_features = self.context_modules[i-1](y_hat_prev)
                
                if hasattr(self.context_modules[i-1], 'intermediates'):
                    for key, value in self.context_modules[i-1].intermediates.items():
                        if key in intermediates:
                            intermediates[key].append(value)
                
                params = self.entropy_parameters[i](
                    torch.cat([context_features, channel_ctx, hyper_params], dim=1)
                )
                
                scales, means = params.chunk(2, 1)
                y_slice_hat = ste_round(y_slice - means) + means
                
                lrp = self.lrp[i](torch.cat(([hyper_means] + y_hat_slices + [y_slice_hat]), dim=1))
                y_slice_hat = y_slice_hat + lrp
                
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales, means)
                y_hat_slices.append(y_slice_hat)
                y_likelihoods.append(y_slice_likelihoods)
        
        y_hat = torch.cat(y_hat_slices, dim=1)  
        x_hat = self.g_s(y_hat)  
        
        return {
            "x_hat": x_hat,  
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}, 
            "intermediates": intermediates  
        }

    def compress(self, x):
        torch.cuda.synchronize()
        start_time = time.time()
        
        y = self.g_a(x)
        z = self.h_a(y)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        
        y_slices = y.chunk(self.S, dim=1)
        y_hat_slices = []
        
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        
        for i, y_slice in enumerate(y_slices):
            if i == 0:
                params = self.entropy_parameters[i](hyper_params)
                scales, means = params.chunk(2, 1)
                
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_slice_q = self.gaussian_conditional.quantize(y_slice, "symbols", means)
                
                symbols_list.extend(y_slice_q.reshape(-1).tolist())
                indexes_list.extend(indexes.reshape(-1).tolist())
                
                y_slice_hat = y_slice_q + means
                
                lrp = self.lrp[i](torch.cat(([hyper_means] + y_hat_slices + [y_slice_hat]), dim=1))
                y_slice_hat = y_slice_hat + lrp
                
                y_hat_slices.append(y_slice_hat)
            else:
                channel_ctx = self.channel_context[i-1](torch.cat(y_hat_slices, dim=1))
                
                y_hat_prev = y_hat_slices[-1]
                context_features = self.context_modules[i-1](y_hat_prev)
                
                params = self.entropy_parameters[i](
                    torch.cat([context_features, channel_ctx, hyper_params], dim=1)
                )
                
                scales, means = params.chunk(2, 1)
                
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_slice_q = self.gaussian_conditional.quantize(y_slice, "symbols", means)
                
                symbols_list.extend(y_slice_q.reshape(-1).tolist())
                indexes_list.extend(indexes.reshape(-1).tolist())
                
                y_slice_hat = y_slice_q + means
                
                lrp = self.lrp[i](torch.cat(([hyper_means] + y_hat_slices + [y_slice_hat]), dim=1))
                y_slice_hat = y_slice_hat + lrp
                
                y_hat_slices.append(y_slice_hat)
        
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        
        y_string = encoder.flush()
        
        torch.cuda.synchronize()
        end_time = time.time()
        cost_time = end_time - start_time
        
        return {"strings": [y_string, z_strings], "shape": z.size()[-2:], "cost_time": cost_time}

    def decompress(self, strings, shape):
        torch.cuda.synchronize()
        start_time = time.time()
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        
        y_shape = [z_hat.shape[0], self.M, z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_slice_shape = [z_hat.shape[0], self.M // self.S, z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0])
        
        y_hat_slices = []
        
        for i in range(self.S):
            if i == 0:
                params = self.entropy_parameters[i](hyper_params)
                scales, means = params.chunk(2, 1)
                
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_slice_q = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                y_slice_hat = torch.Tensor(y_slice_q).reshape(scales.shape).to(scales.device) + means
                
                lrp = self.lrp[i](torch.cat(([hyper_means] + y_hat_slices + [y_slice_hat]), dim=1))
                y_slice_hat = y_slice_hat + lrp
                
                y_hat_slices.append(y_slice_hat)
            else:
                channel_ctx = self.channel_context[i-1](torch.cat(y_hat_slices, dim=1))
                
                y_hat_prev = y_hat_slices[-1]
                context_features = self.context_modules[i-1](y_hat_prev)
                
                params = self.entropy_parameters[i](
                    torch.cat([context_features, channel_ctx, hyper_params], dim=1)
                )
                
                scales, means = params.chunk(2, 1)
                
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_slice_q = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                y_slice_hat = torch.Tensor(y_slice_q).reshape(scales.shape).to(scales.device) + means
                
                lrp = self.lrp[i](torch.cat(([hyper_means] + y_hat_slices + [y_slice_hat]), dim=1))
                y_slice_hat = y_slice_hat + lrp
                
                y_hat_slices.append(y_slice_hat)
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        
        x_hat = self.g_s(y_hat)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        cost_time = end_time - start_time
        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }