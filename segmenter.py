import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import VisionTransformer
from decoder import DecoderLinear, MaskTransformer

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        patch_size,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        # im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1# + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        # masks = unpadding(masks, (H_ori, W_ori))

        return masks

if __name__ == "__main__":
    B_ = 10
    C_ = 3
    H_ = 384
    W_ = 384
    n_cls_ = 9
    d_model_ = 768
    patch_size_ = 16
    num_patches_ = (H_ // patch_size_) * (W_ // patch_size_)

    x = torch.rand(size=(B_, C_, H_, W_))
    
    ### linear layer as decoder 
    # segmenter = Segmenter(encoder=VisionTransformer(), 
    #                       decoder=DecoderLinear(n_cls=n_cls_, patch_size=patch_size_, d_encoder=d_model_), 
    #                       n_cls=n_cls_, patch_size=patch_size_)

    ### mask transformer as decoder 
    segmenter = Segmenter(encoder=VisionTransformer(), 
                          decoder=MaskTransformer(n_cls=n_cls_, 
                                                  patch_size=patch_size_, 
                                                  d_encoder=d_model_,
                                                  n_layers=2,
                                                  n_heads=12,
                                                  d_model=d_model_,
                                                  ), 
                          n_cls=n_cls_, patch_size=patch_size_)

    pred = segmenter(x)
    print(pred.shape)