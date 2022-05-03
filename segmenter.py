
import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import MaskTransformer

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

        x = self.encoder.patch_embed(im)
        x = torch.cat((self.encoder.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1# + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        # masks = unpadding(masks, (H_ori, W_ori))

        return masks

if __name__ == "__main__":
    B_ = 10
    C_ = 3
    H_ = 384
    W_ = 384
    n_cls_ = 150
    d_model_ = 768
    patch_size_ = 16
    num_patches_ = (H_ // patch_size_) * (W_ // patch_size_)

    x = torch.rand(size=(B_, C_, H_, W_))
    
    # !pip install timm
    from timm import create_model

    ### mask transformer as decoder 
    timm_vit = create_model("vit_base_patch16_224", pretrained=True)
    segmenter = Segmenter(encoder=timm_vit, 
                          decoder=MaskTransformer(n_cls=150, 
                                                  patch_size=16, 
                                                  d_encoder=768,
                                                  n_layers=2,
                                                  n_heads=12,
                                                  d_model=768,
                                                  ), 
                          n_cls=150, patch_size=16)

    pred = segmenter(x)
    print(pred.shape)