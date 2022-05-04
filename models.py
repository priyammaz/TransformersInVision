import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from segm_blocks import Block, FeedForward
from segm_utils import init_weights, padding, unpadding
from timm.models.layers import trunc_normal_


#### VISION TRANSFORMER IMPLEMENTATION ####
class PatchEmbedding(nn.Module):
    """
    This function will split our image into patches
    """

    def __init__(self, image_size, patch_size, in_channels=3, embedding=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (int(image_size / patch_size)) ** 2

        self.patching_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=embedding,
                                       kernel_size=patch_size,
                                       stride=patch_size)

    def forward(self, x):
        # Convert single image to embedding x (root n x root n patches)
        x = self.patching_conv(x)

        # Flatten on second dimension to get embedding x n patches
        x = x.flatten(2)

        # Transpose to get n patches x embedding, thus giving each patch a 768 vector embedding
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    """
    Build the attention mechanism (nearly identical to original Transformer Paper
    """

    def __init__(self, embedding, num_heads, qkv_b=True, attention_drop_p=0, projection_drop_p=0):
        super().__init__()
        self.embedding = embedding  # Size of embedding vector
        self.num_heads = num_heads  # Number of heads in multiheaded attention layers
        self.qkv_b = qkv_b  # Do we want a bias term on our QKV linear layer
        self.attention_drop_p = attention_drop_p  # Attention layer dropout probability
        self.projection_drop_p = projection_drop_p  # Projection layer dropout probability
        self.head_dimension = int(self.embedding / self.num_heads)  # Dimension of each head in multiheaded attention
        self.scaling = self.head_dimension ** 0.5  # Scaling recommended by original transformer paper for exploding grad

        self.qkv = nn.Linear(embedding, embedding * 3)
        self.attention_drop = nn.Dropout(self.attention_drop_p)
        self.projection = nn.Linear(embedding, embedding)
        self.projection_drop = nn.Dropout(self.projection_drop_p)

    def forward(self, x):
        # Get shape of input layer, samples x patches + patchembedding (1) x embedding
        samples, patches, embedding = x.shape  # (samples, patches+1, embedding)

        # Expand embedding to 3 x embedding for QKV
        qkv = self.qkv(x)  # (sample, patches+1, 3*embedding)

        # Reshape so that for every patch + 1 in every sample we have QKV with dimension number of heads by its dimension
        # Remember that num_heads * head_dimension = embedding
        qkv = qkv.reshape(samples, patches, 3, self.num_heads,
                          self.head_dimension)  # (samples, patches+1, 3, num_heads, head_dim)

        # Permute such that we have QKV so each has all samples, and each head in each sample
        # has dimensions patches + 1 by heads dimension
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, samples, heads, patches+1, head_dim)

        # Separate out QKV
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Transpose patches and head dimension of K
        transpose_k = k.transpose(-2, -1)  # (samples, heads, head_dim, patches+1)

        # Matrix Multiplication of Q and K scaled
        # (samples, heads, patches+1, head_dim) (samples, heads, head_dim, patches + 1)
        # output: (sample, heads, patches+1, patches+1)
        scaled_mult = torch.matmul(q, transpose_k) / self.scaling

        # Run scaled multiplication through softmax layer along last dimension
        attention = scaled_mult.softmax(dim=-1)
        attention = self.attention_drop(attention)

        # Calculate weighted average along V
        # (sample, heads, patches+1, patches+1) x (samples, heads, patches+1, head_dim)
        # Output (sample, heads, patches+1, head_dim)
        weighted_average = torch.matmul(attention, v)

        # Transpose to (samples, patches+1, heads, head_dim)
        weighted_average = weighted_average.transpose(1, 2)

        # Flatten on last layer to get back original shape of (sample, patches + 1, embedding)
        weighted_average = weighted_average.flatten(2)

        # Run through our projection layer with dropout
        x = self.projection_drop(self.projection(x))
        return x

class MultiLayerPerceptron(nn.Module):
    """
    Simple Multi Layer Perceptron with GELU activation
    """

    def __init__(self, input_features, hidden_features, output_features, dropout_p=0):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.drop_1 = nn.Dropout(dropout_p)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.drop_2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.drop_1(self.gelu(self.fc1(x)))
        x = self.drop_2(self.fc2(x))
        return x

class TransformerBlock(nn.Module):
    """
    Create Self Attention Block with alyer normalization
    """

    def __init__(self, embedding, num_heads, hidden_features=2048, qkv_b=True, attention_dropout_p=0,
                 projection_dropout_p=0, mlp_dropout_p=0):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding, eps=1e-6)
        self.attention = Attention(embedding=embedding,
                                   num_heads=num_heads,
                                   qkv_b=qkv_b,
                                   attention_drop_p=attention_dropout_p,
                                   projection_drop_p=projection_dropout_p)
        self.layernorm2 = nn.LayerNorm(embedding, eps=1e-6)
        self.feedforward = MultiLayerPerceptron(input_features=embedding,
                                                hidden_features=hidden_features,
                                                output_features=embedding,
                                                dropout_p=mlp_dropout_p)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feedforward(self.layernorm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Putting together the Vision Transformer
    """

    def __init__(self, image_size=384, patch_size=16, in_channels=3, num_outputs=1000, embeddings=768,
                 num_blocks=12, num_heads=12, hidden_features=2048, qkv_b=True, attention_dropout_p=0,
                 projection_dropout_p=0, mlp_dropout_p=0, pos_embedding_dropout=0, return_features=True):
        super().__init__()
        self.return_features = return_features
        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              in_channels=in_channels,
                                              embedding=embeddings)
        self.class_token = nn.Parameter(torch.zeros(size=(1, 1, embeddings)))
        self.positional_embedding = nn.Parameter(
            torch.zeros(size=(1, 1 + self.patch_embedding.num_patches, embeddings)))
        self.positional_dropout = nn.Dropout(pos_embedding_dropout)
        self.transformer_block = TransformerBlock(embedding=embeddings,
                                                  num_heads=num_heads,
                                                  hidden_features=hidden_features,
                                                  qkv_b=qkv_b,
                                                  attention_dropout_p=attention_dropout_p,
                                                  projection_dropout_p=projection_dropout_p,
                                                  mlp_dropout_p=mlp_dropout_p)
        self.transformer_blocks = nn.ModuleList([
            self.transformer_block for _ in range(num_blocks)
        ])

        self.layernorm = nn.LayerNorm(embeddings, eps=1e-6)
        self.out = nn.Linear(embeddings, num_outputs)

    def forward(self, x):
        num_samples = x.shape[0]
        x = self.patch_embedding(x)
        class_token = self.class_token.expand(num_samples, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.positional_embedding
        x = self.positional_dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.layernorm(x)

        if self.return_features:
            # return output tensor as it is
            return x
        if not self.return_features:
            # return classification output
            output_class_token = x[:, 0]
            x = self.out(output_class_token)
            return x

#### SEGMENTER IMPLEMENTATION ###
class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x

# mask trasformer in Segmenter
class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff=2048,  # num of hidden features in mlp
        drop_path_rate=0.0,
        dropout=0.1,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

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
        im = padding(im, self.patch_size)
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

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks


### WNET IMPLEMENTATION ###
class WNet_Encoder(nn.Module):

    def __init__(self):
        super(WNet_Encoder, self).__init__()

        # separable conv layers
        # consists of a depthwise convolution and a pointwise convolution
        def separable_conv(in_channels, out_channels):
            layers = []
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)]  # depthwise
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1)]  # pointwise
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]  # depthwise
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=1)]  # pointwise
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            conv_layers = nn.Sequential(*layers)
            return conv_layers

            # conv layers for module 1, 9, 10, 18

        def conv(in_channels, out_channels):
            layers = []
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            conv_layers = nn.Sequential(*layers)
            return conv_layers

            # separable_conv + up sampling

        def up(in_channels, out_channels):
            up_layers = []
            up_layers += separable_conv(in_channels, out_channels)
            up_layers += [nn.ConvTranspose2d(int(in_channels / 2), int(out_channels / 2), kernel_size=2, stride=2)]
            up_layers = nn.Sequential(*up_layers)
            return up_layers

            # Max Pooling layer used in the each of contracting step

        self.max = nn.MaxPool2d(2)

        # Contracting Path
        self.down1 = conv(3, 64)
        self.down2 = separable_conv(64, 128)
        self.down3 = separable_conv(128, 256)
        self.down4 = separable_conv(256, 512)

        # The bottom covolutional layer
        self.bottom = separable_conv(512, 1024)
        self.up_sample = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Expanding Path
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = conv(128, 64)

        # Last output layer
        self.last = nn.Conv2d(64, 9, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        x1 = self.down1(x)
        x1_max = self.max(x1)
        x2 = self.down2(x1_max)
        x2_max = self.max(x2)
        x3 = self.down3(x2_max)
        x3_max = self.max(x3)
        x4 = self.down4(x3_max)
        x4_max = self.max(x4)

        # Bottom
        x5 = self.bottom(x4_max)
        x5_up = self.up_sample(x5)

        # Expanding Path
        x6 = self.up1(torch.cat((x4, x5_up), dim=1))  # Skip Connection
        x7 = self.up2(torch.cat((x3, x6), dim=1))  # Skip Connection
        x8 = self.up3(torch.cat((x2, x7), dim=1))  # Skip Connection
        x9 = self.up4(torch.cat((x1, x8), dim=1))  # Skip Connection
        output = self.last(x9)

        return output

class WNet_Decoder(nn.Module):

    def __init__(self):
        super(WNet_Decoder, self).__init__()

        # separable conv layers
        # consists of a depthwise convolution and a pointwise convolution
        def separable_conv(in_channels, out_channels):
            layers = []
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)]  # depthwise
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1)]  # pointwise
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]  # depthwise
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=1)]  # pointwise
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            conv_layers = nn.Sequential(*layers)
            return conv_layers

            # conv layers for module 1, 9, 10, 18

        def conv(in_channels, out_channels):
            layers = []
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            conv_layers = nn.Sequential(*layers)
            return conv_layers

            # conv + up sampling

        def up(in_channels, out_channels):
            up_layers = []
            up_layers += separable_conv(in_channels, out_channels)
            up_layers += [nn.ConvTranspose2d(int(in_channels / 2), int(out_channels / 2), kernel_size=2, stride=2)]
            up_layers = nn.Sequential(*up_layers)
            return up_layers

            # SoftMax

        self.softmax = nn.Softmax(dim=1)
        # Max Pooling layer used in the each of contracting step
        self.max = nn.MaxPool2d(2)

        # Contracting Path
        self.down1 = conv(9, 64)
        self.down2 = separable_conv(64, 128)
        self.down3 = separable_conv(128, 256)
        self.down4 = separable_conv(256, 512)

        # The bottom covolutional layer
        self.bottom = separable_conv(512, 1024)
        self.up_sample = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        # Expanding Path
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = conv(128, 64)

        # Last output layer
        self.last = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        x10 = self.down1(x)
        x10_max = self.max(x10)
        x11 = self.down2(x10_max)
        x11_max = self.max(x11)
        x12 = self.down3(x11_max)
        x12_max = self.max(x12)
        x13 = self.down4(x12_max)
        x13_max = self.max(x13)

        # Bottom
        x14 = self.bottom(x13_max)
        x14_up = self.up_sample(x14)

        # Expanding Path
        x15 = self.up1(torch.cat((x13, x14_up), dim=1))  # Skip Connection
        x16 = self.up2(torch.cat((x12, x15), dim=1))  # Skip Connection
        x17 = self.up3(torch.cat((x11, x16), dim=1))  # Skip Connection
        x18 = self.up4(torch.cat((x10, x17), dim=1))  # Skip Connection

        output = self.last(x18)

        return output

class WNet(nn.Module):

    def __init__(self):
        super(WNet, self).__init__()
        # Encoder
        self.WNet_Encoder = WNet_Encoder()
        self.WNet_Decoder = WNet_Decoder()
        # Decoder

    def forward(self, x, enc_rec):

        enc = self.WNet_Encoder(x)

        if enc_rec == "enc":
            return enc

        dec = self.WNet_Decoder(F.softmax(enc, 1))
        if enc_rec == 'dec':
            return dec

        if enc_rec == 'both':
            return enc, dec

        return dec