import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    This function will split our image into patches
    """

    def __init__(self, image_size, patch_size, in_channels=3, embedding=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (int(image_size/patch_size))**2

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
        x = x.transpose(1,2)
        return x

class Attention(nn.Module):
    """
    Build the attention mechanism (nearly identical to original Transformer Paper
    """
    def __init__(self, embedding, num_heads, qkv_b=True, attention_drop_p=0, projection_drop_p=0):
        super().__init__()
        self.embedding = embedding # Size of embedding vector
        self.num_heads = num_heads # Number of heads in multiheaded attention layers
        self.qkv_b = qkv_b # Do we want a bias term on our QKV linear layer
        self.attention_drop_p = attention_drop_p # Attention layer dropout probability
        self.projection_drop_p = projection_drop_p # Projection layer dropout probability
        self.head_dimension = int(self.embedding/self.num_heads) # Dimension of each head in multiheaded attention
        self.scaling = self.head_dimension ** 0.5 # Scaling recommended by original transformer paper for exploding grad


        self.qkv = nn.Linear(embedding, embedding * 3)
        self.attention_drop = nn.Dropout(self.attention_drop_p)
        self.projection = nn.Linear(embedding, embedding)
        self.projection_drop = nn.Dropout(self.projection_drop_p)

    def forward(self, x):
        # Get shape of input layer, samples x patches + patchembedding (1) x embedding
        samples, patches, embedding = x.shape # (samples, patches+1, embedding)

        # Expand embedding to 3 x embedding for QKV
        qkv = self.qkv(x) # (sample, patches+1, 3*embedding)

        # Reshape so that for every patch + 1 in every sample we have QKV with dimension number of heads by its dimension
        # Remember that num_heads * head_dimension = embedding
        qkv = qkv.reshape(samples, patches, 3, self.num_heads, self.head_dimension) # (samples, patches+1, 3, num_heads, head_dim)

        # Permute such that we have QKV so each has all samples, and each head in each sample
        # has dimensions patches + 1 by heads dimension
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, samples, heads, patches+1, head_dim)

        # Separate out QKV
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Transpose patches and head dimension of K
        transpose_k = k.transpose(-2, -1) # (samples, heads, head_dim, patches+1)

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
        weighted_average = weighted_average.transpose(1,2)

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
    def __init__(self, embedding, num_heads, hidden_features=2048,qkv_b=True, attention_dropout_p=0,
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
        x += self.attention(self.layernorm1(x))
        x += self.feedforward(self.layernorm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Putting together the Vision Transformer
    """
    def __init__(self, image_size=384, patch_size=16, in_channels=3, num_outputs=1000, embeddings=768,
                 num_blocks=12, num_heads=12, hidden_features=2048, qkv_b=True, attention_dropout_p=0,
                 projection_dropout_p=0, mlp_dropout_p=0, pos_embedding_dropout=0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              in_channels=in_channels,
                                              embedding=embeddings)
        self.class_token = nn.Parameter(torch.zeros(size=(1,1,embeddings)))
        self.positional_embedding = nn.Parameter(torch.zeros(size=(1,1+self.patch_embedding.num_patches, embeddings)))
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
        x += self.positional_embedding
        x = self.positional_dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.layernorm(x)
        output_class_token = x[:, 0]
        x = self.out(output_class_token)
        return x
if __name__ == "__main__":
    rand = torch.rand(size=(10,3,384,384))
    vit = VisionTransformer()
    out = vit(rand)

