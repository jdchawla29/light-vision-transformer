import torch
import torch.nn as nn
import torch.nn.functional as F

class Patchify(nn.Module):


    pass



class TransformerLayer(nn.Module):


    def __init__(self, 
                 embed_dim=192, 
                 num_heads=3, 
                 mlp_dim=384, 
                 dropout=0.1):
        
        super(TransformerLayer, self).__init__()

        self.la1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.la2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


    def forward(self, x):

        x = self.la1(x)
        out = self.attention(x,x,x,need_weights=False) + x
        out = self.mlp(self.la2(out)) + out

        return out
    


class Encoder(nn.Module):


    def __init__(self,
                 num_patches=144,
                 patch_dim=27, 
                 embed_dim=192, 
                 num_heads=3, 
                 mlp_dim=384,
                 num_layers=21, 
                 dropout=0.1):
    
        super(Encoder, self).__init__()

        self.proj = nn.Linear(patch_dim,embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        enc_list = [TransformerLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
    
    
    def forward(self,x):
        
        out = self.proj(x) + self.pos_emb
        out = self.enc(out)
        out = out[:,0]
        out = self.fc(out)
        return out



class Decoder(nn.Module):


    pass



class ViT_MAE(torch.nn.Module):



    def __init__(self,
                 image_size=36,
                 patch_size=3,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ):
        
        super().__init__()

        self.encoder = Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head)
        self.decoder = Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)


    def forward(self, x):

        features, backward_indexes = self.encoder(x)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask