"""Neural network architecture for the flow model."""
import torch
import math
import torch.nn.functional as F
from torch import nn

# from data import utils as du
import torch.nn.init as init
import numpy as np
from core.config.config import Config
from core.model.watermark import ipa_pytorch
from core.evaluation.utils import rotate_molecule, reflect_molecules, translate_molecule


class WaterMarkModule(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.cfg = config
        self.device=device
        self.batch_size = self.cfg.optimization.batch_size
        self._encoder_decoder_cfg = config
        self.encoder = Encoder(self._encoder_decoder_cfg)
        self.decoder = Decoder(self._encoder_decoder_cfg)

        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)

        self.fn = nn.MSELoss()
    


    def recover_coordinates(self, D, k=3):

        # Number of points
        bs, n = D.shape[0], D.shape[1]

        X = torch.zeros(bs, n, k).to(self.device)
        # Centering matrix
        
        for b in range(bs):
            # Compute Gram matrix (B)
            D_squared = D[b] ** 2
            H = (torch.eye(n) - torch.ones(n, n) / n).to(self.device)
            B = -0.5 * H @ D_squared @ H
        
            # Eigenvalue decomposition
            eigvals, eigvecs = torch.linalg.eigh(B)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigvals, descending=True)

            eigvals = torch.clamp(eigvals[idx], min=0.0)
            eigvecs = eigvecs[:, idx]
            
            # Select top k eigenvalues and corresponding eigenvectors
            L = torch.diag(torch.sqrt(eigvals[:k]))
            V = eigvecs[:, :k]
            
            # Compute the coordinate matrix
            X[b] = V @ L
        
        return X

    def fn_enc(self, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs()
        return loss
    
    def fn_enc_dist(self, x_pred, x):
        x_pred = x_pred.reshape(self.batch_size, -1, x_pred.shape[-1])
        x = x.reshape(self.batch_size, -1, x.shape[-1])
        pre_dist = torch.cdist(x_pred, x_pred, p=1)
        ori_dist = torch.cdist(x, x, p=1)
        loss = torch.abs(pre_dist - ori_dist)
        return loss
    
    def forward(self, position, charges, atom_types, edge_index, watermark, attack=False, mode=0, axis=0, angle=0, trans=0):

        pred_pos_enc = self.encoder(position, watermark, charges, atom_types, edge_index)
        pred_pos = pred_pos_enc.reshape(self.batch_size, -1, pred_pos_enc.shape[-1])
        if attack:
            if mode == 0:
                pred_pos = rotate_molecule(pred_pos, axis, angle)
            elif mode == 1:
                pred_pos = translate_molecule(pred_pos, axis, trans)
            elif mode == 2:
                pred_pos = reflect_molecules(pred_pos, axis)
            else:
                raise NotImplementedError
            
        distance_matrix = torch.cdist(pred_pos, pred_pos, p=2)
        recovered_coords = self.recover_coordinates(distance_matrix)
        recovered_coords = recovered_coords.reshape(-1, recovered_coords.shape[-1])
        pred_code = self.decoder(recovered_coords, charges, atom_types, edge_index)

        original_classes = (watermark > 0.5).float()
        predicted_classes = (pred_code > 0.5).float() 
        recovery = ((predicted_classes == original_classes).float().mean(dim=-1))
   
        pred_pos_enc = pred_pos_enc.reshape(-1, pred_pos_enc.shape[-1])
        
        return recovery, pred_pos_enc, pred_code

        
    def loss_one_step(self, position, charges, atom_types, edge_index, watermark):
    
        recovery, pred_pos, pred_code = self(position, charges, atom_types, edge_index, watermark, attack=True)  
        post_loss = self.fn_enc(pred_pos, position)
        code_loss = self.fn(pred_code, watermark)

        return post_loss, code_loss, recovery, pred_pos, pred_code

        

def coord2diff(x, edge_index, norm_constant=1):
    x_t = x.reshape(-1, 3)
    row, col = edge_index
    coord_diff = x_t[row] - x_t[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor=1, aggregation_method="sum"):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding



class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.embed_size = 0

        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Linear(5, self.c_s) # Always 5 because of 5 kinds of molecules 
            self.embed_size += self.c_s
        if self._cfg.embed_position:
            self.embed_size += self.c_s

        self.linear = nn.Linear(self.embed_size, 3)
            


    def forward(self, charges, atom_types):

        # charges [b, 1, n, 1]   atom_types  [b, 1, n, 5]
        bs, c, num_mol, n_features = atom_types.shape
        device = atom_types.device
        # [b, num_mol, c_pos_emb]
        pos = torch.arange(num_mol, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)

        pos_emb = pos_emb.repeat([bs, 1, 1, 1])
        charge_emb = pos_emb * charges # [b, 1, num_mol, c_pos_emb]

        # [b, num_mol, c_timestep_emb]
        input_feats = [charge_emb]
        # input_feats.append(position_embed)
        if self._cfg.embed_aatype:
            atom_types_embed = self.aatype_embedding(atom_types)
            input_feats.append(atom_types_embed)
        output = self.linear(torch.cat(input_feats, dim=-1))
        return output

class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(4, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)
        self.normalization_factor = 1
        self.aggregation_method = "sum"
        total_edge_feats = self.feat_dim * 3
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )
        self.final_layer = nn.Linear(self.c_s, 3)

    def embed_relpos(self, pos):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = get_index_embedding(rel_pos, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, init_node_embed, edge_index, radial):

        num_batch, c, num_res, feat_dim = init_node_embed.shape
        row, col = edge_index
        agg = unsorted_segment_sum(
            radial,
            row,
            num_segments=num_batch * num_res,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        agg = agg.reshape(num_batch, 1, num_res, -1)
        p_i = self.linear_s_p(torch.cat([init_node_embed, agg], dim=-1))
        cross_node_feats = self._cross_concat(p_i.squeeze(1), num_batch, num_res)
        pos = torch.arange(num_res, device=init_node_embed.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(pos)
        all_edge_feats = torch.concat([cross_node_feats, relpos_feats], dim=-1)
        edge_feats = self.edge_embedder(all_edge_feats)
        edge_final = self.final_layer(edge_feats)
        edge_final = torch.sum(edge_final, dim=1, keepdim=True)

        return edge_final




def conv2d(in_channel, out_chanenl):
    return nn.Conv2d(in_channels=in_channel, 
                     out_channels=out_chanenl, 
                     stride=1, 
                     kernel_size=3, 
                     padding=1)

def ConvBNLReLU(in_channel, out_channel):
    return nn.Sequential(conv2d(in_channel, out_channel), 
                         nn.BatchNorm2d(out_channel, track_running_stats=False), 
                         nn.LeakyReLU(inplace=True))


class Encoder(nn.Module):

    def __init__(self, model_conf):
        super(Encoder, self).__init__()
        self.cfg = model_conf
        self.model_conf = model_conf.encoder_decoder
        self.conv_channels = self.model_conf.node_embed_size

        self.node_embedder = NodeEmbedder(self.model_conf.node_features) # atom embedder
        self.edge_embedder = EdgeEmbedder(self.model_conf.edge_features) # edge embedder

        self.first_layer = nn.Sequential(conv2d(1, self.conv_channels))
        self.second_layer = nn.Sequential(conv2d(self.conv_channels + 2, self.conv_channels))
        self.Dense_layer1 = Densecross(self.conv_channels + self.model_conf.watermark_emb, self.conv_channels)
        self.Dense_layer2 = Densecross(self.conv_channels * 2 + self.model_conf.watermark_emb, self.conv_channels)
        self.Dense_layer3 = Densecross(self.conv_channels * 3 + self.model_conf.watermark_emb, self.conv_channels)
        self.Dense_atten1 = Densecross(self.conv_channels, self.conv_channels * 2)
        self.Dense_atten2 = Densecross(self.conv_channels * 2, self.conv_channels * 3)
        self.Dense_atten3 = Densecross(self.conv_channels * 3, self.conv_channels)

        self.third_layer = nn.Sequential(
            ConvBNLReLU(self.conv_channels + self.model_conf.watermark_emb, self.conv_channels),
            ConvBNLReLU(self.conv_channels, self.model_conf.watermark_emb)
        )
        self.forth_layer = nn.Sequential(
            ConvBNLReLU(self.conv_channels, self.conv_channels),
            ConvBNLReLU(self.conv_channels, self.model_conf.watermark_emb),
            nn.Softmax(dim=1)
        )

        self.final_layer = conv2d(self.model_conf.watermark_emb, 1)
        

    def forward(self, position, message, charges, atom_types, edge_index):
        radial, coord_diff = coord2diff(position, edge_index)
        position = position.reshape(self.cfg.optimization.batch_size, -1, position.shape[-1])
        position = position.unsqueeze(1) # [bs, 1, n, 3]
        bs, c, num_mol, num_pos = position.shape

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1, -1, num_mol, num_pos)

        charges = charges.reshape(self.cfg.optimization.batch_size, -1, charges.shape[-1]).unsqueeze(1)  # [bs, 1, n, 1]
        atom_types = atom_types.reshape(self.cfg.optimization.batch_size, -1, atom_types.shape[-1]).unsqueeze(1) # [bs, 1, n, 5]
        node_embed = self.node_embedder(charges, atom_types)  # [bs, 1, num_mol, 3]
        edge_embed = self.edge_embedder(node_embed, edge_index, radial) # [bs, 1, num_mol, 3]


        feature_pos = self.first_layer(position)
        feature_embed = self.second_layer(torch.cat((feature_pos, node_embed, edge_embed), dim=1))
        feature1 = self.Dense_layer1(torch.cat([feature_embed, expanded_message], dim=1))
        feature2 = self.Dense_layer2(torch.cat([feature_embed, expanded_message, feature1], dim=1))
        feature3 = self.Dense_layer3(torch.cat([feature_embed, expanded_message, feature1, feature2], dim=1))
        feature4 = self.third_layer(torch.cat([feature3, expanded_message], dim=1))
        feature_atten1 = self.Dense_atten1(feature_embed)
        feature_atten2 = self.Dense_atten2(feature_atten1)
        feature_atten3 = self.Dense_atten3(feature_atten2)
        feature_mask = self.forth_layer(feature_atten3)
        feature_final = feature4 * feature_mask
        high_level_position = self.final_layer(feature_final)
        pred_position = high_level_position + position

        pred_position = pred_position.squeeze(1)
        pred_position = pred_position.reshape(-1, pred_position.shape[-1])

        return pred_position




class Decoder(nn.Module):

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)
    def __init__(self, model_conf):
        super(Decoder, self).__init__()
        self.cfg = model_conf

        self.model_conf = model_conf.encoder_decoder
        self.channels = self.model_conf.node_embed_size

        self.node_embedder = NodeEmbedder(self.model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(self.model_conf.edge_features)

        self.first_layer = ConvBNLReLU(1, self.channels)
        self.second_layer = ConvBNLReLU(self.channels + 2, self.channels)
        self.third_layer = ConvBNLReLU(self.channels * 2, self.channels)
        self.fourth_layer = ConvBNLReLU(self.channels * 3, self.channels)
        self.fivth_layer = ConvBNLReLU(self.channels, self.model_conf.watermark_emb)

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(self.model_conf.watermark_emb, self.model_conf.watermark_emb)
    

    def compute_distance_matrix(self, coords):
        return torch.cdist(coords, coords).unsqueeze(1)


    def forward(self, position, charges, atom_types, edge_index):

        radial, coord_diff = coord2diff(position, edge_index)
        position = position.reshape(self.cfg.optimization.batch_size, -1, position.shape[-1])
        position = position.unsqueeze(1) # [bs, 1, n, 3]
        # bs, c, num_mol, num_pos = position.shape

        charges = charges.reshape(self.cfg.optimization.batch_size, -1, charges.shape[-1]).unsqueeze(1)  # [bs, 1, n, 1]
        atom_types = atom_types.reshape(self.cfg.optimization.batch_size, -1, atom_types.shape[-1]).unsqueeze(1) # [bs, 1, n, 5]
        node_embed = self.node_embedder(charges, atom_types)  # [bs, 1, num_mol, 3]
        edge_embed = self.edge_embedder(node_embed, edge_index, radial) # [bs, 1, num_mol, 3]


        feature_with_w = self.first_layer(position)
        feature1 = self.second_layer(torch.cat((feature_with_w, node_embed, edge_embed), dim=1))
        feature2 = self.third_layer(torch.cat([feature_with_w, feature1], dim=1))
        feature3 = self.fourth_layer(torch.cat([feature_with_w, feature1, feature2], dim=1))
        x = self.fivth_layer(feature3)
        x = self.pooling(x)
        x = self.linear(x.squeeze_(3).squeeze_(2))
        return x
      
 

class Densecross(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Densecross, self).__init__()
        temp_channels = 4 * out_channels
        self.conv1 = ConvBNLReLU(in_channels, temp_channels)
        self.conv2 = ConvBNLReLU(in_channels + temp_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(torch.cat([x, out], dim=1))
        return out
    


