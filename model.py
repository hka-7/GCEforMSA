import torch
from torch import nn
import torch.nn.functional as F
from modules.encoders import LanguageEmbeddingLayer, SubNet, conv_SubNet, Clip

class GlobalTokenTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, target_dim, num_heads=4, num_layers=3):
        super(GlobalTokenTransformerEncoder, self).__init__()
        
        
        self.projection_input = nn.Linear(input_dim, embed_dim)
        

        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection_utt = nn.Linear(embed_dim, 64)
        self.projection_seg = nn.Linear(embed_dim, target_dim)

    def forward(self, features, lengths):
        features = features.permute(1,0,2)
        features = self.projection_input(features)
        
        batch_size = features.size(0)
        
        max_len = max(lengths)
        padded_features = torch.zeros(batch_size, max_len, features.size(2)).to(features.device)
        
        for i in range(batch_size):
            padded_features[i, :lengths[i], :] = features[i, :lengths[i], :]
        
        attention_mask = torch.zeros(batch_size, max_len).to(features.device)
        for i in range(batch_size):
            attention_mask[i, :lengths[i]] = 1
        
        global_token = self.global_token.repeat(batch_size, 1, 1)  # (batch_size, 1, embed_dim)
        features_with_token = torch.cat([global_token, padded_features], dim=1)  # (batch_size, 1 + max_len, embed_dim)
        
        global_token_mask = torch.ones(batch_size, 1).to(features.device)
        full_attention_mask = torch.cat([global_token_mask, attention_mask], dim=1)  # (batch_size, 1 + max_len)
        
        features_with_token = features_with_token.permute(1, 0, 2)  # (seq_len + 1, batch_size, embed_dim)
        
        transformer_output = self.transformer_encoder(features_with_token, src_key_padding_mask=(full_attention_mask == 0))
        
        utt = self.projection_utt(transformer_output[0,:,:])
        seg = self.projection_seg(transformer_output[1:,:,:])
        
        
        return utt, seg 
       

class DistillationKernel_3modality(nn.Module):
    def __init__(self, hp, n_classes, hidden_size, gd_size, to_idx, from_idx,
                gd_prior, gd_reg, w_losses, metric, alpha, mode):
        super().__init__()
        self.args = hp
        self.error_score = torch.zeros(3).cuda()
        self.error_value = torch.zeros(3).cuda()
        self.W_repr = nn.Linear(hidden_size, gd_size)
        self.W_edge_only_repr = nn.Linear(gd_size * 2, 1)

        self.gd_size = gd_size
        self.to_idx = to_idx
        self.from_idx = from_idx
        self.alpha = alpha
        self.mad = MAD()
        self.criterion = nn.L1Loss(reduction="mean")
        self.start = 1
        self.a = torch.tensor(1).cuda()
        self.from_m = self.to_m = [0,1,2]
        self.b = torch.tensor(0.7).cuda()
        self.mode = mode

    def relation_transfer_3modality(self, detach_logits, gt, reprs, ex = 0):
        distilled_samples = 0
        self.logger = ex
        self.error_value[0] = self.criterion(detach_logits[0], gt) # (1,)
        self.error_value[1] = self.criterion(detach_logits[1], gt)
        self.error_value[2] = self.criterion(detach_logits[2], gt)
        self.error_score = self.error_value
                
        min_value, min_index = torch.min(self.error_score, dim=0)
        min_error_score = min_value.item() # min_value
        min_error_modality = min_index.item() # 0/1/2
        
        loss_repr = torch.tensor(0.0, device=detach_logits[0].device)
        for fro in self.from_m:
            for t in self.to_m:
                if fro == min_error_modality and self.error_score[fro] < self.b *self.error_score[t]:
                    loss_repr += self.transfer_relation(reprs[fro].detach(), reprs[t]) 
                    batch_size = reprs[fro].size(0)
                    distilled_samples += batch_size
        return loss_repr, self.error_score, distilled_samples
        
    def transfer_relation(self, detach_reprs_from, reprs_to):
        self.weight = self.reprs2weight(detach_reprs_from, reprs_to) # (batch_size,)
        
        loss_repr = self.mad(reprs_to, detach_reprs_from, self.weight)
        return loss_repr
        
    def reprs2weight(self, reprs_from, reprs_to):
        batch_size, hidden_size = reprs_from.size()
        reprs_from = F.normalize(reprs_from, p=2, dim=1)
        reprs_to = F.normalize(reprs_to, p=2, dim=1)
        reprs_cat = torch.stack([reprs_from, reprs_to], dim=0) # (2, batch_size, hidden_size)
        z_reprs = self.W_repr(reprs_cat.view(2 * batch_size, -1)) # (2*batch_size, hidden_size)
        # (n_modalities, batch_size, self.gd_size)
        z = z_reprs.view(2, batch_size, self.gd_size)

        e = self.W_edge_only_repr(torch.cat((z[0], z[1]), dim=1)) # (batch_size, 2* self.gd_size)
        edges = e # (batch_size,)
        edges = torch.sigmoid(edges * self.alpha)
        
        return edges

class MAD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fm_s, fm_t, logit_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        fm_s_normalized = F.normalize(fm_s, p=2, dim=1)
        G_s = torch.mm(fm_s_normalized, fm_s_normalized.t())

        fm_t = fm_t.view(fm_t.size(0), -1) 
        fm_t_normalized = F.normalize(fm_t, p=2, dim=1)
        G_t = torch.mm(fm_t_normalized, fm_t_normalized.t())

        loss = F.mse_loss(G_s, G_t,reduction='none')
        loss = torch.sum(loss * logit_t) / fm_s.size(0)
        
        return loss

  
def get_distillation_kernel_3modality(hp, n_classes = 1,
                            hidden_size = 512,
                            gd_size = 256,
                            to_idx = [],
                            from_idx = [],
                            gd_prior = 0,
                            gd_reg = 0,
                            w_losses = 0,
                            metric = 0,
                            alpha=1 / 8,
                            mode = 0):
  return DistillationKernel_3modality(hp, n_classes, hidden_size, gd_size, to_idx, from_idx,
                            gd_prior, gd_reg, w_losses, metric, alpha, mode)

class GCE(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp
        self.dim_l = 768
        self.dim_v = 64
        self.dim_a = 64

        # encode
        self.text_enc = LanguageEmbeddingLayer(hp)
        self.transformer_encoder_v = GlobalTokenTransformerEncoder(hp.d_vin, 24, 32)
        self.transformer_encoder_a = GlobalTokenTransformerEncoder(hp.d_ain, 16, 32)
        
        # detach lva
        self.fusion_prj_l = SubNet(  # (linear_1): Linear(in_features=768, out_features=128, bias=True)
            # (linear_2): Linear(in_features=128, out_features=128, bias=True)
            # (linear_3): Linear(in_features=128, out_features=1, bias=True)
            in_size=768,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fusion_prj_v = SubNet(
            in_size=self.dim_v,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fusion_prj_a = SubNet(
            in_size=self.dim_a,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )

        # contrastive
        self.ta_clip = Clip(self.dim_l, self.dim_a)
        self.tv_clip = Clip(self.dim_l, self.dim_v)
        self.av_clip = Clip(self.dim_a, self.dim_v)

        # fusion
        self.dim_conv_out = 64
        self.my_conv_lv = conv_SubNet(self.dim_l, self.dim_v, self.dim_conv_out)
        self.my_conv_va = conv_SubNet(self.dim_v, self.dim_a, self.dim_conv_out)
        self.my_conv_la = conv_SubNet(self.dim_l, self.dim_a, self.dim_conv_out)
        # relation filter
        self.fusion_modal_relation_trans = get_distillation_kernel_3modality(hp, hidden_size = 64 ,gd_size = 64, mode = 'fusion')
        # fusion detach
        self.fusion_prj_fusion_lv = SubNet(
            in_size=self.dim_conv_out,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fusion_prj_fusion_la = SubNet(
            in_size=self.dim_conv_out,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fusion_prj_fusion_va = SubNet(
            in_size=self.dim_conv_out,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )

        self.getresNet_uni = nn.Linear(self.dim_l+self.dim_v+self.dim_a, 512)
        self.getresNet_pair = nn.Linear(64*3, 512)
        self.fusion_uni_pair = nn.Conv1d(
            in_channels=1024, out_channels=512, kernel_size=1)
        self.fusion_prj = SubNet(
            in_size=512,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """  
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type,
                                 bert_sent_mask)
        text = enc_word[:, 0, :]  # (batch_size, emb_size) 128, 768
        visual, _ = self.transformer_encoder_v(visual, v_len) # 128, 64
        acoustic, _ = self.transformer_encoder_a(acoustic, a_len) # 128, 64
     
        _, self.prj_l = self.fusion_prj_l(text.detach()) # 128, 1
        _, self.prj_v = self.fusion_prj_v(visual.detach()) # 128, 1
        _, self.prj_a = self.fusion_prj_a(acoustic.detach()) # 128, 1
        
        # fusion
        massagehub_lv = self.my_conv_lv(text, visual).squeeze() # 128, 64
        massagehub_la = self.my_conv_la(text, acoustic).squeeze() # 128, 64
        massagehub_va = self.my_conv_va(visual, acoustic).squeeze() # 128, 64
        
        # detach_pred
        # fusion_lv,fusion_la,fusion_va: list
        # self.prj_fusion_lv, self.prj_fusion_la, self.prj_fusion_va: 128, 1
        fusion_lv, self.prj_fusion_lv = self.fusion_prj_fusion_lv(massagehub_lv.detach())
        fusion_la, self.prj_fusion_la = self.fusion_prj_fusion_la(massagehub_la.detach())
        fusion_va, self.prj_fusion_va = self.fusion_prj_fusion_va(massagehub_va.detach())

        # relation_transfer
        gt = y # 128, 1
        fusion_detach_logits = torch.stack([self.prj_fusion_lv, self.prj_fusion_la, self.prj_fusion_va], dim=0) # 3, 128, 1
        fusion_reprs = torch.stack([massagehub_lv, massagehub_la, massagehub_va], dim=0) # 3, 128, 64
        fusion_loss_repr, error_score, distilled_samples = self.fusion_modal_relation_trans.relation_transfer_3modality(fusion_detach_logits, gt, fusion_reprs, ex) # fusion_loss_repr: torch.Size([]); error_score: torch.Size([3]); distilled_samples:list
        
        # pred
        # uni
        tav = torch.cat([text, acoustic, visual], dim=1) # 128, 896
        qwer = self.getresNet_uni(tav) # 128, 512
        # pair
        fusion_tav = torch.cat([massagehub_lv, massagehub_va, massagehub_la], dim=1)
        fusion_tav = self.getresNet_pair(fusion_tav) # 512
        # fusion uni & pair
        xxx = torch.cat([qwer, fusion_tav], dim=1).unsqueeze(2) # 128, 1024, 1
        res_mm = self.fusion_uni_pair(xxx).squeeze(2) # 128, 512
        fusion, preds = self.fusion_prj(res_mm) # fusion: list; preds: 128, 1
        
        # loss
        clip_ta = self.ta_clip(text, acoustic)
        clip_tv = self.tv_clip(text, visual)
        clip_av = self.av_clip(visual, acoustic)
        loss_contrastive = clip_ta + clip_tv + clip_av
        loss_relation_filter = fusion_loss_repr
        
        # detach_pred
        self.ori_prj_list = [self.prj_l, self.prj_v, self.prj_a]
        self.fusion_prj_list = [self.prj_fusion_lv, self.prj_fusion_la, self.prj_fusion_va]

        fea = {'lv': res_mm,'la': res_mm,'va': res_mm,'all': res_mm}
        fea_last =  {'lv': fusion_lv,'la': fusion_la,'va': fusion_va,'all': fusion}
        return [preds, self.ori_prj_list, self.fusion_prj_list], [loss_contrastive, loss_relation_filter], [error_score], [fea, fea_last], [distilled_samples]
        
