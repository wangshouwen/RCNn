import os
import math
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP',
                      "vision_depth": 0, "vision_ctx": 0,
                      "language_depth": 0, "language_ctx": 0
                      }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, if_embedding=True):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ClsnameLearner(nn.Module):
    def __init__(self, cfg, cls_num, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        n_clsn = cfg.TRAINER.TEXT.N_CLSN
        ctx_init = cfg.TRAINER.TEXT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        clsn_vectors = torch.empty(cls_num, n_clsn, ctx_dim, dtype=dtype)
        nn.init.normal_(clsn_vectors, std=0.02)
        clsn_sign = " ".join(["X"] * n_clsn)
        self.clsn = nn.Parameter(clsn_vectors)

        prompt = ctx_init + " " + clsn_sign + "."
        tokenized_prompt = clip.tokenize(prompt)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompt).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + n_clsn + 1:, :])
        self.cls_num = cls_num
        self.tokenized_prompt = tokenized_prompt

    def forward(self):
        clsn = self.clsn
        if clsn.dim() == 2:
            clsn = clsn.unsqueeze(0).expand(self.cls_name, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if prefix.shape[0] == 1:
            prefix = prefix.expand(self.cls_num, -1, -1)
            suffix = suffix.expand(self.cls_num, -1, -1)   

        prompts = torch.cat([prefix, clsn, suffix], dim=1)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = ClsnameLearner(cfg, len(classnames), clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompt
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual

        self.logit_scale = 20
        self.dtype = clip_model.dtype
        self.clip_model = clip_model

    def encode_visual_features(self, image):
        image_features, image_features_ori, attn_weights = self.image_encoder(image.type(self.dtype)) # NLD, ND, 12*N * L+n_ctx * L+n_ctx

        image_features_ = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, image_features_, attn_weights
    
    def encode_text_features(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        cls_embeddings = self.text_encoder(prompts, tokenized_prompts)

        cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)

        return cls_embeddings
    
    def forward(self, caption):
        text_features = self.text_encoder(caption, None, if_embedding=False)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        cls_embeddings = self.text_encoder(prompts, tokenized_prompts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)

        return text_features, cls_embeddings
    
    def forward_for_open(self, image, text_features):
        logit_scale = self.logit_scale
        image_features, image_features_, attn_weights = self.encode_visual_features(image)
        logit_local = logit_scale * image_features[:, 1:, :] @ text_features.t()
        logit_local = logit_local.transpose(1, 2)

        w_avg = F.softmax(logit_local, dim=2) 
        logit_local = torch.sum(logit_local * w_avg, dim=2) # N * C
        
        logit_glob = logit_scale * image_features_ @ text_features.t() 
        logit_final = 0.5 * (logit_glob + logit_local)
        
        return logit_final
    
    def forward_for_test(self, image, text_features):
        image_features, image_features_, attn_weights = self.encode_visual_features(image)

        logit_local = image_features[:, 1:, :] @ text_features.t()
        logit_local = logit_local.transpose(1, 2) # N*c*hw 
        _, C, HW = logit_local.shape
        H = int(math.sqrt(HW))
        logit_local = logit_local.view(-1, C, H, H)
        
        logit_glob = image_features_ @ text_features.t() 
    
        return logit_local, logit_glob

    def aggregatorplus(self, x, x_list, count_list=[2,3]):
        logit_scale = self.logit_scale
        logit_local, logit_glob = x
        list_local, list_glob = list(zip(*x_list))
        multiscale_list_local = []
        multiscale_list_glob = []
        for count in count_list:
            tmp_scale_l = list_local[: count*count]
            tmp_scale_g = list_glob[: count*count]
            img_cat = []
            for i in range(count):
                img_cat.append(torch.cat(tmp_scale_l[i*count:(i+1)*count], dim=-1))
            img_cat = torch.cat(img_cat, dim=-2)
            tmp_local = F.max_pool2d(img_cat, count, count)
            multiscale_list_local.append(tmp_local)

            tmp_glob = torch.stack(tmp_scale_g, dim=0)
            tmp_glob = tmp_glob.max(dim=0)[0]
            multiscale_list_glob.append(tmp_glob)
            list_local = list_local[count*count:]
            list_glob = list_glob[count*count:]

        multiscale_list_local.append(logit_local)
        logit_local = torch.stack(multiscale_list_local, dim=0)
        logit_local = logit_local.mean(dim=0)
        logit_local = logit_local.view(logit_local.size(0), logit_local.size(1), -1)
        logit_local = logit_scale * logit_local
        w_avg = F.softmax(logit_local, dim=2) 
        logit_local = torch.sum(logit_local * w_avg, dim=2)

        multiscale_list_glob.append(logit_glob)
        logit_glob = torch.stack(multiscale_list_glob, dim=0)
        logit_glob = logit_glob.mean(dim=0)
        logit_glob = logit_glob * logit_scale

        logit_final = 0.5 * (logit_glob + logit_local)

        return logit_final


def build_text_model(cfg, classnames):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    
    model = CustomCLIP(cfg, classnames, clip_model)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    return model