import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class PRTreIDBoxMOTAdapter(nn.Module):
    def __init__(self, weights_path, prtreid_repo_path, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.hijacked_foreground_emb = None

        # Move path manipulation into constructor using passed path
        if prtreid_repo_path not in sys.path:
            sys.path.insert(0, prtreid_repo_path)

        try:
            from default_config import get_default_config
            from prtreid.models import build_model
        except ImportError as e:
            raise RuntimeError(
                f"Failed to load PRTreID modules. Check if prtreid_repo_path is correct: "
                f"'{prtreid_repo_path}'. Error: {e}"
            )

        # Load Config
        cfg = get_default_config()
        cfg.model.name = 'bpbreid'
        if hasattr(cfg.model, 'bpbreid'):
            cfg.model.bpbreid.backbone = 'hrnet32'

        # Build Model
        self.model = build_model(
            name='bpbreid',
            num_classes=1000,
            loss='triplet',
            pretrained=False,
            use_gpu=(device == 'cuda' or device == 'cuda:0'),
            config=cfg
        )

        # Patch the base forward pass
        if hasattr(self.model, 'base'):
            original_base_forward = self.model.base.forward
            def patched_base_forward(x, **kwargs):
                try:
                    return original_base_forward(x, return_featuremaps=True)
                except TypeError:
                    return original_base_forward(x)
            self.model.base.forward = patched_base_forward

        # Load Weights
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"Successfully loaded PRTreID weights from {weights_path}")
        except FileNotFoundError:
            print(f"Warning: PRTreID weights not found at {weights_path}.")

        self.model.to(self.device)
        self.model.eval()

        # Setup Normalization Tensors
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        def extract_foreground_features(module, input, output):
            self.hijacked_foreground_emb = output.detach().clone()

        try:
            target_layer = self.model.foreground_team_classifier.bn
            target_layer.register_forward_hook(extract_foreground_features)
        except AttributeError:
            pass

    def forward(self, x):
        x = x / 255.0
        x = (x - self.pixel_mean) / self.pixel_std
        if x.shape[2:] != (256, 128):
            x = F.interpolate(x, size=(256, 128), mode='bilinear', align_corners=False)

        with torch.no_grad():
            outputs = self.model(x)
            global_emb = outputs[0]['globl']
            if global_emb.dim() > 2:
                global_emb = F.adaptive_avg_pool2d(global_emb, 1).flatten(1)
            normalized_features = F.normalize(global_emb, p=2, dim=1)

        return normalized_features

    def get_features(self, xyxys, img):
        if len(xyxys) == 0:
            return torch.empty((0, 512)).to(self.device)

        crops = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for box in xyxys:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (128, 256))
            crops.append(crop.transpose(2, 0, 1))

        crops_tensor = torch.tensor(np.array(crops), dtype=torch.float32).to(self.device)
        return self.forward(crops_tensor)
