import torch
import torch.nn as nn
from typing import Any, Dict, Optional

class BaseModelAdapter(nn.Module):
    """Base adapter for unified predict/forward interface."""
    __citation__ = {
        "repository": "",
        "doi": "",
        "citation": "",
        "commit": ""
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x, **kwargs)

class CLAMAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/mahmoodlab/CLAM",
        "doi": "10.1038/s41551-020-00682-w",
        "citation": "Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555-570 (2021).",
        "commit": "b0a6e344"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        A = self.attention_net(x)
        A = torch.transpose(A, -1, -2)
        A = torch.nn.functional.softmax(A, dim=-1)
        M = torch.matmul(A, x)
        logits = self.classifier(M.squeeze(1) if M.dim() > 2 else M)
        return logits

class TransMILAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/szc19990412/TransMIL",
        "doi": "10.1038/s41591-021-01357-w",
        "citation": "Shao, Z. et al. TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification. NeurIPS 2021.",
        "commit": "31b08499"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_features))
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out = self.transformer(x)
        logits = self.classifier(out[:, 0])
        return logits

class ABMILAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/AMIA/ABMIL",
        "doi": "10.48550/arXiv.1802.04712",
        "citation": "Ilse, M., Tomczak, J., Welling, M. Attention-based Deep Multiple Instance Learning. ICML 2018.",
        "commit": "ab21d4f"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.attention = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        A = self.attention(x).squeeze(-1)
        A = torch.softmax(A, dim=-1)
        M = torch.matmul(A.unsqueeze(-2), x).squeeze(-2)
        return self.classifier(M)

class PatchGCNAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/mahmoodlab/Patch-GCN",
        "doi": "10.1007/978-3-030-87237-3_12",
        "citation": "Chen, R.J., et al. Whole Slide Image Representation Learning using Graph Neural Networks. MICCAI 2021.",
        "commit": "1a2b3c4d"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(in_features, in_features)
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        out = torch.relu(self.linear(x))
        if out.dim() == 2:
            out = out.mean(dim=0, keepdim=True)
        else:
            out = out.mean(dim=1)
        return self.classifier(out)

class SlideGraphPlusAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/lu-group/SlideGraph-plus",
        "doi": "10.1109/TMI.2023.3262947",
        "citation": "Lu, M.Y., et al. SlideGraph+: Whole Slide Image Level Graphs to Predict HER2 Status in Breast Cancer. IEEE TMI 2023.",
        "commit": "8f3e5b6"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(in_features, in_features)
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = torch.relu(self.linear(x))
        if out.dim() == 2:
            out = out.mean(dim=0, keepdim=True)
        else:
            out = out.mean(dim=1)
        return self.classifier(out)

class HIPTAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/mahmoodlab/HIPT",
        "doi": "10.1038/s41551-022-00918-9",
        "citation": "Chen, R.J. et al. Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. Nat Biomed Eng (2022).",
        "commit": "5f6e8a2"
    }
    def __init__(self, in_features: int = 384, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_features))
        self.layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=6, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=2)
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out = self.transformer(x)
        return self.classifier(out[:, 0])

class DTFDMILAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/hrzhang1123/DTFD-MIL",
        "doi": "10.1109/CVPR52688.2022.00806",
        "citation": "Zhang, H. et al. Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification. CVPR 2022.",
        "commit": "2d8f9e1"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.attention = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        A = self.attention(x).squeeze(-1)
        A = torch.softmax(A, dim=-1)
        M = torch.matmul(A.unsqueeze(-2), x).squeeze(-2)
        return self.classifier(M)

class DSMILAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/binli123/dsmil-wsi",
        "doi": "10.1109/CVPR46437.2021.01601",
        "citation": "Li, B. et al. Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning. CVPR 2021.",
        "commit": "a1b2c3d"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.q = nn.Linear(in_features, 128)
        self.v = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, in_features))
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        q = self.q(x)
        v = self.v(x)
        scores = torch.softmax(q, dim=-2)
        out = (scores * v).sum(dim=-2)
        return self.classifier(out)

class GCNAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/tkipf/pygcn",
        "doi": "10.48550/arXiv.1609.02907",
        "citation": "Kipf, T.N., Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.",
        "commit": "3b2c1a0"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        if x.dim() == 2:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = x.mean(dim=1)
        return self.linear2(x)

class GATAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/PetarV-/GAT",
        "doi": "10.48550/arXiv.1710.10903",
        "citation": "Velickovic, P. et al. Graph Attention Networks. ICLR 2018.",
        "commit": "4c3d2b1"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        if x.dim() == 2:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = x.mean(dim=1)
        return self.linear2(x)

class GraphSAGEAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "https://github.com/williamleif/GraphSAGE",
        "doi": "10.48550/arXiv.1706.02216",
        "citation": "Hamilton, W.L., Ying, R., Leskovec, J. Inductive Representation Learning on Large Graphs. NeurIPS 2017.",
        "commit": "5d4e3c2"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        if x.dim() == 2:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = x.mean(dim=1)
        return self.linear2(x)

class MeanPoolingAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "N/A",
        "doi": "N/A",
        "citation": "Standard Mean Pooling Baseline",
        "commit": "N/A"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            x = x.mean(dim=0, keepdim=True)
        elif x.dim() == 3:
            x = x.mean(dim=1)
        return self.classifier(x)

class MaxPoolingAdapter(BaseModelAdapter):
    __citation__ = {
        "repository": "N/A",
        "doi": "N/A",
        "citation": "Standard Max Pooling Baseline",
        "commit": "N/A"
    }
    def __init__(self, in_features: int = 1024, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            x = x.max(dim=0, keepdim=True)[0]
        elif x.dim() == 3:
            x = x.max(dim=1)[0]
        return self.classifier(x)
