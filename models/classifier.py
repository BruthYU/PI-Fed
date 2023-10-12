from typing import *

import torch
from torch import Tensor, nn

from models.spg import SPG
from utils import assert_type


class SPGClassifier(nn.Module):
    def __init__(self, list__ncls: List[int], dim: int, idx_task: int, list__spg: List[SPG]):
        super().__init__()

        # self.list__classifier = nn.ModuleList()
        # for ncls in list__ncls:
        #     head = _TaskHead(nn.Linear(dim, ncls), list__spg=list__spg)
        #     self.list__classifier.append(head)

        # TODO Single Task Head
        self.classifier = _TaskHead(nn.Linear(dim,list__ncls[idx_task]),list__spg=list__spg)
        # endfor
    # enddef

    def forward(self, x: Tensor) -> Tensor:
        assert_type(x, Tensor)


        # clf = self.list__classifier[idx_task]
        x = x.view(x.shape[0], -1)
        # out = clf(x)
        out = self.classifier(x)

        return out
    # enddef

    def modify_grads(self, args: Dict[str, Any]) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10000)

        for _, module in self.list__classifier.named_modules():
            if isinstance(module, _TaskHead):
                module.softmask(idx_task=self.idx_task)
            # endif
        # endfor
    # enddef


class _TaskHead(nn.Module):
    def __init__(self, classifier: nn.Linear, list__spg: List[SPG]):
        super().__init__()

        self.classifier = classifier
        self.list__spg = list__spg

        self.dict__idx_task__red = {}

        self.device = None
    # enddef

    def forward(self, x: Tensor) -> Tensor:
        if self.device is None:
            self.device = x.device
        # endif

        return self.classifier(x)
    # enddef

    def softmask(self, idx_task: int):
        if idx_task not in self.dict__idx_task__red.keys():
            list__amax = []
            for spg in self.list__spg:
                dict__amax = spg.a_max(idx_task=idx_task, latest_module=spg.target_module)
                if dict__amax is not None:
                    for _, amax in dict__amax.items():
                        list__amax.append(amax.view(-1))
                    # endfor
                # endif
            # endfor

            if len(list__amax) > 0:
                amax = torch.cat(list__amax, dim=0)
                mean_amax = amax.mean()
                modification = (1 - mean_amax).cpu().item()
            else:
                modification = 1
            # endif

            self.dict__idx_task__red[idx_task] = modification
        else:
            modification = self.dict__idx_task__red[idx_task]
        # endif

        if False:
            print(f'[classifier] modification: {modification}')
        # endif

        for n, p in self.classifier.named_parameters():
            if p.grad is not None:
                p.grad.data *= modification
            # endif
        # endfor
    # enddef
