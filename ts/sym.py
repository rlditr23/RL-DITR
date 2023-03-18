import torch
import json
from pathlib import Path


if (__name__ == '__main__') or (__package__ == ''):
    from datasets.text_dataset import TextClfDataset
else:
    from .datasets.text_dataset import TextClfDataset

class SymptomExtractor(object):
    def __init__(self, model_dir, threshold=0.5, device='cpu'):
        super(SymptomExtractor, self).__init__()
        model_dir = Path(model_dir)
        self.model_dir = model_dir

        self.threshold = threshold
        self.sym_list = (Path(model_dir)/'multilabel.txt').read_text().strip().splitlines()
        self.model_config = json.loads((model_dir/'model.config.json').read_text())
        self.model = torch.jit.load(model_dir/'scripted_model.zip')
        self.max_seq_len = self.model_config['max_seq_len']
        self.device = device

    def text2tensor(self, text):
        datset_single = TextClfDataset(text=[text], label=[0], 
                                       model_name_or_path=self.model_dir, max_seq_len=self.max_seq_len)
        x, y = datset_single[0]
        x = torch.Tensor(x).long()
        x = torch.stack([x])
        return x

    def transform(self, text):
        x = self.text2tensor(text)
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits) # [1, xx]
        probs = probs[0] # [xx]
        probs = probs.to('cpu')
        result = [sym for sym, prob in zip(self.sym_list,probs) if prob>self.threshold]
        return result

    
