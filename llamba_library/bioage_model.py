import pandas as pd
import torch
from torch import nn
import shap
import numpy as np

class BioAgeModel():
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.model.freeze()

    def inference(self, data: pd.DataFrame, device: torch.device):
        self.model.to(device)
        self.model.eval()
        if str(device) == "cuda":
            res = self.model(torch.from_numpy(data.values)).cuda().detach().numpy().ravel()
        else:
            res = self.model(torch.from_numpy(data.values)).cpu().detach().numpy().ravel()
        return res
    
    def get_nearby_data(self, train_data: pd.DataFrame, data: pd.DataFrame):
        ref_age = data.loc[0, 'Age']
        mask = (train_data['Age'] >= ref_age - 10) & (train_data['Age'] <= ref_age + 10)
        matches_df = train_data[mask].copy()
        E_age = matches_df['Age'].mean()
        return matches_df.drop(['Age'], axis=1), E_age

    def get_top_shap(self, n, data, feats, train_data: pd.DataFrame, predict_func):
        top_shap = {}
        np.random.seed(0)
        torch.manual_seed(0)

        trn_data, E_age = self.get_nearby_data(train_data, data)
        kernel_explainer = shap.KernelExplainer(predict_func, trn_data)
        shap_values_trgt = kernel_explainer.shap_values(data.drop(['Age'], axis=1))[0].flatten()

        explanation = shap.Explanation(
            values=shap_values_trgt,
            base_values=E_age,
            data=data.loc[0, feats].values,
            feature_names=feats)

        permutation = np.abs(np.array(explanation.values)).argsort()
        
        # Top-n values
        top_shap['values'] = np.array(explanation.values)[permutation][-n:].tolist()
        top_shap['data'] = np.array(explanation.data)[permutation][-n:].tolist()
        top_shap['feats'] = np.array(feats)[permutation][-n:].tolist()
        top_shap['explanation'] = explanation
        top_shap['explainer'] = kernel_explainer
        return top_shap