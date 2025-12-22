import pandas as pd
import torch
from torch import nn
import shap
import numpy as np
from typing import Callable
from shap.maskers import Independent

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
        ref_age = data['Age'].values[0]
        mask = (train_data['Age'] >= ref_age - 10) & (train_data['Age'] <= ref_age + 10)
        matches_df = train_data[mask].copy()
        E_age = matches_df['Age'].mean()
        return matches_df.drop(['Age'], axis=1), E_age

    def get_top_shap(self, n, data, feats, train_data: pd.DataFrame, predict_func: Callable):
        top_shap = {}
        np.random.seed(0)
        torch.manual_seed(0)

        trn_data, E_age = self.get_nearby_data(train_data, data)
        #kernel_explainer = shap.KernelExplainer(predict_func, trn_data)
        #shap_values_trgt = kernel_explainer.shap_values(data.drop(['Age'], axis=1)).flatten()
        
        explainer = shap.Explainer(model=predict_func, algorithm="permutation", masker=Independent(trn_data, max_samples=100))
        explanation = explainer(data.drop(['Age'], axis=1))
        permutation = np.abs(np.array(explanation.values.flatten())).argsort()
        '''
        explanation = shap.Explanation(
            values=shap_values_trgt,
            base_values=E_age,
            data=data[feats].values[0],
            feature_names=feats)
        
        '''
        # Top-n values
        top_shap['values'] = np.array(explanation.values.flatten())[permutation][-n:].tolist()
        top_shap['data'] = np.array(explanation.data.flatten())[permutation][-n:].tolist()
        top_shap['feats'] = np.array(feats)[permutation][-n:].tolist()
        top_shap['explanation'] = explanation
        top_shap['explainer'] = explainer
        return top_shap
        