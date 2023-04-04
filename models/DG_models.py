from models.CNN_Ag_models import cnn_Ag
from models.xresnet_1d_simple import xresnet18,xresnet34,xresnet50,xresnet101,xresnet152
from models.inception_1d_simple import inception1d
from models.resnet_1d import resnet18,resnet34,resnet50,resnet101,resnet152
from models.resnet1d_wang import resnet1d_wang
from models.seresnet1d import seresnet_big_18,seresnet_big_34,seresnet_big_50,seresnet_big_101,seresnet_big_152
from models.densenet1d import densenet1d

def generate_model(base_model='', DG_method=None ,**kwargs):
    print('base_model='+base_model)
    if DG_method in [None,'IRM','cls_awr_ali','origin_add_fft_amp','origin_add_fft_phase','origin_add_fft_amp_phase']:
        return eval(base_model)(**kwargs)
    
    elif DG_method in ['DG_GR','DG_GR+IRM','DG_GR_ensemble','origin_add_fft_amp_DGGR','DGGR_smooth','origin_add_fft_amp_DGGR_smooth']:
        return eval(base_model)(DG_method='DG_GR', **kwargs)  # gradient reverse

    elif DG_method in['MMD','CORAL','CausIRL_MMD','CausIRL_CORAL']:
        return eval(base_model)(DG_method='MMD', **kwargs)

    elif DG_method=='my':
        return eval(base_model)(DG_method='my', **kwargs) # 和MMD的基础模型一样

    
#     elif DG_method=='mid_feat':
#         return eval(base_model)(DG_method='mid_feat', **kwargs)  # gradient reverse
    
    else:
        print('generate_model_error!')
    
# 'Adversarial learning'的方法包括文章：    
    
# Discriminative Domain Generalization with Adversarial Feature Learning
# 基于对抗性特征学习的深度识别域推广

# Adversarial Domain Generalization
# 具有对抗性多源域泛化的12导联心电图信号的分类





