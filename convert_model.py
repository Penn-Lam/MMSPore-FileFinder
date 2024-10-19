import torch
import mindspore as ms
from mindspore import Tensor, Parameter
from clip import clip

def convert_pytorch_to_mindspore(pt_path, ms_path):
    # 加载PyTorch模型
    pt_model = torch.jit.load(pt_path, map_location="cpu")
    
    # 创建MindSpore模型
    ms_model, _ = clip.load("ViT-B/32", device="CPU")

    # 获取PyTorch模型的参数
    pt_params = {}
    for name, param in pt_model.named_parameters():
        pt_params[name] = param.detach().numpy()

    # 转换参数
    ms_param_dict = {}
    for param in ms_model.get_parameters():
        name = param.name
        # 处理 transformer.resblock 参数
        if 'transformer.resblocks' in name:
            pt_name = name.replace('transformer.resblocks', 'transformer.transformer.resblocks')
            if pt_name in pt_params:
                ms_param_dict[name] = Parameter(Tensor(pt_params[pt_name]), name=name)
            else:
                print(f"Warning: Parameter {name} not found in PyTorch model")
                ms_param_dict[name] = param
        elif name in pt_params:
            ms_param_dict[name] = Parameter(Tensor(pt_params[name]), name=name)
        else:
            print(f"Warning: Parameter {name} not found in PyTorch model")
            ms_param_dict[name] = param

    # 加载参数到MindSpore模型
    ms.load_param_into_net(ms_model, ms_param_dict)

    # 保存MindSpore模型
    ms.save_checkpoint(ms_model, ms_path)

    print(f"Model converted and saved to {ms_path}")

if __name__ == "__main__":
    pt_path = "./models/ViT-B-32.pt"
    ms_path = "./models/ViT-B-32.ckpt"
    convert_pytorch_to_mindspore(pt_path, ms_path)
