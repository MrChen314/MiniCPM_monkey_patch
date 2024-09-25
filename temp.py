from mindspeed_mm.configs.config import MMConfig
from mindspeed_mm.data import build_mm_dataset, build_mm_dataloader
import torch
from torch_npu.contrib import transfer_to_npu
from megatron.core import mpu
from megatron.core.parallel_state import initialize_model_parallel


torch.distributed.init_process_group(backend='hccl')
initialize_model_parallel(1)
print(mpu.get_data_parallel_group())
import pydevd_pycharm
# pydevd_pycharm.settrace('90.255.87.106', port=1000, stdoutToServer=True, stderrToServer=True)

# args_dict = {'data': '/data/c30061641/MindSpeed-MM/mindspeed_mm/configs/internvl1.5_data_test.json'}

def train_valid_test_datasets_provider():
    args_dict = {'data': '/data/c30061641/MindSpeed-MM/mindspeed_mm/configs/internvl1.5_data_mpu.json'}


    data_args = MMConfig(args_dict)
    dataset = build_mm_dataset(data_args.data.dataset_param)
    # train_dataloader = build_mm_dataloader(dataset, data_args.data.dataloader_param)
    train_dataloader = build_mm_dataloader(
        dataset,
        data_args.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
    )

    for i in train_dataloader:
        # print('input_ids: ', (torch.load("/data/c30061641/UnitTest/internvl/input_ids.pt") - i['input_ids']).abs().max())
        # print('labels: ', (torch.load("/data/c30061641/UnitTest/internvl/labels.pt") - i['labels']).abs().max())
        # print('attention_mask: ', (torch.load("/data/c30061641/UnitTest/internvl/attention_mask.pt")).equal(i['attention_mask']))
        # print('pixel_values: ', (torch.load("/data/c30061641/UnitTest/internvl/pixel_values.pt") - i['pixel_values']).abs().max())
        # print('image_flags: ', (torch.load("/data/c30061641/UnitTest/internvl/image_flags.pt") - i['image_flags']).abs().max())
        print(i['input_ids'].device)
        break

train_valid_test_datasets_provider.is_distributed = True
train_valid_test_datasets_provider()

上面代码打印的i['input_ids'].device是cpu，如何更改代码，使其到npu上
