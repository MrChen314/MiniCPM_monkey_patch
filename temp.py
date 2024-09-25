Traceback (most recent call last):
  File "/data/c30061641/MindSpeed-MM/internvl_dataloader_test.py", line 34, in <module>
    train_valid_test_datasets_provider()
  File "/data/c30061641/MindSpeed-MM/internvl_dataloader_test.py", line 21, in train_valid_test_datasets_provider
    process_group=mpu.get_data_parallel_group(),
  File "/data/c30061641/MindSpeed-MM/megatron/core/parallel_state.py", line 588, in get_data_parallel_group
    assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
AssertionError: data parallel group is not initialized
