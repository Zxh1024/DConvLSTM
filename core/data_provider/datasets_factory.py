from core.data_provider import mnist

datasets_map = {
    'mnist': mnist,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, test_data_paths, batch_size,
                  img_width, seq_length, injection_action, is_training=True):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    test_data_list = test_data_paths.split(',')
    if dataset_name == 'mnist':
        # test输入数据参数
        test_input_param = {'paths': test_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)

        if is_training:

            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=False)

            valid_input_param = {'paths': valid_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + 'valid iterator'}
            valid_input_handle = datasets_map[dataset_name].InputHandle(valid_input_param)
            valid_input_handle.begin(do_shuffle=False)
            return train_input_handle, valid_input_handle, test_input_handle
        else:
            return test_input_handle
