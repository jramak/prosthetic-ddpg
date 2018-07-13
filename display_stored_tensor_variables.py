import argparse
from tensorflow.python import pywrap_tensorflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore-model-name', type=str, default=None)
    args = parser.parse_args()

    saved_model_dir = 'saved-models/'
    if not args.restore_model_name:
        ValueError('Please provide restore model name')
    else:
        restore_model_path = saved_model_dir + args.restore_model_name

    reader = pywrap_tensorflow.NewCheckpointReader(restore_model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in sorted(var_to_shape_map):
        print("tensor_name: ", key)
