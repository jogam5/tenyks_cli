import subprocess
import time

def automate_cli_input(cli_command, input_values):
    # Start the CLI subprocess
    cli_process = subprocess.Popen(cli_command.split(), stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, universal_newlines=True)

    # Iterate over the expected input prompts
    for prompt, value in input_values.items():
        # Send the input value to the CLI subprocess
        cli_process.stdin.write(f'{value}\n')
        cli_process.stdin.flush()
        time.sleep(1)

    # Read the CLI output
    output = cli_process.communicate()[0]

    # Return the CLI output
    return output

def upload_to_tenyks(dataset_key=None, image_folder_path=None, annotation_file_path=None,
                    prediction_file_path=None, class_path=None, model_name=None, prediction_type='coco'):
    commands = {'tenyks dataset-create': {'Enter dataset name:': f'{dataset_key}'},
                'tenyks dataset-images-upload': {'Enter dataset key:': f'{dataset_key}',
                                                 'Enter image folder path:': f'{image_folder_path}'},
                'tenyks dataset-annotations-upload': {'Enter dataset key:': f'{dataset_key}',
                                                      'Enter annotation file path:': f'{annotation_file_path}'},
                'tenyks dataset-class-names-upload': {'Enter dataset key:': f'{dataset_key}',
                                                        'Enter path to class names:': f'{class_path}'},
                'tenyks model-create': {'Enter model name:': f'{model_name}', 'Enter dataset key:': f'{dataset_key}'},
                'tenyks model-predictions-upload': {'Enter dataset key:': f'{dataset_key}',
                                                    'Enter model key:': f'{model_name}',
                                                    'Enter prediction file path:': f'{prediction_file_path}',
                                                    'Enter prediction file type (coco, vott_csv, yolo, deepstream, classification) [coco]:': f'{prediction_type}'
                                                    }
                }
                    
    for command, input_values in commands.items():
        output = automate_cli_input(command, input_values)
        print(output)
    print('Done')
