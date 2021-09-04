from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset


if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/sanskrit_letter_images'))
    print('dataset done')

    experiment = Experiment(workspace=ws, name='day1-experiment-data')
    print('exp done')

    config = ScriptRunConfig(
        source_directory='./src',
        script='train.py',
        compute_target='ocr-sanskrit-ds',
        arguments=['--data_path', dataset.as_named_input('input').as_mount()])
    print('config set')

    # use curated pytorch environment
    env = ws.environments['AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu']
    config.run_config.environment = env
    print('env set')

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute instance. Click link below")
    print("")
    print(aml_url)
