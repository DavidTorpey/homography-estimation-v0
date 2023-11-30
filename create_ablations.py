import os
from itertools import combinations, chain
from pathlib import Path

MODEL = 'byol'

f = f'config/msl/resnet50/{MODEL}-affine/tiny_imagenet/affine-components-ablations'

template_config_path = f'config/msl/resnet50/{MODEL}-affine/tiny_imagenet/config.yaml.template'
template_job_path = f'config/msl/resnet50/{MODEL}-affine/tiny_imagenet/job.sh.template'

s = ['shear', 'rotation', 'translation', 'scale']


def process_le_pair(dataset, model_p):
    config_t_path = f'config/msl/resnet50/{MODEL}-affine/tiny_imagenet/config-le-{dataset}.yaml.template'
    job_t_path = f'config/msl/resnet50/{MODEL}-affine/tiny_imagenet/job-le-{dataset}.sh.template'
    le_config_content = open(config_t_path).read()
    le_config_content = le_config_content.replace('<COMBO>', ee)
    le_config_content = le_config_content.replace('<PROJ_SIZE>', str(proj_size))
    new_le_config_path = os.path.join(p, f'config-le-{dataset}.yaml')
    with open(new_le_config_path, 'w') as file:
        file.write(le_config_content)

    le_job_content = open(job_t_path).read()
    le_job_content = le_job_content.replace('<JOB_FOLDER>', os.path.dirname(new_config_path))
    le_job_content = le_job_content.replace('<CONFIG_RELATIVE_PATH>', new_config_path)
    le_job_content = le_job_content.replace('<MODEL_PATH>', model_p)
    new_le_job_path = os.path.join(p, f'job-le-{dataset}.sh')
    with open(new_le_job_path, 'w') as file:
        file.write(le_job_content)


def powerset(iterable):
    s = list(iterable)
    a = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    a = list(filter(lambda x: len(x) > 0, a))
    return list(map(lambda x: sorted(x), a))


for e in powerset(s):
    ee = '-'.join(e)
    p = os.path.join(f, ee)
    Path(p).mkdir(parents=True, exist_ok=True)

    rotation = 'rotation' in e
    translation = 'translation' in e
    scale = 'scale' in e
    shear = 'shear' in e
    proj_size = 0
    if rotation:
        proj_size += 1
    if scale:
        proj_size += 1
    if shear:
        proj_size += 2
    if translation:
        proj_size += 2
    config_content = open(template_config_path).read()
    config_content = config_content.replace('<ROTATION>', str(rotation))
    config_content = config_content.replace('<TRANSLATION>', str(translation))
    config_content = config_content.replace('<SCALE>', str(scale))
    config_content = config_content.replace('<SHEAR>', str(shear))
    config_content = config_content.replace('<COMBO>', ee)
    config_content = config_content.replace('<PROJ_SIZE>', str(proj_size))
    new_config_path = os.path.join(p, 'config.yaml')
    with open(new_config_path, 'w') as file:
        file.write(config_content)

    job_content = open(template_job_path).read()
    job_content = job_content.replace('<JOB_FOLDER>', os.path.dirname(new_config_path))
    job_content = job_content.replace('<CONFIG_RELATIVE_PATH>', new_config_path)
    new_job_path = os.path.join(p, 'job.sh')
    with open(new_job_path, 'w') as file:
        file.write(job_content)

    model_path = f'/home-mscluster/dtorpey/code/homography-estimation-v0/results/resnet50/{MODEL}-affine/tiny_imagenet/affine-components-ablations/{ee}/90_model_tiny_imagenet.pth'
    process_le_pair('cifar10', model_path)
    process_le_pair('cifar100', model_path)
    process_le_pair('food101', model_path)
