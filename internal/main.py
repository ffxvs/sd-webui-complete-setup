import json
import os
import re
import subprocess
from urllib.parse import urlparse

import ipywidgets as widgets
import requests
from IPython.display import display, Markdown

# #################### GLOBAL PATHS ####################

root = '/notebooks'
webui = root + os.environ.get('WEBUI_DIR', '')
modules_path = webui + '/modules'

outputs_path = webui + '/outputs'
extensions_path = webui + '/extensions'
controlnet_models_path = webui + '/models/ControlNet'
embeddings_path = webui + '/embeddings'
models_path = webui + '/models/Stable-diffusion'
lora_path = webui + '/models/Lora'
upscaler_path = webui + '/models/ESRGAN'
vae_path = webui + '/models/VAE'

shared_storage = root + '/shared-storage'
shared_models_path = shared_storage + '/models'
shared_embeddings_path = shared_storage + '/embeddings'
shared_lora_path = shared_storage + '/lora'
shared_upscaler_path = shared_storage + '/esrgan'
shared_vae_path = shared_storage + '/vae'
shared_controlnet_models_path = shared_storage + '/controlNet'
shared_outputs_path = shared_storage + '/outputs'
shared_config_path = shared_storage + '/config'

# Resource URLs
main_repo_url = 'https://raw.githubusercontent.com/ffxvs/sd-webui-complete-setup/dev'
builtin_exts_url = main_repo_url + '/res/builtin-extensions.json'
extensions_url = main_repo_url + '/res/extensions.json'
upscaler_list_url = main_repo_url + '/res/upscaler.json'

sd15_controlnet_url = main_repo_url + '/res/sd15/sd15-controlnet.json'
sd15_t2i_adapter_url = main_repo_url + '/res/sd15/sd15-t2i-adapter.json'
sd15_anime_models_url = main_repo_url + '/res/sd15/models/sd15-anime-models.json'
sd15_general_models_url = main_repo_url + '/res/sd15/models/sd15-general-models.json'
sd15_realistic_models_url = main_repo_url + '/res/sd15/models/sd15-realistic-models.json'
sd15_woman_models_url = main_repo_url + '/res/sd15/models/sd15-woman-models.json'
sd15_builtin_resources_url = main_repo_url + '/res/sd15/sd15-builtin-resources.json'
sd15_lora_list_url = main_repo_url + '/res/sd15/sd15-lora.json'
sd15_embedding_list_url = main_repo_url + '/res/sd15/sd15-embeddings.json'
sd15_vae_list_url = main_repo_url + '/res/sd15/sd15-vae.json'

sdxl_controlnet_url = main_repo_url + '/res/sdxl/sdxl-controlnet.json'
sdxl_anime_models_url = main_repo_url + '/res/sdxl/models/sdxl-anime-models.json'
sdxl_general_models_url = main_repo_url + '/res/sdxl/models/sdxl-general-models.json'
sdxl_realistic_models_url = main_repo_url + '/res/sdxl/models/sdxl-realistic-models.json'
sdxl_woman_models_url = main_repo_url + '/res/sdxl/models/sdxl-woman-models.json'
sdxl_builtin_resources_url = main_repo_url + '/res/sdxl/sdxl-builtin-resources.json'
sdxl_lora_list_url = main_repo_url + '/res/sdxl/sdxl-lora.json'

# #################### VARIABLES ####################

sd = 'sd'
sdxl = 'sdxl'
boolean = [False, True]
exclude_exts_forge = ['controlNet']
request_headers = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
}


# #################### FUNCTIONS ####################

# Apply environment variables 1
def apply_envs1():
    os.environ['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libtcmalloc.so.4'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'


# Apply environment variables 2
def apply_envs2():
    os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    os.environ['FORCE_CUDA'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.9,max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['CUDA_AUTO_BOOST'] = '1'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    os.environ['CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT'] = '0'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['SAFETENSORS_FAST_GPU'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '16'


# Run external program
def run_process(command: str, hide_output=True):
    # command = command.split()
    # subprocess.run(command, capture_output=hide_output, text=True, check=True)
    subprocess.run(command, shell=True)


# Update ubuntu dependencies
def update_deps():
    print('⏳ Updating dependencies...')
    run_process('apt -y -q update')


# Create symlink
def symlink(source: str, destination: str):
    if os.path.exists(source) and not os.path.islink(destination):
        run_process(f'rm -r -f {destination}')
        run_process(f'ln -s {source} {destination}')


# Complete message
def completed_message():
    completed = widgets.Button(description='Completed', button_style='success', icon='check')
    print('\n')
    display(completed)


# Hide samplers
def hide_samplers():
    config_file = f'{shared_config_path}/config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
        samplers_to_hide = ["DPM fast", "PLMS"]
        samplers_to_show = ["DPM++ SDE", "DPM++ 2M", "LMS", "DPM2", "DPM++ 2M SDE", "DPM++ 2M SDE Heun"]
        for sampler in samplers_to_hide:
            if sampler not in config['hide_samplers']:
                config['hide_samplers'].append(sampler)
        config['hide_samplers'] = [sampler for sampler in config['hide_samplers'] if sampler not in samplers_to_show]
        with open(config_file, 'w') as file:
            json.dump(config, file, indent=4)


# Create shared storage
def create_shared_storage():
    print('⏳ Creating shared storage directory...')
    os.chdir(root)
    shared_storage_folders = [
        shared_storage,
        f"{shared_models_path}/sd",
        f"{shared_models_path}/sdxl",
        f"{shared_embeddings_path}/sd",
        f"{shared_embeddings_path}/sdxl",
        f"{shared_lora_path}/sd",
        f"{shared_lora_path}/sdxl",
        f"{shared_vae_path}/sd",
        f"{shared_vae_path}/sdxl",
        f"{shared_controlnet_models_path}/sd",
        f"{shared_controlnet_models_path}/sdxl",
        shared_upscaler_path,
        shared_outputs_path,
        shared_config_path
    ]

    for folder in shared_storage_folders:
        os.makedirs(folder, exist_ok=True)


# Create shared storage symlinks
def storage_symlinks():
    symlink(shared_models_path, models_path)
    symlink(shared_lora_path, lora_path)
    symlink(shared_embeddings_path, embeddings_path)
    symlink(shared_upscaler_path, upscaler_path)
    symlink(shared_vae_path, vae_path)
    symlink(shared_controlnet_models_path, controlnet_models_path)
    symlink(shared_outputs_path, outputs_path)
    symlink(f'{shared_config_path}/config.json', f'{webui}/config.json')
    symlink(f'{shared_config_path}/ui-config.json', f'{webui}/ui-config.json')


# Get resources list json
def get_resource(url: str):
    res = requests.get(url, headers=request_headers)
    if res.status_code == 200:
        return res.json()
    else:
        return False


# Download files using aria2c
def downloader(url: str, path: str, overwrite=False, civitai_token=''):
    if url.startswith('https://civitai.com/api/download/') and civitai_token:
        if '?' in url:
            url += f'&token={civitai_token}'
        else:
            url += f'?token={civitai_token}'

    parsed_url = urlparse(url)
    url_path = parsed_url.path
    filename = os.path.basename(url_path)
    formatted_url = f'"{url}"'
    aria2c = f'aria2c --download-result=hide --console-log-level=error -c -x 16 -s 16 -k 1M -d {path} {formatted_url}'

    if overwrite:
        aria2c += ' --allow-overwrite'
    if '.' in filename and filename.split('.')[-1] != '':
        aria2c += f' -o {filename}'
    run_process(aria2c, hide_output=False)


# Git clone repo
def silent_clone(command: str, path: str, update=False, overwrite=False):
    directory = f'{path}/{command.split("/")[-1]}'
    git_clone = f'git clone -q --depth 20 {command} {directory}'
    if os.path.exists(directory):
        if update:
            os.chdir(f'{directory}')
            run_process('git pull -q')
        elif overwrite:
            run_process(f'rm -r {directory}')
            run_process(git_clone)
    else:
        run_process(git_clone)


# Download files using WGet
def silent_get(command: str):
    run_process(f'wget -nv {command}')


# Install Web UI Forge
def install_forge():
    os.chdir(root)
    print('⏳ Installing Stable Diffusion Web UI Forge...')
    webui_version = 'main'
    silent_clone(f'-b {webui_version} https://github.com/lllyasviel/stable-diffusion-webui-forge', root, update=True)
    os.chdir(webui)

    # Download configs
    if not os.path.exists(f'{shared_config_path}/config.json'):
        downloader(f'{main_repo_url}/configs/config.json', shared_config_path)
    if not os.path.exists(f'{shared_config_path}/ui-config.json'):
        downloader(f'{main_repo_url}/configs/ui-config.json', shared_config_path)

    storage_symlinks()


# Install built-in extensions
def download_default_exts(update_exts=False):
    print("⏳ Installing built-in extensions...")
    for ext in get_resource(builtin_exts_url)['extensions']:
        if not ext['id'] in exclude_exts_forge:
            print(ext['name'] + '...')
            silent_clone(ext['url'], extensions_path, update_exts)


# Install selected extensions
def download_exts(update_exts=False):
    print("\n⏳ Installing selected extensions...")
    for ext in get_resource(extensions_url)['extensions']:
        try:
            if eval(ext['id']):
                if ext['id'] == 'bmab':
                    run_process('pip install -q basicsr')
                print(ext['name'] + '...')
                silent_clone(ext['url'], extensions_path, update_exts)
        except:
            pass


# Install other extensions
def download_other_exts(extensions: list, update_exts=False):
    if extensions:
        print("⏳ Installing extensions...")
        for ext in extensions:
            name = ext.split('/')[-1]
            print(name + '...')
            silent_clone(ext, extensions_path, update_exts)


# Download ControlNet
def download_controlnet(controlnet: list, url: str, subdir: str):
    controlnet_data = get_resource(url)
    for model in controlnet:
        if controlnet[model]:
            print('\n' + model + '...')
            for url in controlnet_data[model]:
                downloader(url, f'{controlnet_models_path}/{subdir}')
                print('')


# Model mapper
def model_mapper(name: str, version: str, url: str):
    return {
        'name': name,
        'version': version,
        'url': url
    }


class ModelMapper:
    def __init__(self, name: str, version: str, url: str):
        self.name = name
        self.version = version
        self.url = url


# Selected models
def selected_models(models_url: str):
    model_list = []
    for model in get_resource(models_url)['models']:
        is_selected = eval(model['id'])
        if is_selected != 'Select version...':
            for variant in model['variants']:
                if variant['version'] == is_selected:
                    model_list.append(ModelMapper(model['name'], variant['version'], variant['url']))
    return model_list


# Download selected models
def download_models(models_url: str, subdir: str, civitai=''):
    print('⏳ Downloading selected models...')
    for model in selected_models(models_url):
        print(f"\n* {model.name} | {model.version}")
        downloader(model.url, f'{models_path}/{subdir}', civitai_token=civitai)


# Download built-in resources
def download_builtin_resources(resources_url: str, subdir: str):
    section = ''
    parentdir = ''
    resources = get_resource(resources_url)
    for resource_type, items in resources.items():
        match resource_type:
            case 'embeddings':
                section = 'Embeddings'
                parentdir = embeddings_path
            case 'lora':
                section = 'LoRA'
                parentdir = lora_path
            case 'upscaler':
                section = 'Upscaler'
                parentdir = upscaler_path
            case 'vae':
                section = 'VAE'
                parentdir = vae_path

        print(f'⏳ Downloading built-in {section}...')
        for item in items:
            print(item['name'] + '...')
            if resource_type == 'embeddings':
                silent_clone(item['url'], f'{parentdir}/{subdir}', True)
            else:
                downloader(item['url'], f'{parentdir}/{subdir}')


# Selected resources
def selected_resources(resource_list: list, resource_path: str):
    for resource in resource_list:
        if eval(resource['id']):
            print(f"\n {resource['name']}...")
            for url in resource['url']:
                downloader(url, resource_path)
                print('')


# Download selected LoRA
def download_lora(lora_url: str, subdir: str):
    print('\n\n⏳ Downloading selected LoRA...')
    selected_resources(get_resource(lora_url)['lora'], f'{lora_path}/{subdir}')


# Download selected embeddings
def download_embeddings(embeddings_url: str, subdir: str):
    print('\n\n⏳ Downloading selected embeddings...')
    for embedding in get_resource(embeddings_url)['embeddings']:
        if eval(embedding['id']):
            print(embedding['name'] + '...')
            silent_clone(embedding['url'], f'{embeddings_path}/{subdir}', True)


# Download selected upscaler
def download_upscaler():
    print('\n\n⏳ Downloading selected upscaler...')
    selected_resources(get_resource(upscaler_list_url)['upscaler'], upscaler_path)


# Download selected VAE
def download_vae(vae_url: str, subdir: str):
    print('\n\n⏳ Downloading selected LoRA...')
    selected_resources(get_resource(vae_url)['vae'], f'{vae_path}/{subdir}')


# Other resources
def other_resources(resource_list: list, resource_path: str, civitai=''):
    for resource in resource_list:
        print('\n' + resource)
        downloader(resource, resource_path, civitai_token=civitai)


# Download other resources
def download_other_resources(subdir: str, lora: list, embeddings: list, upscaler: list, vae: list, civitai=''):
    if lora:
        print('\n\n⏳ Downloading LoRA...')
        other_resources(lora, f'{lora_path}/{subdir}', civitai)
    if embeddings:
        print('\n\n⏳ Downloading embeddings...')
        other_resources(embeddings, f'{embeddings_path}/{subdir}', civitai)
    if upscaler:
        print('\n\n⏳ Downloading upscaler...')
        other_resources(upscaler, upscaler_path, civitai)
    if vae:
        print('\n\n⏳ Downloading VAE...')
        other_resources(vae, f'{vae_path}/{subdir}', civitai)


# Launch Web UI
def launch_webui(dark_theme: bool, username: str, password: str, ngrok_token: str, ngrok_domain: str, cors: str):
    print('⏳ Preparing...')
    print('It will take a little longer...')
    args = '--disable-console-progressbars --disable-safe-unpickle --enable-insecure-extension-access --no-download-sd-model --no-hashing --api --xformers'
    blocks_path = '/usr/local/lib/python3.10/dist-packages/gradio/blocks.py'
    os.chdir(webui)

    run_process(f'python launch.py {args} --exit')
    run_process('pip install -q pillow==9.5.0')

    with open(blocks_path, 'r') as file:
        content = file.read()

    pattern = re.compile(r'print\(\s*strings\.en\["RUNNING_LOCALLY_SEPARATED"]\.format\(\s*self\.protocol, self\.server_name, self\.server_port\s*\)\s*\)')
    replace = re.sub(pattern, 'print(strings.en["RUNNING_LOCALLY"].format(f\'https://{os.environ.get("RUNPOD_POD_ID")}-3001.proxy.runpod.net\'))',content)

    with open(blocks_path, 'w') as file:
        file.write(replace)

    if dark_theme:
        args += ' --theme dark'
    if username and password:
        args += f' --gradio-auth {username}:{password}'
    if ngrok_token:
        args += f' --ngrok {ngrok_token}'
        if ngrok_domain:
            ngrok_options = '\'{"hostname":"' + ngrok_domain + '"}\''
            args += f' --ngrok-options {ngrok_options}'
    if cors:
        args += f' --cors-allow-origins {cors}'

    args += ' --listen --port 3000'
    print('Launching Web UI...')
    run_process(f'python webui.py {args}', False)


def initialization_forge():
    apply_envs1()
    apply_envs2()
    hide_samplers()
    update_deps()
    create_shared_storage()
    install_forge()
    completed_message()


def models_selection(models_url: str, subdir: str, civitai=''):
    dropdowns = []
    for model in get_resource(models_url):
        options = ['Select version'] + [variant['version'] for variant in model['variants']]
        dropdown = widgets.Dropdown(options=options, value='Select version', layout=widgets.Layout(width='200px'))
        label = widgets.Label(model['name'], layout=widgets.Layout(width='150px'))
        homepage_links = " | ".join([f"<a href='{site['url']}' target='_blank'>{site['name']}</a>" for site in model['homepage']])
        homepage_label = widgets.HTML(f'<div class="jp-RenderedText" style="white-space:nowrap; display: inline-flex;"><pre>{homepage_links}</pre></div>')
        items = widgets.HBox([label, dropdown, homepage_label])
        dropdowns.append((model["name"], dropdown, model["variants"], items))

    submit_button = widgets.Button(description='Download', button_style='success')
    output = widgets.Output()

    def on_press(button):
        selected_versions = []
        for name, menu, variants, _ in dropdowns:
            selected_version = menu.value
            if selected_version != 'Select version':
                for variant in variants:
                    if variant['version'] == selected_version:
                        selected_versions.append((name, selected_version, variant["url"]))

        with output:
            output.clear_output()
            for name, version, url in selected_versions:
                print(f'\n* {name} | {version}')
                downloader(url, f'{models_path}/{subdir}', civitai_token=civitai)

    submit_button.on_click(on_press)
    for _, _, _, items in dropdowns:
        display(items)
        print('')

    display(submit_button, output)
