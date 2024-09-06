import json
import os
import pty
import re
import signal
import subprocess
from urllib.parse import urlparse

import ipywidgets as widgets
import requests
from IPython.display import display

# #################### GLOBAL PATHS ####################

root = '/notebooks'
webui = root + os.environ.get('WEBUI_DIR', '')
modules_path = webui + '/modules'
oncompleted_path = '/internal/on-completed.sh'

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
builtin_exts_url = main_repo_url + '/res/extensions/builtin-extensions.json'
builtin_exts_forge_url = main_repo_url + '/res/extensions/builtin-extensions-forge.json'
extensions_url = main_repo_url + '/res/extensions/extensions.json'
extensions_forge_url = main_repo_url + '/res/extensions/extensions-forge.json'
upscaler_list_url = main_repo_url + '/res/upscaler.json'

sd15_controlnet_url = main_repo_url + '/res/sd15/sd15-controlnet.json'
sd15_t2i_adapter_url = main_repo_url + '/res/sd15/sd15-t2i-adapter.json'
sd15_anime_models_url = main_repo_url + '/res/sd15/models/sd15-anime-models.json'
sd15_general_models_url = main_repo_url + '/res/sd15/models/sd15-general-models.json'
sd15_realistic_models_url = main_repo_url + '/res/sd15/models/sd15-realistic-models.json'
sd15_woman_models_url = main_repo_url + '/res/sd15/models/sd15-woman-models.json'
sd15_builtin_resources_url = main_repo_url + '/res/sd15/sd15-builtin-resources.json'
sd15_resources_url = main_repo_url + '/res/sd15/sd15-resources.json'

sdxl_controlnet_url = main_repo_url + '/res/sdxl/sdxl-controlnet.json'
sdxl_anime_models_url = main_repo_url + '/res/sdxl/models/sdxl-anime-models.json'
sdxl_general_models_url = main_repo_url + '/res/sdxl/models/sdxl-general-models.json'
sdxl_realistic_models_url = main_repo_url + '/res/sdxl/models/sdxl-realistic-models.json'
sdxl_woman_models_url = main_repo_url + '/res/sdxl/models/sdxl-woman-models.json'
sdxl_builtin_resources_url = main_repo_url + '/res/sdxl/sdxl-builtin-resources.json'
sdxl_resources_url = main_repo_url + '/res/sdxl/sdxl-resources.json'


# #################### VARIABLES ####################

sd = 'sd'
sdxl = 'sdxl'
runpod = 'runpod'
paperspace = 'paperspace'
runpod_port = 3000
boolean = [False, True]
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
def run_process(command: str, hide_output=True, use_shell=False):
    if not use_shell:
        command = command.split()
    subprocess.run(command, capture_output=hide_output, text=True, bufsize=1, shell=use_shell)


# Update ubuntu dependencies
def update_deps():
    print('⏳ Updating dependencies...')
    run_process('apt -y -q update')


def set_oncompleted_permission():
    run_process(f'chmod +x {oncompleted_path}')


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


def close_port(port_number):
    result = subprocess.run(['lsof', '-i', f':{port_number}'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')

    if len(lines) > 1:
        pid = int(lines[1].split()[1])
        os.kill(pid, signal.SIGTERM)


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
def get_resources(url: str):
    session = requests.Session()
    session.cache_disabled = True
    res = session.get(url, headers=request_headers)
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

    prev_line = ''
    parsed_url = urlparse(url)
    url_path = parsed_url.path
    filename = os.path.basename(url_path)
    aria2c = f'stdbuf -oL aria2c --on-download-complete={oncompleted_path} --download-result=hide --console-log-level=error -c -x 16 -s 16 -k 1M -d {path} {url}'

    if overwrite:
        aria2c += ' --allow-overwrite'
    if '.' in filename and filename.split('.')[-1] != '':
        aria2c += f' -o {filename}'

    with subprocess.Popen(aria2c.split(), stdout=subprocess.PIPE, text=True, bufsize=1) as sp:
        for line in sp.stdout:
            if line.startswith('[#'):
                text = 'Download progress {}'.format(line.strip('\n'))
                print('\r' + ' ' * 80 + '\r' + text, end='\r', flush=True)
                prev_line = text
            elif line.startswith('[COMPLETED]'):
                if prev_line != '': print('')
                print(f'Download completed')
            else:
                print(line)


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
    print('⏳ Installing/Updating Stable Diffusion Web UI Forge...')
    webui_version = 'main'
    silent_clone(f'-b {webui_version} https://github.com/lllyasviel/stable-diffusion-webui-forge', root, update=True)
    os.chdir(webui)

    # Download configs
    if not os.path.exists(f'{shared_config_path}/config.json'):
        downloader(f'{main_repo_url}/configs/config.json', shared_config_path)
    if not os.path.exists(f'{shared_config_path}/ui-config.json'):
        downloader(f'{main_repo_url}/configs/ui-config.json', shared_config_path)

    storage_symlinks()


# Install selected extensions
def extensions_selection(_builtin_exts_url: str, _exts_url: str):
    checkboxes = []
    exts_header = widgets.HTML('<h3 style="width: 250px; text-align: center;">Extensions</h3>')
    status_header = widgets.HTML('<h3 style="width: 120px; text-align: center;">Status</h3>')
    homepages_header = widgets.HTML('<h3 style="width: 120px; text-align: center;">Homepages</h3>')
    headers = widgets.HBox([exts_header, status_header, homepages_header])
    output = widgets.Output()
    display(headers)

    for ext in get_resources(_exts_url):
        directory = f"{extensions_path}/{ext['url'].split('/')[-1]}"
        if os.path.exists(directory):
            installed_status = 'installed'
            enabled = True
        else:
            installed_status = 'not installed'
            enabled = True if ext['enabled'] else False

        checkbox = widgets.Checkbox(value=enabled, description=ext['name'], indent=False, layout={'width': '250px'})
        status = widgets.HTML(f'<div style="text-align: center; width: 120px;">{installed_status}</div>')
        homepage = widgets.HTML(f'<div class="jp-RenderedText" style="width: 105px; text-align: center; white-space:nowrap; display: inline-grid;">'
                                f'<pre><a href="{ext["url"]}" target="_blank">GitHub</a></pre></div>')
        item = widgets.HBox([checkbox, status, homepage])
        checkboxes.append((ext, checkbox))
        display(item)

    def on_press(button):
        selected_exts = [(_ext['id'], _ext['name'], _ext['url']) for _ext, _checkbox in checkboxes if _checkbox.value]
        with output:
            output.clear_output()
            try:
                install_builtin_exts(_builtin_exts_url, update_exts.value)
                print("\n⏳ Installing selected extensions...")
                for _id, name, url in selected_exts:
                    if _id == 'bmab':
                        run_process('pip install -q basicsr')
                    print(f'{name}...')
                    silent_clone(url, extensions_path, update=update_exts.value)
                completed_message()
            except KeyboardInterrupt:
                print('\n\n--Install interrupted--')

    update_exts = widgets.Checkbox(value=False, description='Update installed extensions', indent=False, layout={'margin': '2px 0 0 50px'})
    download_button = widgets.Button(description='Install', button_style='success')
    footer = widgets.HBox([download_button, update_exts])
    download_button.on_click(on_press)
    print('')
    display(footer, output)


# Install built-in extensions
def install_builtin_exts(exts_url: str, update=False):
    print("\n⏳ Installing built-in extensions...")
    for ext in get_resources(exts_url):
        print(ext['name'] + '...')
        silent_clone(ext['url'], extensions_path, update)


# Install other extensions
def install_other_exts(extensions: list, update_exts=False):
    if extensions:
        print("⏳ Installing extensions...")
        for ext in extensions:
            name = ext.split('/')[-1]
            print(name + '...')
            silent_clone(ext, extensions_path, update_exts)


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
    replace = re.sub(pattern, 'print(strings.en["RUNNING_LOCALLY"].format(f\'https://{os.environ.get("RUNPOD_POD_ID")}-3001.proxy.runpod.net\'))', content)

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

    args += f' --listen --port {runpod_port}'
    print('Launching Web UI...')
    try:
        pty.spawn(f'python webui.py {args}'.split())
    except KeyboardInterrupt:
        print('--Process terminated--')


def initialization_forge():
    apply_envs1()
    apply_envs2()
    update_deps()
    create_shared_storage()
    set_oncompleted_permission()
    hide_samplers()
    install_forge()
    completed_message()


def models_selection(models_url: str, subdir: str, civitai=''):
    dropdowns = []
    models_header = widgets.HTML('<h3 style="width: 200px;">Models Name</h3>')
    versions_header = widgets.HTML('<h3 style="width: 250px;">Versions</h3>')
    homepages_header = widgets.HTML('<h3 style="width: 100px;">Homepages</h3>')
    headers = widgets.HBox([models_header, versions_header, homepages_header])
    display(headers)

    for model in get_resources(models_url):
        options = ['Select version'] + [variant['version'] for variant in model['variants']]
        dropdown = widgets.Dropdown(options=options, value='Select version', layout=widgets.Layout(width='230px'))
        label = widgets.Label(model['name'], layout=widgets.Layout(width='200px'))
        homepage_links = ' | '.join([f'<a href="{site["url"]}" target="_blank">{site["name"]}</a>' for site in model['homepage']])
        homepage_label = widgets.HTML(
            f'<div class="jp-RenderedText" style="padding-left:20px; white-space:nowrap; display: inline-flex;"><pre>{homepage_links}</pre></div>')
        item = widgets.HBox([label, dropdown, homepage_label])
        dropdowns.append((model['name'], dropdown, model['variants']))
        display(item)
        print('')

    download_button = widgets.Button(description='Download', button_style='success')
    output = widgets.Output()

    def on_press(button):
        selected_versions = []
        for name, menu, variants in dropdowns:
            selected_version = menu.value
            if selected_version != 'Select version':
                for variant in variants:
                    if variant['version'] == selected_version:
                        selected_versions.append((name, selected_version, variant["url"]))

        with output:
            output.clear_output()
            try:
                for name, version, url in selected_versions:
                    print(f'\n* {name} | {version}')
                    downloader(url, f'{models_path}/{subdir}', civitai_token=civitai)
                completed_message()
            except KeyboardInterrupt:
                print('\n\n--Download interrupted--')

    download_button.on_click(on_press)
    display(download_button, output)


# Download ControlNet
def download_controlnet(controlnet: list, url: str, subdir: str):
    controlnet_data = get_resources(url)
    for model in controlnet:
        if controlnet[model]:
            print('\n' + model + '...')
            for url in controlnet_data[model]:
                downloader(url, f'{controlnet_models_path}/{subdir}')
                print('')


def resources_selection(builtin_res_url: str, resources_url: str, subdir: str, civitai=''):
    checkboxes = []
    resources = get_resources(resources_url)

    for resource_type, items in resources.items():
        res_header = widgets.HTML(f'<h3 style="width: 300px; margin-bottom: 0;">{resource_type}</h3>')
        homepages_header = widgets.HTML('<h3 style="width: 100px; margin-bottom: 0;">Homepages</h3>')
        headers = widgets.HBox([res_header, homepages_header])
        display(headers)

        for item in items:
            checkbox = widgets.Checkbox(value=False, description=item['name'], indent=False, layout={'width': '300px'})
            homepage_links = ' | '.join([f'<a href="{site["url"]}" target="_blank">{site["name"]}</a>' for site in item['homepage']])
            homepage_label = widgets.HTML(
                f'<div class="jp-RenderedText" style="padding-left: 0; white-space: nowrap; display: inline-flex;"><pre>{homepage_links}</pre></div>')
            cb_item = widgets.HBox([checkbox, homepage_label])
            checkboxes.append((item, resource_type, checkbox))
            display(cb_item)

    download_button = widgets.Button(description='Download', button_style='success')
    output = widgets.Output()

    def on_press(button):
        selected_res = {
            'LoRA': [],
            'Embeddings': [],
            'Upscaler': [],
            'VAE': []
        }

        for _res, _resource_type, _checkbox in checkboxes:
            if _checkbox.value:
                if _resource_type == 'LoRA':
                    selected_res['LoRA'].append((_res['name'], _res['url']))
                elif _resource_type == 'Embeddings':
                    selected_res['Embeddings'].append((_res['name'], _res['url']))
                elif _resource_type == 'Upscaler':
                    selected_res['Upscaler'].append((_res['name'], _res['url']))
                elif _resource_type == 'VAE':
                    selected_res['VAE'].append((_res['name'], _res['url']))

        with output:
            output.clear_output()
            try:
                download_builtin_resources(builtin_res_url, subdir)
                for _type in selected_res:
                    if _type == 'LoRA' and selected_res[_type]:
                        download_selected_res(selected_res[_type], _type, f'{lora_path}/{subdir}', civitai)
                    elif _type == 'Embeddings' and selected_res[_type]:
                        print(f'\n⏳ Downloading selected {_type}...')
                        for name, url in selected_res['Embeddings']:
                            print(name + '...')
                            silent_clone(url, f'{embeddings_path}/{subdir}', True)
                    elif _type == 'Upscaler' and selected_res[_type]:
                        download_selected_res(selected_res[_type], _type, upscaler_path, civitai)
                    elif _type == 'VAE' and selected_res[_type]:
                        download_selected_res(selected_res[_type], _type, f'{vae_path}/{subdir}', civitai)
                completed_message()
            except KeyboardInterrupt:
                print('\n\n--Download interrupted--')

    download_button.on_click(on_press)
    print('')
    display(download_button, output)


def download_selected_res(res_list: list, _type: str, path: str, civitai=''):
    print(f'\n\n⏳ Downloading selected {_type}...')
    for name, urls in res_list:
        print(f'\n* {name}...')
        for url in urls:
            downloader(url, path, civitai_token=civitai)
        print('')


# Download built-in resources
def download_builtin_resources(resources_url: str, subdir: str):
    parentdir = ''
    resources = get_resources(resources_url)
    for resource_type, items in resources.items():
        match resource_type:
            case 'Embeddings':
                parentdir = embeddings_path
            case 'LoRA':
                parentdir = lora_path
            case 'Upscaler':
                parentdir = upscaler_path
            case 'VAE':
                parentdir = vae_path

        print(f'\n\n⏳ Downloading built-in {resource_type}...')
        for item in items:
            print(f"\n* {item['name']}...")
            if resource_type == 'Embeddings':
                silent_clone(item['url'], f'{parentdir}/{subdir}', True)
            elif resource_type == 'Upscaler':
                downloader(item['url'], parentdir)
            else:
                downloader(item['url'], f'{parentdir}/{subdir}')


# Other resources
def other_resources(resource_list: list, resource_path: str, civitai=''):
    for resource in resource_list:
        print(f'\n{resource}')
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