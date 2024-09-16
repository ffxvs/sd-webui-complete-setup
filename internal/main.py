import json
import os
import pty
import signal
import subprocess
from urllib.parse import urlparse

import ipywidgets as widgets
import requests
from IPython.display import display

# #################### GLOBAL PATHS ####################

root = '/notebooks'
webui_dir = os.environ.get('WEBUI_DIR', '')
webui_path = root + webui_dir
oncompleted_path = '/internal/on-completed.sh'
outputs_path = webui_path + '/outputs'
extensions_path = webui_path + '/extensions'
modules_path = webui_path + '/modules'

models_path = webui_path + '/models/Stable-diffusion'
embeddings_path = webui_path + '/embeddings'
lora_path = webui_path + '/models/Lora'
upscaler_path = webui_path + '/models/ESRGAN'
vae_path = webui_path + '/models/VAE'
controlnet_models_path = webui_path + '/models/ControlNet'
text_encoder_path = webui_path + '/models/text_encoder'

if webui_dir == '/stable-diffusion-webui-forge':
    preprocessor_path = webui_path + '/models/ControlNetPreprocessor'
else:
    preprocessor_path = extensions_path + '/sd-webui-controlnet/annotator/downloads'

shared_storage = root + '/shared-storage'
shared_models_path = shared_storage + '/models'
shared_embeddings_path = shared_storage + '/embeddings'
shared_lora_path = shared_storage + '/lora'
shared_upscaler_path = shared_storage + '/esrgan'
shared_vae_path = shared_storage + '/vae'
shared_controlnet_models_path = shared_storage + '/controlNet'
shared_text_encoder_path = shared_storage + '/text_encoder'
shared_outputs_path = shared_storage + '/outputs'
shared_config_path = shared_storage + '/config'

temp_storage = '/temp-storage'
temp_models_path = temp_storage + '/models'
temp_lora_path = temp_storage + '/lora'
temp_controlnet_models_path = temp_storage + '/controlNet'
temp_preprocessor_path = temp_controlnet_models_path + '/preprocessor'
temp_text_encoder_path = temp_storage + '/text_encoder'

# #################### RESOURCE URLs ####################

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

flux_general_models_url = main_repo_url + '/res/flux/models/flux-general-models.json'
flux_resources_url = main_repo_url + '/res/flux/flux-resources.json'


# #################### CLASSES ####################

class BaseModel:
    def __init__(self):
        self.SD15 = 'sd'
        self.SDXL = 'sdxl'
        self.FLUX = 'flux'


class ResourceType:
    def __init__(self):
        self.embedding = 'Embedding'
        self.lora = 'LoRA'
        self.upscaler = 'Upscaler'
        self.vae = 'VAE'
        self.text_encoder = 'TextEncoder'


class Platform:
    def __init__(self):
        self.runpod = 'runpod'
        self.paperspace = 'paperspace'


class Port:
    def __init__(self):
        self.runpod = 3000
        self.paperspace = 6006


class UI:
    def __init__(self):
        self.auto1111 = 'auto1111'
        self.forge = 'forge'


class WebUI:
    def __init__(self, _ui: str, _platform: str, dark_theme: bool, username: str, password: str, cors: str, ngrok_token='', ngrok_domain=''):
        self.ui = _ui
        self.platform = _platform
        self.dark_theme = dark_theme
        self.username = username
        self.password = password
        self.ngrok_token = ngrok_token
        self.ngrok_domain = ngrok_domain
        self.cors = cors


class OtherRes:
    def __init__(self, lora: list, embedding: list, upscaler: list, vae: list, text_encoder: list):
        self.lora = lora
        self.embedding = embedding
        self.upscaler = upscaler
        self.vae = vae
        self.text_encoder = text_encoder


# #################### VARIABLES ####################

base = BaseModel()
res_type = ResourceType()
platform = Platform()
port = Port()
ui = UI()

other = OtherRes([], [], [], [], [])
webUI = WebUI('', '', True, '', '', '')

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
def run_process(command: str, use_shell=False):
    if not use_shell:
        command = command.split()
    return subprocess.run(command, shell=use_shell, capture_output=True, text=True, bufsize=1)


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


# Remove symlink
def unlink(target: str):
    if os.path.exists(target) and os.path.islink(target):
        run_process(f'unlink {target}')


# Complete message
def completed_message():
    completed = widgets.Button(description='Completed', button_style='success', icon='check')
    print('\n')
    display(completed)


def close_port(port_number: int):
    result = run_process(f'lsof -i :{port_number}').stdout.strip().split('\n')
    if len(result) > 1:
        pid = int(result[1].split()[1])
        os.kill(pid, signal.SIGTERM)


# Hide samplers
def remove_old_config():
    config_file = f'{shared_config_path}/config.json'
    if os.path.exists(config_file):
        ui_tab_order = ["txt2img", "Txt2img", "img2img", "Img2img", "Extras",
                        "PNG Info", "Checkpoint Merger", "Train", "Cleaner",
                        "Mini Paint", "Photopea", "Infinite image browsing"]
        with open(config_file, 'r') as file:
            config = json.load(file)
        if sorted(config['ui_tab_order']) != sorted(ui_tab_order):
            os.remove(config_file)


def remove_old_forge():
    forge_path = f'{root}/stable-diffusion-webui-forge'
    if os.path.exists(forge_path):
        last_commit = run_process(f'git -C {forge_path} log -1 --oneline').stdout.strip()
        if last_commit.startswith('bfee03d'):
            run_process(f'rm -r -f {forge_path}')


def temp_storage_symlink(option, source, destination):
    if option:
        symlink(source, destination)
    else:
        unlink(destination)
        run_process(f'rm -r -f {source}/*', use_shell=True)
        run_process(f'mkdir -p {destination}')


# Create shared storage
def create_shared_storage():
    print('⏳ Creating shared storage directory...')
    os.chdir(root)
    shared_storage_folders = [
        shared_storage,
        f"{shared_models_path}/sd",
        f"{shared_models_path}/sdxl",
        f"{shared_models_path}/flux",
        f"{shared_embeddings_path}/sd",
        f"{shared_embeddings_path}/sdxl",
        f"{shared_lora_path}/sd",
        f"{shared_lora_path}/sdxl",
        f"{shared_lora_path}/flux",
        f"{shared_vae_path}/sd",
        f"{shared_vae_path}/sdxl",
        f"{shared_vae_path}/flux",
        f"{shared_controlnet_models_path}/sd",
        f"{shared_controlnet_models_path}/sdxl",
        f"{shared_controlnet_models_path}/flux",
        f"{shared_text_encoder_path}/flux",
        shared_upscaler_path,
        shared_outputs_path,
        shared_config_path
    ]

    for folder in shared_storage_folders:
        os.makedirs(folder, exist_ok=True)


# Create symlinks from temporary storage to shared storage
def temp_storage_settings():
    checkboxes = []
    ts_list = [
        ('SD v1.5 Models', f'{temp_models_path}/sd', f'{shared_models_path}/sd'),
        ('SD v1.5 LoRA', f'{temp_lora_path}/sd', f'{shared_lora_path}/sd'),
        ('SD v1.5 ControlNet', f'{temp_controlnet_models_path}/sd', f'{shared_controlnet_models_path}/sd'),
        ('SDXL Models', f'{temp_models_path}/sdxl', f'{shared_models_path}/sdxl'),
        ('SDXL LoRA', f'{temp_lora_path}/sdxl', f'{shared_lora_path}/sdxl'),
        ('SDXL ControlNet', f'{temp_controlnet_models_path}/sdxl', f'{shared_controlnet_models_path}/sdxl'),
        ('FLUX Models', f'{temp_models_path}/flux', f'{shared_models_path}/flux'),
        ('FLUX LoRA', f'{temp_lora_path}/flux', f'{shared_lora_path}/flux'),
        ('FLUX ControlNet', f'{temp_controlnet_models_path}/flux', f'{shared_controlnet_models_path}/flux'),
        ('FLUX Text Encoder', f'{temp_text_encoder_path}/flux', f'{shared_text_encoder_path}/flux'),
        ('ControlNet Preprocessor', temp_preprocessor_path, preprocessor_path)
    ]

    ts_header = widgets.HTML('<h3 style="width: 250px;">Options</h3>')
    status_header = widgets.HTML('<h3 style="width: 100px; text-align: center;">Status</h3>')
    headers = widgets.HBox([ts_header, status_header])
    output = widgets.Output()
    display(headers)

    def set_status(state): return '<div style="text-align: center; width: 100px;{0}</div>'.format(' color: lawngreen;">ON' if state else '">OFF')

    for opt, source, destination in ts_list:
        is_link = os.path.islink(destination)
        checkbox = widgets.Checkbox(value=is_link, description=opt, indent=False, layout={'width': '250px'})
        status = widgets.HTML(set_status(is_link))
        item = widgets.HBox([checkbox, status])
        checkboxes.append((checkbox, status, source, destination))
        display(item)

    def on_press(button):
        with output:
            output.clear_output()
            for cb, sts, src, des in checkboxes:
                temp_storage_symlink(cb.value, src, des)
                sts.value = set_status(os.path.islink(des))
            shared_storage_symlinks()

    apply_button = widgets.Button(description='Apply', button_style='success')
    apply_button.on_click(on_press)
    print('')
    display(apply_button)
    print('\n* Selected   : ON')
    print('* Not Selected : OFF\n')
    display(output)


# Create symlinks from shared storage to webui
def shared_storage_symlinks():
    symlink(shared_models_path, models_path)
    symlink(shared_lora_path, lora_path)
    symlink(shared_embeddings_path, embeddings_path)
    symlink(shared_upscaler_path, upscaler_path)
    symlink(shared_vae_path, vae_path)
    symlink(shared_controlnet_models_path, controlnet_models_path)
    symlink(shared_outputs_path, outputs_path)
    symlink(shared_text_encoder_path, text_encoder_path)
    symlink(f'{shared_config_path}/config.json', f'{webui_path}/config.json')
    symlink(f'{shared_config_path}/ui-config.json', f'{webui_path}/ui-config.json')


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


def install_auto1111():
    os.chdir(root)
    print('⏳ Installing/Updating Stable Diffusion Web UI...')
    webui_version = 'v1.10.1'
    silent_clone(f'-b {webui_version} https://github.com/AUTOMATIC1111/stable-diffusion-webui', root)
    os.chdir(webui_path)
    run_process('git remote set-branches origin master')
    run_process(f'git fetch origin tag {webui_version} -q --depth=5')
    run_process(f'git checkout -q -f {webui_version}')

    download_configs()
    shared_storage_symlinks()

    downloader(f'https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/{webui_version}/modules/extras.py', modules_path, True)
    downloader(f'https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/{webui_version}/modules/sd_models.py', modules_path, True)

    # From ThelastBen
    run_process(f'sed -i \'s@shared.opts.data\["sd_model_checkpoint"] = checkpoint_info.title@shared.opts.data\["sd_model_checkpoint"] = checkpoint_info.title;model.half()@\' {modules_path}/sd_models.py', use_shell=True)
    run_process(f"sed -i \"s@map_location='cpu'@map_location='cuda'@\" {modules_path}/extras.py", use_shell=True)


# Install Web UI Forge
def install_forge():
    os.chdir(root)
    print('⏳ Installing/Updating Stable Diffusion Web UI Forge...')
    webui_version = 'main'
    silent_clone(f'-b {webui_version} https://github.com/lllyasviel/stable-diffusion-webui-forge', root, update=True)
    os.chdir(webui_path)
    download_configs()
    shared_storage_symlinks()


def download_configs():
    if not os.path.exists(f'{shared_config_path}/config.json'):
        print('Download config.json')
        downloader(f'{main_repo_url}/configs/config.json', shared_config_path)
    if not os.path.exists(f'{shared_config_path}/ui-config.json'):
        print('Download ui-config.json')
        downloader(f'{main_repo_url}/configs/ui-config.json', shared_config_path)


# Install selected extensions
def extensions_selection(_builtin_exts_url: str, _exts_url: str):
    checkboxes = []
    exts_header = widgets.HTML('<h3 style="width: 250px; text-align: center;">Extensions</h3>')
    status_header = widgets.HTML('<h3 style="width: 120px; text-align: center;">Status</h3>')
    homepages_header = widgets.HTML('<h3 style="width: 120px; text-align: center;">Homepages</h3>')
    headers = widgets.HBox([exts_header, status_header, homepages_header])
    output = widgets.Output()
    display(headers)

    def set_status(state): return f'<div style="text-align: center; width: 120px;">{state}</div>'

    for ext in get_resources(_exts_url):
        directory = f"{extensions_path}/{ext['url'].split('/')[-1]}"
        if os.path.exists(directory):
            install_status = 'installed'
            enabled = True
        else:
            install_status = 'not installed'
            enabled = True if ext['enabled'] else False

        checkbox = widgets.Checkbox(value=enabled, description=ext['name'], indent=False, layout={'width': '250px'})
        status = widgets.HTML(set_status(install_status))
        homepage = widgets.HTML(f'<div class="jp-RenderedText" style="width: 105px; text-align: center; white-space:nowrap; display: inline-grid;">'
                                f'<pre><a href="{ext["url"]}" target="_blank">GitHub</a></pre></div>')
        item = widgets.HBox([checkbox, status, homepage])
        checkboxes.append((ext, checkbox, status))
        display(item)

    def on_press(button):
        selected_exts = [(_ext['id'], _ext['name'], _ext['url'], _status) for _ext, _checkbox, _status in checkboxes if _checkbox.value]
        with output:
            output.clear_output()
            try:
                install_builtin_exts(_builtin_exts_url, update_exts.value)
                print("\n⏳ Installing selected extensions...")
                for _id, name, url, sts in selected_exts:
                    if _id == 'bmab':
                        run_process('pip install -q basicsr')
                    print(f'{name}...')
                    silent_clone(url, extensions_path, update=update_exts.value)
                    ext_dir = f"{extensions_path}/{url.split('/')[-1]}"
                    if os.path.exists(ext_dir): sts.value = set_status('installed')
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
def launch_webui(webui: WebUI):
    os.chdir(webui_path)
    print('⏳ Preparing...')
    print('It will take a little longer...')
    args = '--text-encoder-dir /temp-storage/text_encoder --disable-console-progressbars --disable-safe-unpickle --enable-insecure-extension-access --no-download-sd-model --no-hashing --api --xformers'
    proxy_url = 'http://127.0.0.1'
    webui_port = 7860
    replace_done = False
    run_process(f'python launch.py {args} --exit')

    if webui.ui == ui.auto1111:
        run_process('pip install -q pillow==9.5.0')

    if webui.platform == platform.paperspace:
        webui_port = port.paperspace
        proxy_url = f'https://tensorboard-{os.environ.get("PAPERSPACE_FQDN")}'
    elif webui.platform == platform.runpod:
        webui_port = port.runpod
        proxy_url = f'https://{os.environ.get("RUNPOD_POD_ID")}-{str(port.runpod + 1)}.proxy.runpod.net'

    def read(fd: int):
        nonlocal replace_done
        output = os.read(fd, 1024)
        if not replace_done:
            if output.decode().startswith('Running on local URL:'):
                replace_done = True
                return f'\nRunning on URL : {proxy_url}\n'.encode()
        return output

    if webui.dark_theme:
        args += ' --theme dark'
    if webui.username and webui.password:
        args += f' --gradio-auth {webui.username}:{webui.password}'
    if webui.ngrok_token:
        run_process('pip install -q ngrok')
        args += f' --ngrok {webui.ngrok_token}'
        if webui.ngrok_domain:
            ngrok_options = '{"domain":"' + webui.ngrok_domain + '"}'
            args += f' --ngrok-options {ngrok_options}'
    if webui.cors:
        args += f' --cors-allow-origins {webui.cors}'

    args += f' --listen --port {webui_port}'
    close_port(webui_port)
    print('Launching Web UI...')
    try:
        pty.spawn(f'python webui.py {args}'.split(), read)
    except KeyboardInterrupt:
        print('\n--Process terminated--')


def initialization(_ui_: str):
    apply_envs1()
    apply_envs2()
    update_deps()
    create_shared_storage()
    set_oncompleted_permission()
    remove_old_config()
    if _ui_ == ui.forge:
        remove_old_forge()
        install_forge()
    elif _ui_ == ui.auto1111:
        install_auto1111()
    completed_message()


def models_selection(models_url: str, base_model: str, civitai=''):
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
        homepage_label = widgets.HTML(f'<div class="jp-RenderedText" style="padding-left:20px; white-space:nowrap; display: inline-flex;">'
                                      f'<pre>{homepage_links}</pre></div>')
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
                    downloader(url, f'{models_path}/{base_model}', civitai_token=civitai)
                completed_message()
            except KeyboardInterrupt:
                print('\n\n--Download interrupted--')

    download_button.on_click(on_press)
    display(download_button, output)


# Download ControlNet
def download_controlnet(controlnet: list, url: str, base_model: str):
    controlnet_data = get_resources(url)
    for model in controlnet:
        if controlnet[model]:
            print('\n* ' + model + '...')
            for url in controlnet_data[model]:
                downloader(url, f'{controlnet_models_path}/{base_model}')


def get_res_directory(_type: str, base_model: str):
    directory = None
    match _type:
        case res_type.embedding:
            directory = f'{embeddings_path}/{base_model}'
        case res_type.lora:
            directory = f'{lora_path}/{base_model}'
        case res_type.upscaler:
            directory = upscaler_path
        case res_type.vae:
            directory = f'{vae_path}/{base_model}'
        case res_type.text_encoder:
            directory = f'{text_encoder_path}/{base_model}'
    return directory


# Download built-in resources
def download_builtin_resources(resources_url: str, base_model: str):
    resources = get_resources(resources_url)
    for resource_type, items in resources.items():
        directory = get_res_directory(resource_type, base_model)
        print(f'\n⏳ Downloading built-in {resource_type}...')
        for item in items:
            print(f"\n* {item['name']}...")
            if resource_type == res_type.embedding:
                silent_clone(item['url'], directory, True)
            else:
                downloader(item['url'], directory)


def resources_selection(builtin_res_url: str | None, resources_url: str, base_model: str, civitai=''):
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
            homepage_label = widgets.HTML(f'<div class="jp-RenderedText" style="padding-left: 0; white-space: nowrap; display: inline-flex;">'
                                          f'<pre>{homepage_links}</pre></div>')
            cb_item = widgets.HBox([checkbox, homepage_label])
            checkboxes.append((item, resource_type, checkbox))
            display(cb_item)

    download_button = widgets.Button(description='Download', button_style='success')
    output = widgets.Output()

    def on_press(button):
        selected_res = {
            res_type.lora: [],
            res_type.embedding: [],
            res_type.upscaler: [],
            res_type.vae: [],
            res_type.text_encoder: []
        }

        for _res, _type, _checkbox in checkboxes:
            if _checkbox.value:
                selected_res[_type].append((_res['name'], _res['url']))

        with output:
            output.clear_output()
            try:
                if builtin_res_url: download_builtin_resources(builtin_res_url, base_model)
                for _type_ in selected_res:
                    if selected_res[_type_]:
                        directory = get_res_directory(_type_, base_model)
                        print(f'\n⏳ Downloading selected {_type_}...')
                        for name, urls in selected_res[_type_]:
                            print(f'\n* {name}...')
                            if _type_ == res_type.embedding:
                                silent_clone(urls, directory, True)
                            else:
                                for url in urls:
                                    downloader(url, directory, civitai_token=civitai)
                completed_message()
            except KeyboardInterrupt:
                print('\n\n--Download interrupted--')

    download_button.on_click(on_press)
    print('')
    display(download_button, output)


# Other resources
def download_other_res(resource_list: list, resource_path: str, civitai=''):
    for resource in resource_list:
        print(f'\n* {resource}')
        downloader(resource, resource_path, civitai_token=civitai)


# Download other resources
def other_resources(other_res: OtherRes, base_model: str, civitai=''):
    if other_res.lora:
        print('\n\n⏳ Downloading LoRA...')
        download_other_res(other_res.lora, f'{lora_path}/{base_model}', civitai)
    if other_res.embedding:
        print('\n\n⏳ Downloading embedding...')
        download_other_res(other_res.embedding, f'{embeddings_path}/{base_model}', civitai)
    if other_res.upscaler:
        print('\n\n⏳ Downloading upscaler...')
        download_other_res(other_res.upscaler, upscaler_path, civitai)
    if other_res.vae:
        print('\n\n⏳ Downloading VAE...')
        download_other_res(other_res.vae, f'{vae_path}/{base_model}', civitai)
    if other_res.text_encoder:
        print('\n\n⏳ Downloading Text Encoder...')
        download_other_res(other_res.text_encoder, f'{text_encoder_path}/{base_model}', civitai)
