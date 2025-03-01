import json
import os
import pty
import shlex
import shutil
import signal
import subprocess
import time
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import dotenv
import ipywidgets as widgets
import requests
from IPython.display import display

# #################### GLOBAL PATHS ####################


branch_id = os.environ['BRANCH_ID']
webui_id = os.environ['WEBUI_ID']
webui_dir = os.environ['WEBUI_DIR']
platform_id = os.environ['PLATFORM_ID']

root = '/notebooks'
webui_path = Path(root + webui_dir)
oncompleted_path = '/internal/on-completed.sh'
outputs_path = webui_path / 'outputs'
extensions_path = webui_path / 'extensions'
modules_path = webui_path / 'modules'

models_path = webui_path / 'models/Stable-diffusion'
embeddings_path = webui_path / 'embeddings'
lora_path = webui_path / 'models/Lora'
esrgan_path = webui_path / 'models/ESRGAN'
dat_path = webui_path / 'models/DAT'
vae_path = webui_path / 'models/VAE'
controlnet_models_path = webui_path / 'models/ControlNet'
text_encoder_path = webui_path / 'models/text_encoder'

shared_storage = Path(root + '/shared-storage')
shared_models_path = shared_storage / 'models'
shared_embeddings_path = shared_storage / 'embeddings'
shared_lora_path = shared_storage / 'lora'
shared_esrgan_path = shared_storage / 'upscaler/esrgan'
shared_dat_path = shared_storage / 'upscaler/dat'
shared_vae_path = shared_storage / 'vae'
shared_controlnet_models_path = shared_storage / 'controlNet'
shared_controlnet_preprocessor_path = shared_controlnet_models_path / 'preprocessor'
shared_text_encoder_path = shared_storage / 'text_encoder'
shared_outputs_path = shared_storage / 'outputs'
shared_config_path = shared_storage / 'config'
env_path = shared_storage / '.env'

temp_storage = Path('/temp-storage')
temp_models_path = temp_storage / 'models'
temp_lora_path = temp_storage / 'lora'
temp_controlnet_models_path = temp_storage / 'controlNet'
temp_controlnet_preprocessor_path = temp_controlnet_models_path / 'preprocessor'
temp_text_encoder_path = temp_storage / 'text_encoder'

# #################### RESOURCE URLs ####################

main_repo_url = f'https://raw.githubusercontent.com/ffxvs/sd-webui-complete-setup/{branch_id}'
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
        self.runpod = 'RUNPOD'
        self.paperspace = 'PAPERSPACE'


class Port:
    def __init__(self):
        self.runpod = 3000
        self.paperspace = 6006


class UI:
    def __init__(self):
        self.auto1111 = 'AUTO1111'
        self.forge = 'FORGE'


class Envs:
    def __init__(self):
        self.CIVITAI_TOKEN = 'CIVITAI_TOKEN'
        self.HUGGINGFACE_TOKEN = 'HUGGINGFACE_TOKEN'
        self.NGROK_TOKEN = 'NGROK_TOKEN'
        self.NGROK_DOMAIN = 'NGROK_DOMAIN'
        self.CUSTOM_CONFIG_URL = 'CUSTOM_CONFIG_URL'
        self.CUSTOM_UI_CONFIG_URL = 'CUSTOM_UI_CONFIG_URL'
        self.AUTOUPDATE_FORGE = 'AUTOUPDATE_FORGE'
        self.DARK_THEME = 'DARK_THEME'


class WebUI:
    def __init__(self, username: str, password: str, cors: str):
        self.username = username
        self.password = password
        self.cors = cors


class OtherRes:
    def __init__(self, lora: list, embedding: list, esrgan: list, dat: list, vae: list, text_encoder: list):
        self.lora = lora
        self.embedding = embedding
        self.esrgan = esrgan
        self.dat = dat
        self.vae = vae
        self.text_encoder = text_encoder


# #################### VARIABLES ####################

base = BaseModel()
res_type = ResourceType()
platform = Platform()
port = Port()
ui = UI()

other = OtherRes([], [], [], [], [], [])
webUI = WebUI('', '', '')
envs = Envs()

boolean = [False, True]
request_headers = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
}

if webui_id == ui.forge:
    controlnet_preprocessor_path = webui_path / 'models/ControlNetPreprocessor'
else:
    controlnet_preprocessor_path = extensions_path / 'sd-webui-controlnet/annotator/downloads'


# #################### FUNCTIONS ####################

# Apply environment variables 1
def apply_envs1():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'


# Apply environment variables 2
def apply_envs2():
    os.environ['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libtcmalloc.so.4'
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


def base_model():
    return os.environ['BASE_MODEL']


def get_env(key: str, default=None) -> str | None:
    dotenv.load_dotenv(env_path, override=True)
    return os.environ.get(key, default)


def civitai_token():
    return get_env(envs.CIVITAI_TOKEN)


def huggingface_token():
    return get_env(envs.HUGGINGFACE_TOKEN)


def ngrok_token():
    return get_env(envs.NGROK_TOKEN)


def ngrok_domain():
    return get_env(envs.NGROK_DOMAIN)


def custom_config_url():
    return get_env(envs.CUSTOM_CONFIG_URL)


def custom_ui_config_url():
    return get_env(envs.CUSTOM_UI_CONFIG_URL)


def autoupdate_forge() -> bool:
    return get_env(envs.AUTOUPDATE_FORGE, 'True').lower() == 'true'


def dark_theme() -> bool:
    return get_env(envs.DARK_THEME, 'True').lower() == 'true'


# Run external program
def run_process(command: str, use_shell=False):
    if not use_shell:
        command = shlex.split(command)
    return subprocess.run(command, shell=use_shell, capture_output=True, text=True, bufsize=1)


# Update ubuntu dependencies
def update_deps():
    print('⏳ Updating dependencies...')
    run_process('apt -y -q update')


def set_oncompleted_permission():
    run_process(f'chmod +x {oncompleted_path}')


# Create symlink
def symlink(source: str | Path, destination: str | Path):
    if os.path.exists(source) and not os.path.islink(destination):
        shutil.rmtree(destination, ignore_errors=True)
        os.symlink(source, destination)


# Remove symlink
def unlink(target: str | Path):
    if os.path.islink(target):
        os.unlink(target)


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


def remove_old_dirs():
    unlink(models_path / 'ControlNetPreprocessor')
    old_upscaler_path = shared_storage / 'esrgan'
    if os.path.exists(old_upscaler_path):
        shutil.rmtree(old_upscaler_path, ignore_errors=True)
    if os.path.islink(controlnet_preprocessor_path):
        link_path = os.readlink(controlnet_preprocessor_path)
        if link_path != shared_controlnet_preprocessor_path:
            unlink(controlnet_preprocessor_path)


def sync_config():
    config_file = shared_config_path / 'config.json'
    if os.path.exists(config_file):
        try:
            default_order = ["txt2img", "Txt2img", "img2img", "Img2img", "Extras",
                             "PNG Info", "Checkpoint Merger", "Train", "Cleaner",
                             "Mini Paint", "Photopea", "Infinite image browsing"]
            with open(config_file, 'r') as file:
                config = json.load(file)

            # Add 'ui_tab_order' if it doesn't exist or empty
            if 'ui_tab_order' not in config or not config['ui_tab_order']:
                config['ui_tab_order'] = default_order

            ui_tab_order: list = config['ui_tab_order']

            # Remove "Model Downloader" from 'ui_tab_order'
            if 'Model Downloader' in config['ui_tab_order']:
                ui_tab_order.remove('Model Downloader')

            # Add "Txt2img" and "Img2img" if they don't exist
            if ui_tab_order[0] == 'txt2img' and ui_tab_order[1] == 'img2img':
                ui_tab_order.insert(1, 'Txt2img')
                ui_tab_order.insert(3, 'Img2img')

            config['show_progress_type'] = 'TAESD'
            config['show_progress_every_n_steps'] = 4
            config['live_previews_image_format'] = 'jpeg'

            keys_to_add = ['sd_t2i_height', 'xl_t2i_width', 'flux_t2i_width']
            for key in keys_to_add:
                if key not in config:
                    config[key] = 768

            with open(config_file, 'w') as file:
                json.dump(config, file, indent=4)
        except (Exception, json.JSONDecodeError) as e:
            print(e)


def remove_old_forge():
    forge_path = f'{root}/stable-diffusion-webui-forge'
    if os.path.exists(forge_path):
        last_commit = run_process(f'git -C {forge_path} log -1 --oneline').stdout.strip()
        if last_commit.startswith('bfee03d'):
            shutil.rmtree(forge_path, ignore_errors=True)


def temp_storage_symlink(option: bool, source: str, destination: str):
    if option:
        symlink(source, destination)
    else:
        unlink(destination)
        run_process(f'rm -r -f {source}/*', use_shell=True)
        os.makedirs(destination, exist_ok=True)


# Create shared storage
def create_shared_storage():
    print('⏳ Creating shared storage directory...')
    if platform_id == platform.paperspace:
        symlink('/storage', shared_storage)
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
        shared_controlnet_preprocessor_path,
        shared_esrgan_path,
        shared_dat_path,
        shared_outputs_path,
        shared_config_path
    ]

    for folder in shared_storage_folders:
        os.makedirs(folder, exist_ok=True)


# Create symlinks from temporary storage to shared storage
def temp_storage_settings():
    checkboxes = []
    ts_list = [
        ('SD v1.5 Models', temp_models_path / 'sd', shared_models_path / 'sd'),
        ('SD v1.5 LoRA', temp_lora_path / 'sd', shared_lora_path / 'sd'),
        ('SD v1.5 ControlNet', temp_controlnet_models_path / 'sd', shared_controlnet_models_path / 'sd'),
        ('SDXL Models', temp_models_path / 'sdxl', shared_models_path / 'sdxl'),
        ('SDXL LoRA', temp_lora_path / 'sdxl', shared_lora_path / 'sdxl'),
        ('SDXL ControlNet', temp_controlnet_models_path / 'sdxl', shared_controlnet_models_path / 'sdxl'),
        ('FLUX Models', temp_models_path / 'flux', shared_models_path / 'flux'),
        ('FLUX LoRA', temp_lora_path / 'flux', shared_lora_path / 'flux'),
        ('FLUX ControlNet', temp_controlnet_models_path / 'flux', shared_controlnet_models_path / 'flux'),
        ('FLUX Text Encoder', temp_text_encoder_path / 'flux', shared_text_encoder_path / 'flux'),
        ('ControlNet Preprocessor', temp_controlnet_preprocessor_path, shared_controlnet_preprocessor_path)
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
    symlink(shared_esrgan_path, esrgan_path)
    symlink(shared_dat_path, dat_path)
    symlink(shared_vae_path, vae_path)
    symlink(shared_controlnet_models_path, controlnet_models_path)
    symlink(shared_controlnet_preprocessor_path, controlnet_preprocessor_path)
    symlink(shared_outputs_path, outputs_path)
    symlink(shared_text_encoder_path, text_encoder_path)
    symlink(shared_config_path / 'config.json', webui_path / 'config.json')
    symlink(shared_config_path / 'ui-config.json', webui_path / 'ui-config.json')


def webui_settings():
    settings = []
    input_list = [
        (envs.CIVITAI_TOKEN, 'CivitAI API Key', 'Paste your API key here', civitai_token(), False),
        (envs.HUGGINGFACE_TOKEN, 'HuggingFace Token', 'Paste your HuggingFace token here', huggingface_token(), False),
        (envs.NGROK_TOKEN, 'Ngrok Token', 'Paste your Ngrok token here', ngrok_token(), False),
        (envs.NGROK_DOMAIN, 'Ngrok Domain', 'Paste your Ngrok domain here', ngrok_domain(), False),
        (envs.CUSTOM_CONFIG_URL, 'Custom config.json URL', 'Paste your config.json URL here', custom_config_url(), True),
        (envs.CUSTOM_UI_CONFIG_URL, 'Custom ui-config.json URL', 'Paste your ui-config.json URL here', custom_ui_config_url(), True)
    ]

    checkbox_list = [
        (envs.AUTOUPDATE_FORGE, 'Auto Update Forge', autoupdate_forge()),
        (envs.DARK_THEME, 'Enable Dark Theme', dark_theme())
    ]

    save_button = widgets.Button(description='Save', button_style='success')
    output = widgets.Output()

    def callback(button, _key: str, _textfield: widgets.Text):
        with output:
            output.clear_output()
            time.sleep(0.5)
            value = str(_textfield.value).strip()
            if len(value) > 0:
                if _key == envs.CUSTOM_CONFIG_URL:
                    print('\nDownload config.json')
                    downloader(value, shared_config_path, overwrite=True)
                elif _key == envs.CUSTOM_UI_CONFIG_URL:
                    print('\nDownload ui-config.json')
                    downloader(value, shared_config_path, overwrite=True)

    for key, input_label, placeholder, input_value, use_btn in input_list:
        label = widgets.Label(input_label, layout=widgets.Layout(width='200px'))
        textfield = widgets.Text(placeholder=placeholder, value=input_value, layout=widgets.Layout(width='400px'))
        settings.append((key, textfield))
        row = [label, textfield]

        if use_btn:
            dl_btn = widgets.Button(description='Download', button_style='success', layout=widgets.Layout(width='100px', margin='2px 0 0 25px'))
            dl_btn.on_click(partial(callback, _key=key, _textfield=textfield))
            row.append(dl_btn)

        print('')
        display(widgets.HBox(row))

    for cb_env_name, cb_label, cb_value, in checkbox_list:
        checkbox = widgets.Checkbox(description=cb_label, value=cb_value, indent=False)
        settings.append((cb_env_name, checkbox))
        print('')
        display(checkbox)

    def on_save(button):
        with output:
            env_path.touch(mode=0o666, exist_ok=True)
            output.clear_output()
            for env_key, option in settings:
                value = str(option.value).strip()
                if len(value) > 0:
                    dotenv.set_key(env_path, env_key, value)
                else:
                    if get_env(env_key) is not None:
                        os.environ.pop(env_key)
                        dotenv.unset_key(env_path, env_key)
            time.sleep(0.5)
            print('\nSaved ✔')

    print('')
    save_button.on_click(on_save)
    display(save_button, output)


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
def downloader(url: str, path: str | Path, overwrite=False):
    prev_line = ''
    filename = os.path.basename(urlparse(url).path)

    if url.startswith('https://civitai.com/api/download/') and civitai_token() is not None:
        url += f'&token={civitai_token()}' if '?' in url else f'?token={civitai_token()}'

    aria2c = f'stdbuf -oL aria2c {url} -d {path} --on-download-complete={oncompleted_path} --download-result=hide --console-log-level=error -c -x 16 -s 16 -k 1M'

    if url.startswith('https://huggingface.co/') and huggingface_token() is not None:
        aria2c += f' --header="Authorization: Bearer {huggingface_token()}"'

    if overwrite:
        aria2c += ' --allow-overwrite'
    if '.' in filename and filename.split('.')[-1] != '':
        aria2c += f' -o {filename}'

    with subprocess.Popen(shlex.split(aria2c), stdout=subprocess.PIPE, text=True, bufsize=1) as sp:
        for line in sp.stdout:
            if line.startswith('[#'):
                text = 'Download progress {}'.format(line.strip('\n'))
                print('\r' + ' ' * 100 + '\r' + text, end='\r', flush=True)
                prev_line = text
            elif line.startswith('[COMPLETED]'):
                if prev_line != '': print('')
                print(f'Download completed')
            else:
                print(line)


# Git clone repo
def clone_repo(command: str, path: str | Path, update=False, overwrite=False):
    directory = f'{path}/{command.split("/")[-1]}'
    git_clone = f'git clone -q --depth 20 {command} {directory}'
    if os.path.exists(directory):
        if update:
            os.chdir(f'{directory}')
            run_process('git pull -q')
        elif overwrite:
            shutil.rmtree(directory, ignore_errors=True)
            run_process(git_clone)
    else:
        run_process(git_clone)


# Download files using WGet
def wget(command: str):
    run_process(f'wget -nv {command}')


def install_auto1111():
    os.chdir(root)
    print('⏳ Installing/Updating Stable Diffusion Web UI...')
    webui_version = 'v1.10.1'
    clone_repo(f'-b {webui_version} https://github.com/AUTOMATIC1111/stable-diffusion-webui', root)
    os.chdir(webui_path)
    run_process('git remote set-branches origin master')
    run_process(f'git fetch origin tag {webui_version} -q --depth=5')
    run_process(f'git checkout -q -f {webui_version}')

    # Make sure controlNet is installed before creating controlNet preprocessor symlink from shared storage
    clone_repo('https://github.com/Mikubill/sd-webui-controlnet', extensions_path)
    downloader(f'https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/{webui_version}/modules/extras.py', modules_path, True)
    downloader(f'https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/{webui_version}/modules/sd_models.py', modules_path, True)

    # From ThelastBen
    run_process(f'sed -i \'s@shared.opts.data\["sd_model_checkpoint"] = checkpoint_info.title@shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title; model.half()@\' {modules_path}/sd_models.py')
    run_process(f"sed -i \"s@map_location='cpu'@map_location='cuda'@\" {modules_path}/extras.py")


# Install Web UI Forge
def install_forge():
    os.chdir(root)
    print('⏳ Installing/Updating Stable Diffusion Web UI Forge...')
    webui_version = 'main'
    clone_repo(f'-b {webui_version} https://github.com/lllyasviel/stable-diffusion-webui-forge', root, update=autoupdate_forge())
    os.chdir(webui_path)


def download_configs():
    config_url = custom_config_url() if custom_config_url() else f'{main_repo_url}/configs/config.json'
    ui_config_url = custom_ui_config_url() if custom_ui_config_url() else f'{main_repo_url}/configs/ui-config.json'

    if not os.path.exists(shared_config_path / 'config.json'):
        print('Download config.json')
        downloader(config_url, shared_config_path)
    if not os.path.exists(shared_config_path / 'ui-config.json'):
        print('Download ui-config.json')
        downloader(ui_config_url, shared_config_path)


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
                        run_process('pip install -q https://huggingface.co/deauxpas/colabrepo/resolve/main/basicsr-1.4.2-py3-none-any.whl')
                    print(f'{name}...')
                    clone_repo(url, extensions_path, update=update_exts.value)
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
        clone_repo(ext['url'], extensions_path, update)


# Install other extensions
def install_other_exts(extensions: list, update_exts=False):
    if extensions:
        print("⏳ Installing extensions...")
        for ext in extensions:
            name = ext.split('/')[-1]
            print(name + '...')
            clone_repo(ext, extensions_path, update_exts)


# Launch Web UI
def launch_webui(webui: WebUI):
    os.chdir(webui_path)
    print('⏳ Preparing...')
    print('It will take a little longer...')
    args = '--loglevel WARNING --disable-console-progressbars --disable-safe-unpickle --enable-insecure-extension-access --no-download-sd-model --no-hashing --api --xformers'
    proxy_url = 'http://127.0.0.1'
    webui_port = 7860
    replace_done = False
    run_process(f'python launch.py {args} --exit')

    if webui_id == ui.auto1111:
        run_process('pip install -q pillow==9.5.0')

    if webui_id == ui.forge:
        args += ' --text-encoder-dir /temp-storage/text_encoder'

    if platform_id == platform.paperspace:
        webui_port = port.paperspace
        proxy_url = f'https://tensorboard-{os.environ.get("PAPERSPACE_FQDN")}'
    elif platform_id == platform.runpod:
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

    if dark_theme():
        args += ' --theme dark'
    if webui.username and webui.password:
        args += f' --gradio-auth {webui.username}:{webui.password}'
    if ngrok_token():
        run_process('pip install -q ngrok')
        args += f' --ngrok {ngrok_token()}'
        if ngrok_domain():
            ngrok_options = '{\\"domain\\":\\"' + ngrok_domain() + '\\"}'
            args += f' --ngrok-options {ngrok_options}'
    if webui.cors:
        args += f' --cors-allow-origins {webui.cors}'

    args += f' --listen --port {webui_port}'
    print('Launching WebUI...')
    try:
        close_port(webui_port)
        os.chdir(webui_path)
        pty.spawn(shlex.split(f'python webui.py {args}'), read)
    except KeyboardInterrupt:
        close_port(webui_port)
        print('\n--Process terminated--')


def initialization():
    apply_envs1()
    apply_envs2()
    update_deps()
    create_shared_storage()
    set_oncompleted_permission()
    remove_old_dirs()
    sync_config()
    if webui_id == ui.forge:
        # remove_old_forge()
        install_forge()
    elif webui_id == ui.auto1111:
        install_auto1111()
    download_configs()
    shared_storage_symlinks()
    completed_message()


def models_selection(models_url: str):
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
                    downloader(url, models_path / base_model())
                completed_message()
            except KeyboardInterrupt:
                print('\n\n--Download interrupted--')

    download_button.on_click(on_press)
    display(download_button, output)


# Download ControlNet
def download_controlnet(controlnet: list, url: str):
    controlnet_data = get_resources(url)
    for model in controlnet:
        if controlnet[model]:
            print('\n* ' + model + '...')
            for url in controlnet_data[model]:
                downloader(url, controlnet_models_path / base_model())


def get_res_directory(_type: str):
    directory = None
    match _type:
        case res_type.embedding:
            directory = embeddings_path / base_model()
        case res_type.lora:
            directory = lora_path / base_model()
        case res_type.upscaler:
            directory = esrgan_path
        case res_type.vae:
            directory = vae_path / base_model()
        case res_type.text_encoder:
            directory = text_encoder_path / base_model()
    return directory


# Download built-in resources
def download_builtin_resources(resources_url: str):
    resources = get_resources(resources_url)
    for resource_type, items in resources.items():
        directory = get_res_directory(resource_type)
        print(f'\n⏳ Downloading built-in {resource_type}...')
        for item in items:
            print(f"\n* {item['name']}...")
            if resource_type == res_type.embedding:
                clone_repo(item['url'], directory, True)
            else:
                downloader(item['url'], directory)


def resources_selection(builtin_res_url: str | None, resources_url: str):
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
                if builtin_res_url: download_builtin_resources(builtin_res_url)
                for _type_ in selected_res:
                    if selected_res[_type_]:
                        directory = get_res_directory(_type_)
                        print(f'\n⏳ Downloading selected {_type_}...')
                        for name, urls in selected_res[_type_]:
                            print(f'\n* {name}...')
                            if _type_ == res_type.embedding:
                                clone_repo(urls, directory, True)
                            else:
                                for url in urls:
                                    downloader(url, directory)
                completed_message()
            except KeyboardInterrupt:
                print('\n\n--Download interrupted--')

    download_button.on_click(on_press)
    print('')
    display(download_button, output)


# Other resources
def download_other_res(resource_list: list, resource_path: str | Path):
    try:
        for resource in resource_list:
            print(f'\n* {resource}')
            downloader(resource, resource_path)
    except KeyboardInterrupt:
        print('\n\n--Download interrupted--')


# Download other resources
def other_resources(other_res: OtherRes):
    if other_res.lora:
        print('\n\n⏳ Downloading LoRA...')
        download_other_res(other_res.lora, lora_path / base_model())
    if other_res.embedding:
        print('\n\n⏳ Downloading embedding...')
        download_other_res(other_res.embedding, embeddings_path / base_model())
    if other_res.esrgan:
        print('\n\n⏳ Downloading ESRGAN-based Upscaler...')
        download_other_res(other_res.dat, esrgan_path)
    if other_res.dat:
        print('\n\n⏳ Downloading DAT-based Upscaler...')
        download_other_res(other_res.dat, dat_path)
    if other_res.vae:
        print('\n\n⏳ Downloading VAE...')
        download_other_res(other_res.vae, vae_path / base_model())
    if other_res.text_encoder:
        print('\n\n⏳ Downloading Text Encoder...')
        download_other_res(other_res.text_encoder, text_encoder_path / base_model())
