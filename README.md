# Stable Diffusion Webui Complete Setup  
Jupyter notebook for Stable Diffusion Web UI and Stable Diffusion Web UI Forge.

![Notebook_preview](https://github.com/ffxvs/sd-webui-complete-setup/assets/156585597/52855d70-7c6b-42af-aa15-9ea67c99a5e5)

## Features  
* SD Web UI and SD Web UI Forge
* Support for Paperspace and Runpod
* Using small Docker containers under 5GB
* Notebook for installing Web UI, downloading SD v1.5 and SDXL models.
* Checking the latest version of the notebook.
* Storing resources such as models, loRA, embeddings, outputs in shared storage.
* List of some popular extensions.
* List of several popular models in 4 categories, namely Anime/Cartoon/3D, General Purpose, Realistic, Woman.
* List of some useful resources for loRA, embedding, upscaler, and VAE.
* Install extensions, loRA, embedding, upscaler, VAE from URLs.

## How to use  
### Paperspace  
1. [Sign up](https://console.paperspace.com/signup) and subscribe to one of the [subscription plans](https://www.paperspace.com/gradient/pricing) (Pro or Growth)
2. Create a project then create a notebook in it.
3. Select the **"Start from Scratch"** template
4. Choose one of the free GPUs with at least 16GB VRAM (except Free-P5000).
5. Set the Auto-shutdown timeout to 6 hours.
6. Click **View Advanced Options.**
7. Fill in the Container's name and command field as follows and leave other field blank. Just click copy button on the right.
   * Container's name
     * SD Web UI
       ```
       ffxvs/sd-webui-containers:auto1111-latest
       ```
     * SD Web UI Forge
       ```
       ffxvs/sd-webui-containers:forge-latest
       ```
   * Container's command
     ```
     bash /paperspace-start.sh
     ```
8. Start notebook and wait until the machine is running.
9. Click the **"Open in Jupyterlab"** icon button in the left sidebar.
10. There will be 3 ipynb notebook files.
   * `sd_webui_paperspace.ipynb` or `sd_webui_forge_paperspace.ipynb` for installing Web UI.
   * `sd15_resource_lists.ipynb` for downloading SD v1.5 models.
   * `sdxl_resource_lists.ipynb` for downloading SDXL models.
11. Read [Paperspace Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Paperspace-Guide) and [Resource Lists Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Resource-Lists-Guide) to use the notebooks.

<br>

### Runpod  
1. [Sign up](https://runpod.io?ref=synjcfeg) and add some credit to your [balance](https://www.runpod.io/console/user/billing).
2. Open one of these template to create a Pod : [SD Web UI](https://runpod.io/console/gpu-cloud?template=38adx50leu&ref=synjcfeg) / [SD Web UI Forge](https://runpod.io/console/gpu-cloud?template=kwef1wl832&ref=synjcfeg)
3. Make sure the template is : 
   * SD Web UI : `ffxvs/sd-webui-containers:auto1111-latest`
   * SD Web UI Forge : `ffxvs/sd-webui-containers:forge-latest`
4. Select _Secure Cloud_ if you want to use _Network Volume (Persistent Storage)_, or Community Cloud if you want to use cheaper GPU.
5. Choose a GPU with at least 16GB VRAM, for example RTX A4000, RTX A4500, RTX 3090.
6. Continue and Deploy, then go to My Pods. Wait until the Pod is ready.
7. On the Pod you just created, click **Connect** then **Connect to HTTP Service [Port 8888]** to open Jupyterlab.
8. There will be 3 ipynb notebook files.
   * `sd_webui_runpod.ipynb` or `sd_webui_forge_runpod.ipynb` for installing Web UI.
   * `sd15_resource_lists.ipynb` for downloading SD v1.5 models.
   * `sdxl_resource_lists.ipynb` for downloading SDXL models.
9. Read [Runpod Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Runpod-Guide) and [Resource Lists Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Resource-Lists-Guide) to use the notebooks.
10. You can click **Connect to HTTP Service [Port 3001]** after installing and launching the Web UI.
11. Stop the Pod if you don't use it anymore. Terminate the pod to delete the Pod and its content. Don't forget to download images you generated.

<br>

### Download Notebooks manually
In case you can't download the latest notebooks from **Check for Updates** cell, you can download them manually from the links below :
- SD Web UI for Paperspace : [sd_webui_paperspace.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/sd-webui/sd_webui_paperspace.ipynb)
- SD Web UI for RunPod : [sd_webui_runpod.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/sd-webui/sd_webui_runpod.ipynb) 
- SD Web UI Forge for Paperspace : [sd_webui_forge_paperspace.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/sd-webui-forge/sd_webui_forge_paperspace.ipynb)
- SD Web UI Forge for RunPod : [sd_webui_forge_runpod.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/sd-webui-forge/sd_webui_forge_runpod.ipynb)
- SD v1.5 resource lists : [sd15_resource_lists.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/resource-lists/sd15_resource_lists.ipynb)
- SDXL resource lists : [sdxl_resource_lists.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/resource-lists/sdxl_resource_lists.ipynb)

<br>

## Credits
* [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) by TheLastBen
* [stable-diffusion-webui-colab](https://github.com/camenduru/stable-diffusion-webui-colab) by Camenduru
* [stablediffusion_webui](https://github.com/sagiodev/stablediffusion_webui) by sagiodev
