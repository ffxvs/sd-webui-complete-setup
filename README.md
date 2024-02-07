# Stable Diffusion Webui Complete Setup  
Jupyter notebook for Stable Diffusion Webui with a wide range of features.

![notebook-preview](https://github.com/ffxvs/sd-webui-complete-setup/assets/156585597/dd598e9b-bf07-4e46-81d4-7831b7b66568)

## Features  
* Support for Paperspace and Runpod, others to come.
* Checking the latest version of the notebook.
* Storing resources such as models, loRA, embeddings, outputs in shared storage (Paperspace).
* List of some popular extensions.
* List of several popular models in 4 categories, namely Anime/Cartoon/3D, General Purpose, Realistic, Woman.
* List of some useful resources for loRA, embedding, upscaler, and VAE.
* Install extension, loRA, embedding, upscaler, VAE from URLs.
* Zip outputs for easier download (Paperspace).

## How to use  
### Paperspace  
1. [Sign up](https://console.paperspace.com/signup) and subscribe to one of the [subscription plans](https://www.paperspace.com/gradient/pricing) (Pro or Growth)
2. Create a project then create a notebook in it.
3. Select the **"Start from Scratch"** template
4. Choose one of the free GPUs with at least 16GB VRAM (except Free-P5000).
5. Set the Auto-shutdown timeout to 6 hours.
6. Start notebook and wait.
7. Click the **"Open in Jupyterlab"** icon button in the left sidebar.
8. Download [sd_webui_paperspace.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/paperspace/sd_webui_paperspace.ipynb) for SD v1.5 and [sdxl_webui_paperspace.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/paperspace/sdxl_webui_paperspace.ipynb) for SDXL.
9. Upload them to the file browser in the left panel.
10. Read [this guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Paperspace-Guide) to use the notebook.

### Runpod  
1. [Sign up](https://www.runpod.io/console/signup) and add some credit to your [balance](https://www.runpod.io/console/user/billing).
2. Open [this link](https://www.runpod.io/console/gpu-browse) to create a Pod.
3. Choose a GPU with at least 16GB VRAM, for example RTX A4000, RTX A4500, RTX 3090.
4. Click Deploy. Set the Temporary Disk to 10 GB and the Persistent Volume to 20 GB or larger.
5. Make sure the template used is **"Runpod Pytorch 2.0.1"**.
6. Click Deploy and go to "My Pods".
7. On the Pod you just created, click **Connect** then **Connect to Jupyterlab**.
9. Download [sd_webui_paperspace.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/runpod/sd_webui_runpod.ipynb) for SD v1.5 and [sdxl_webui_paperspace.ipynb](https://ffxvs.github.io/sd-webui-complete-setup/runpod/sdxl_webui_runpod.ipynb) for SDXL.
10. Upload them to the file browser in the left panel.
11. Read [this guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Runpod-Guide) to use the notebook.
12. Stop the Pod if you don't use it anymore. Terminate the pod to delete the Pod and its content. Don't forget to download images you generated.
