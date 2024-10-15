# Stable Diffusion WebUI Complete Setup  
Jupyter notebook for Stable Diffusion WebUI and Stable Diffusion WebUI Forge.

***

### [Major Updates 2024.09.21 Announcement](https://github.com/ffxvs/sd-webui-complete-setup/discussions/15)  
### [Update 2024.10.15 Announcement](https://github.com/ffxvs/sd-webui-complete-setup/discussions/)  

***

**Buy Me a Coffee**  
<a href="https://sociabuzz.com/ffxvs/tribe" target="_blank">
    <img src="https://storage.sociabuzz.com/storage/landingpage/img/sociabuzz-logo.png" height="35px" style="border:0;height:35px;">
</a>

<br>

![banner](https://github.com/user-attachments/assets/250bf979-d02f-4021-8ea3-4b31bf415514)

## Features  
* Automatic updates.
* SD WebUI Auto1111 and SD WebUI Forge
* Support for Paperspace and Runpod
* Using small Docker containers under 6GB
* Notebook for installing WebUI, downloading SD v1.5, SDXL and FLUX models.
* Storing resources such as models, loRA, embeddings, outputs in shared storage.
* List of some popular extensions.
* List of several popular models in 4 categories, namely Anime/Cartoon/3D, General Purpose, Realistic, Woman.
* List of some useful resources for loRA, embedding, upscaler, VAE and text encoder.
* Install extensions, loRA, embedding, upscaler, VAE and text encoder from URLs.

## How to use  
### Paperspace  
1. [Sign up](https://console.paperspace.com/signup) and subscribe to one of the [subscription plans](https://www.paperspace.com/gradient/pricing) (Pro or Growth)
2. Create a project then create a notebook in it.
3. Select the **"Start from Scratch"** template
4. Choose one of the free GPUs with at least 16GB VRAM (except Free-P5000).
5. Set the Auto-shutdown timeout to 6 hours.
6. Click **View Advanced Options.**
7. Fill in the Container's name and command field as follows and leave other field blank. Just click copy button on the right.  
   **Note : Paperspace caches the old container in their server, so when there is a new version of the container, you have to create new installation again.**
   * Container's name
     * SD WebUI Auto1111
       ```
       ffxvs/sd-webui-containers:auto1111-2024.09.21
       ```
     * SD WebUI Forge
       ```
       ffxvs/sd-webui-containers:forge-2024.09.21
       ```
   * Container's command
     ```
     bash /paperspace-start.sh
     ```
   
8. Start notebook and wait until the machine is running.
9. Duplicate your tab to keep the console open. Then click the **"Open in Jupyterlab"** button (the orange circle icon) in the left sidebar.
10. There will be 3 ipynb notebook files.
    * `sd_webui_paperspace.ipynb` or `sd_webui_forge_paperspace.ipynb` for installing WebUI.
    * `sd15_resource_lists.ipynb` for downloading SD v1.5 models.
    * `sdxl_resource_lists.ipynb` for downloading SDXL models.
    * `flux_resource_lists.ipynb` for downloading FLUX models.
11. Read [Paperspace Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Paperspace-Guide) and [Resource Lists Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Resource-Lists-Guide) to use the notebooks.

<br>

### Runpod  
1. [Sign up](https://runpod.io?ref=synjcfeg) and add some credit to your [balance](https://www.runpod.io/console/user/billing).
2. Open one of these template to create a Pod : [SD WebUI Auto1111](https://runpod.io/console/deploy?template=38adx50leu&ref=synjcfeg) / [SD WebUI Forge](https://runpod.io/console/deploy?template=kwef1wl832&ref=synjcfeg)
3. Make sure the template is : 
   * SD WebUI Auto1111 : `ffxvs/sd-webui-containers:auto1111-latest`
   * SD WebUI Forge : `ffxvs/sd-webui-containers:forge-latest`
4. Select _Secure Cloud_ if you want to use _Network Volume (Persistent Storage)_, or Community Cloud if you want to use cheaper GPU.
5. Choose a GPU with at least 16GB VRAM, for example RTX A4000, RTX A4500, RTX 3090.
6. Continue and Deploy, then go to My Pods. Wait until the Pod is ready.
7. On the Pod you just created, click **Connect** then **Connect to HTTP Service [Port 8888]** to open Jupyterlab.
8. There will be 3 ipynb notebook files.
   * `sd_webui_runpod.ipynb` or `sd_webui_forge_runpod.ipynb` for installing WebUI.
   * `sd15_resource_lists.ipynb` for downloading SD v1.5 models.
   * `sdxl_resource_lists.ipynb` for downloading SDXL models.
   * `flux_resource_lists.ipynb` for downloading FLUX models.
9. Read [Runpod Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Runpod-Guide) and [Resource Lists Guide](https://github.com/ffxvs/sd-webui-complete-setup/wiki/Resource-Lists-Guide) to use the notebooks.
10. You can click **Connect to HTTP Service [Port 3001]** after installing and launching the WebUI.
11. Stop the Pod if you don't use it anymore. Terminate the pod to delete the Pod and its content. Don't forget to download images you generated.

<br>

## Credits
* [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) by TheLastBen
* [stable-diffusion-webui-colab](https://github.com/camenduru/stable-diffusion-webui-colab) by Camenduru
* [stablediffusion_webui](https://github.com/sagiodev/stablediffusion_webui) by sagiodev
