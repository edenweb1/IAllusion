# %%
# !! {"metadata":{
# !!   "id":"cc-imports"
# !! }}

#<cc-imports>

import subprocess

# %%
# !! {"metadata":{
# !!   "id": "nVTCvBBnFZ2h"
# !! }}
"""
# MindsEye beta (ai art pilot by multimodal.art)
A GUI for generating multimodal art (text-to-image) with multiple models (including Disco Diffusion v5 by <a href="https://twitter.com/somnai_dreams" target="_blank">@somnai_dreams</a> and <a href="https://twitter.com/gandamu" target="_blank">@gandamu</a> and Hypertron v2 VQGAN+CLIP by <a href="https://github.com/Philipuss1" target="_blank">Philipuss</a>).

To run the application:
1. Press `CTRL+F9` (or `command + F9` on a Mac), or go to the Menu, `Runtime > Run all`
![Instructions](https://i.imgur.com/59eRrO0.png)
2. When prompted to connect with Google Drive, accept it to have models and gallery there. If you refuse it all will still work, but you'll lose everything after closing the Colab tab
2. Wait for the wheel below `Install the requirements` to finish spinning (where it reads "4 cells hidden")
2. The wheel at `"Run streamlit (GUI app)..."` should **stay** spinning
4. Click the link that will come after `"your url is..."`. You can **ignore** the Network URL and Extrnal URL links.

If questions still remain on how to launch the application, check out our [Guide](https://multimodal.art/mindseye), or come hang out on [Discord](https://discord.gg/kepXxmv6) or [Twitter](https://twitter.com/multimodalart)
"""

# %%
# !! {"metadata":{
# !!   "id": "vr-fG_VkLGRV"
# !! }}
"""
#### Credits

"""

# %%
# !! {"metadata":{
# !!   "id": "O2_NuBvOLT2F"
# !! }}
"""
Disco Diffusion v5 model by <a href="#">@somnai_dreams</a> and <a href="#">@gandamu</a>, based on the fundational work of <a href="#">@RiversHaveWings</a>, with modifications by <a href="#">@danielrussruss</a>, Dango223, [Chigozie Nri's](https://twitter.com/chigozienri), <a href="#">@softology</a> and others.<br>

<a href="#">Hypertron v2</a> VQGAN model by Philipuss adapted from <a href="#">@RiversHaveWings</a> with modifications by <a href="#">@jbusted</a>, <a href="#">@softology</a> and others. Original GAN+CLIP by <a href="#">@advadnoun</a>. 

CLIP and Diffusion were released by OpenAI. VQGAN by CompVis Heidelberg</small>
"""

# %%
# !! {"metadata":{
# !!   "id": "20LWy-FPFot9"
# !! }}
"""
#### Install the requirements (may take around 3-5 minutes, don't give up!)
(If you wish to save models and generated images on Google Drive, connect with it when prompted `recommended`)
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "hyV7ukOVpHYa"
# !! }}
#@title 1.1 Check GPU time
#@markdown ### Factory reset runtime if you don't have the desired GPU.

#@markdown ---

#@markdown V100 = Excellent (*Available only for Colab Pro Users*)

#@markdown P100 = Very Good

#@markdown T4 = Good

#@markdown K80 = Meh

#@markdown P4 = (Not Recommended)

#@markdown ---
import subprocess



# %%
# !! {"metadata":{
# !!   "id": "iMgK7GklaEFU",
# !!   "cellView": "form"
# !! }}
#@title 1.2 Anti disconnect from Colab
#@markdown ## This will increase your session time
#@markdown (It will anyhow disconnect after 6 - 12 hrs for using the free version of Colab.
#@markdown Colab Pro users will get about 24 hrs usage time)

import IPython
js_code = '''
function ClickConnect(){
console.log("Working");
document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
'''


# %%
# !! {"metadata":{
# !!   "id": "RRBenH-Wv-Cu",
# !!   "cellView": "form"
# !! }}
#@title 1.3 Install Dependencies

import sys
import torch
import os
try:
    from google.colab import drive
    is_colab = True
    try:
      drive.mount('/content/drive')
      is_drive = True
    except:
      is_colab = True
      is_drive = False
except:
    is_colab = False
    is_drive = False
    
sub_p_res = subprocess.run(['pip', 'install', 'streamlit==1.7.0'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'wget'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'fvcore', 'iopath', 'lpips', 'datetime', 'timm', 'ftfy'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'pytorch-lightning'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'omegaconf'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'einops'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'stqdm'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'kora'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'imageio'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'kornia'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'pathvalidate'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'dalle_pytorch'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
#!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
sub_p_res = subprocess.run(['git', 'clone', 'https://github.com/MSFTserver/pytorch3d-lite.git'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sys.path.append('./pytorch3d-lite')
import pathlib, shutil
from os.path import exists as path_exists
import wget
  
root_path = f'.'
if(not is_drive):
  model_path = root_path
else:
  if not path_exists("/content/drive/MyDrive/MindsEye/"):
    os.makedirs("/content/drive/MyDrive/MindsEye/models")
  model_path = f'/content/drive/MyDrive/MindsEye/models'

pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

if not (path_exists(f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt')):
  wget.download("https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt", model_path)
if not (path_exists(f'{model_path}/secondary_model_imagenet_2.pth')):
  wget.download("https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth", model_path)
if not (path_exists(f'{model_path}/AdaBins_nyu.pt')):
  wget.download("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", model_path)
if not (path_exists(f'{model_path}/vqgan_imagenet_f16_16384.ckpt')):
  sub_p_res = subprocess.run(['curl', '-L', '-o', '', "'{model_path}/vqgan_imagenet_f16_16384.yaml'", '-C', '-', "'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'", '#ImageNet', '16384'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
  print(sub_p_res) #<cc-cm>
  sub_p_res = subprocess.run(['curl', '-L', '-o', '', "'{model_path}/vqgan_imagenet_f16_16384.ckpt'", '-C', '-', "'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'", '#ImageNet', '16384'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
  print(sub_p_res) #<cc-cm>
if not (path_exists(f'{model_path}/diffusion.pt')):
  sub_p_res = subprocess.run(['wget', '-c', '-O', "'{model_path}/diffusion.pt'", "'https://dall-3.com/models/glid-3-xl/diffusion.pt'", ''], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
  print(sub_p_res) #<cc-cm>
if not (path_exists(f'{model_path}/finetune.pt')):
  sub_p_res = subprocess.run(['wget', '-c', 'https://dall-3.com/models/glid-3-xl/finetune.pt', '-O', "'{model_path}/finetune.pt'"], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
  print(sub_p_res) #<cc-cm>
if not (path_exists(f'{model_path}/bert.pt')):
  sub_p_res = subprocess.run(['wget', '-c', 'https://dall-3.com/models/glid-3-xl/bert.pt', '-O', "'{model_path}/bert.pt'"], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
  print(sub_p_res) #<cc-cm>
if not (path_exists(f'{model_path}/kl-f8.pt')):  
  sub_p_res = subprocess.run(['wget', '-c', 'https://dall-3.com/models/glid-3-xl/kl-f8.pt', '-O', "'{model_path}/kl-f8.pt'"], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
  print(sub_p_res) #<cc-cm>
  
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/CompVis/taming-transformers.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/openai/CLIP.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/crowsonkb/guided-diffusion.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/assafshocher/ResizeRight.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/isl-org/MiDaS.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
if not path_exists(f'{root_path}/MiDaS/midas_utils.py'):
  os.rename("MiDaS/utils.py", "MiDaS/midas_utils.py")
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/CompVis/latent-diffusion.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/shariqfarooq123/AdaBins.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/alembics/disco-diffusion.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/Jack000/glid-3-xl"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
if not path_exists(f'{root_path}/glid-3-xl/jack_guided_diffusion'):
  os.rename('glid-3-xl/guided_diffusion', 'glid-3-xl/jack_guided_diffusion')
sub_p_res = subprocess.run(['git', 'clone', '"https://github.com/multimodalart/mindseye.git"'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['mkdir', '.streamlit'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
shutil.copyfile("mindseye/.streamlit/config.toml", ".streamlit/config.toml")
shutil.copyfile("mindseye/app.py", "app.py")
shutil.copyfile("mindseye/disco_streamlit_run.py", "disco_streamlit_run.py")
shutil.copyfile("mindseye/hypertron_streamlit_run.py","hypertron_streamlit_run.py")
shutil.copyfile("mindseye/latent_streamlit_run.py", "latent_streamlit_run.py")
shutil.copyfile("mindseye/streamlit_nested_expanders.py", "streamlit_nested_expanders.py")
if not path_exists(f'{root_path}/disco_xform_utils.py'):
  shutil.copyfile("disco-diffusion/disco_xform_utils.py", "disco_xform_utils.py")

#sys.path.append('./mindseye')
sys.path.append('./guided-diffusion')
sys.path.append('./latent-diffusion')
sys.path.append(".")
sys.path.append('./taming-transformers')
sys.path.append('./disco-diffusion')
sys.path.append('./AdaBins')

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "CHtULqEM6mHY"
# !! }}
#@title 1.4 Install LocalTunnel
#Hey that is a dependency too! I know
sub_p_res = subprocess.run(['!npm', 'install', '-g', 'localtunne'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>


# %%
# !! {"metadata":{
# !!   "id": "j7_oaqKSFsDY"
# !! }}
"""
#### Do the run. (Click the link on `your url is:` link to access the tool!)
![Where to click](https://i.imgur.com/4gADD4s.png)

"""

# %%
# !! {"metadata":{
# !!   "id": "HZQ5nkgx6qh_",
# !!   "cellView": "form"
# !! }}
#@title Run the application!
sub_p_res = subprocess.run(['streamlit', 'run', 'app.py', '&', 'npx', 'localtunnel', '--port', '850'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
