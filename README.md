# Lung nodule bounding box detection and segmentation

Use LIDC dataset to improve the segmentation accuracy (dice score) for lung nodule segmentation

Stretch goal: perform malignancy classification of the nodules


### Dataset
LIDC (Lung Imaging Data Consortium) and Image database resource initiative (IDRI) stores images as standard DICOM objects which are annotated lung CT scans of 1000 patients. Refer https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14445

The nodules >= 3 mm were annotated by experts and stored as DICOM segmentation objects. Qualitative and quantitative attributes are also stored.
The annotations were pre-processed using clustering techniques to result in a final 2651 distinct nodules among 875 patients. Data is here, 124 Gb in size, has DICOM and XML.
We will however access the same data from IDC through GCP.


### Working on a GCP VM instance

Note: if you are using IDC credits, you will receive a project code from IDC that you will need to use in your notebook when downloading data from Google cloud storage.

Use the GCP market place to deploy a deep learning notebook.

Create a VM instance on GCP, use the following settings:

4 vCPUs, 26 GB memory

Ensure the VM instance has the following settings:
Allow http and https traffic

Make sure you have a firewall rule created for your ‘network’ of the VM instance


Install gcloud SDK on your local machine / Mac 
ssh into the VM instance on GCP 
Run gcloud auth login

Then, gcloud compute ssh --zone "<zone>" "<VM name>" --project "<project name assigned by IDC>"   
  
 
Check out the code from the repo on the VM instance
git clone <this repo>

We will be using NVIDIA GPUs so the docker image should be able to recognize the underlying GPU hardware and talk to it via the necessary drivers. 
 
There are CUDA toolkits and cuDNN libraries that need to be installed. (This comes with the deep learning VM we deployed from GCP marketplace)
 
However if we use a ready image from NVIDIA NGC, it will be a lot easier.
docker pull nvcr.io/nvidia/tensorflow:21.08-tf2-py3
Check the docker version you have in your VM instance, if its newer tan 19.03 we have a set of instructions to start the docker images for NVIDIA GPU compatibility

docker run --gpus all -it --rm -p 8889:8889 -p 6006:6006 -v ~/capstone/lungct_segmentation:/app nvcr.io/nvidia/tensorflow:21.08-tf2-py3

Using pytorch

docker run --gpus all -it --rm --ipc=host -p 8889:8889 -p 6006:6006 -v ~/capstone/lungct_segmentation:/app nvcr.io/nvidia/pytorch:21.02-py3
  
On your local machine port forward the VM instance so that you can access the jupyter lab in you local machine browser
  
gcloud compute ssh <vm name> --project <idc assigned project name> --zone <zone> -- -NL 8889:localhost:8889

In the container launch the jupyter notebook
jupyter lab --ip 0.0.0.0 --port 8889 --allow-root --notebook-dir=/app

  
The project setting in the notebook is important, make sure this is your project in GCP and you have access to it as a developer and tester. 

When trying to access the Big Query tables and the cloud storage (such as when trying to download the DICOM images using the manifest files using the gsutil tool) in GCP, you will need your authentication done in the notebook. 

But in your jupyter notebook running in a GCP VM instance, it is not that simple.
You need to set up access to the project for yourself as a tester (add your email address) in the OAuth consent screen. You will also have to set up access to a desktop application in the credentials screen.

Go to APIs and services for your project and create credentials for a desktop application.
You're using a remote jupyter notebook (it's running on a GCP VM instance and you are accessing it remotely via your local browser).
  
For setting up access for gsutil, use ‘gsutil config’ on the shell of the docker container or launch a terminal in jupyter lab. You will be asked to authenticate and then supply the token from the authentication.

When using multi-GPU training, ensure that you have set the shared docker memory to a higher value such as 256m.
Moreover, the code you should use to train using multi-GPU should use the distributed strategy.
  
  
For training in pytorch using DataParallel when using a docker container set the --ipc=host argument so that you have enough shared memory





