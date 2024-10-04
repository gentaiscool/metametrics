import subprocess
import os
import logging
import requests
from zipfile import ZipFile
import bz2
import setuptools
from setuptools import find_packages, setup
from setuptools.command.install import install

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define a setup command
class SetupInstallCommand(install):
    description = 'Run additional shell commands for setup'
    user_options = []

    def run(self):
        # Get original working directory
        owd = os.getcwd()

        # Navigate to the meta_metrics/metrics directory
        os.chdir('meta_metrics/metrics')
        
        # Download word embeddings for ROUGE-WE metric
        os.system("rm -rf embeddings")
        os.system("mkdir embeddings")
        if not os.path.exists(os.path.join(os.getcwd(), "embeddings/deps.words")):
            logging.info("Downloading word embeddings")
            url = "https://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
            r = requests.get(url)
            d = bz2.decompress(r.content)
            with open(os.path.join(os.getcwd(), "embeddings/deps.words"), "wb") as outputf:
                outputf.write(d)
            
        # Navigate to the bleurt directory
        os.chdir('bleurt')

        # Install BLEURT
        logging.info("Installing BLEURT ...")
        result = subprocess.run(["pip", "install", "."], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logging.info("BLEURT installed successfully.")
        else:
            logging.error("Failed to install BLEURT.")
            logging.error(result.stderr.decode())
            return
        
        # The paths for downloading and check if BLEURT-20 is already downloaded
        bleurt_zip_path = os.path.join(os.getcwd(), "BLEURT-20.zip")
        bleurt_model_path = os.path.join(os.getcwd(), "BLEURT-20")
        if not os.path.exists(bleurt_model_path):
            # Download BLEURT-20
            logging.info("BLEURT-20 not found. Downloading and extracting...")
            with requests.get("https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip", stream=True) as r:
                r.raise_for_status()
                with open(bleurt_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Unzip      
            with ZipFile(bleurt_zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.getcwd())
            os.remove(bleurt_zip_path)

        # reset working directory
        os.chdir(owd)
        
        # Navigate to the tasks/mteval
        os.chdir('tasks/mteval')
        os.system("rm -rf mt-metrics-eval")
        if not os.path.isdir('mt-metrics-eval'):
            logging.info("Cloning mt-metrics-eval ...")
            subprocess.run(["git", "clone", "https://github.com/google-research/mt-metrics-eval.git"])
        os.chdir("mt-metrics-eval")
        logging.info("Installing mt-metrics-eval ...")
        result = subprocess.run(["pip", "install", "."], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logging.info("mt-metrics-eval installed successfully.")
        else:
            logging.error("Failed to install  mt-metrics-eval .")
            logging.error(result.stderr.decode())
            return
        
        os.chdir(owd)
        
        # Run the standard install process
        install.run(self)
        
        os.chdir('meta_metrics/metrics')
        
        # Setup for ROUGE
        os.environ["ROUGE_HOME"] = os.path.join(os.getcwd(), "ROUGE-1.5.5")
        os.environ["LC_ALL"] = "C.UTF-8"
        os.environ["LANG"] = "C.UTF-8"

        # Install pyrouge
        os.system("pip install -U  git+https://github.com/bheinzerling/pyrouge.git")
        
        # Remove current ROUGE if exists; Tips if it errors out for PERL: sudo apt-get install libxml-parser-perl
        os.system("rm -rf ROUGE-1.5.5")
        subprocess.run(["curl", "-L", "https://github.com/Yale-LILY/SummEval/tarball/7e4330d", "-o", "project.tar.gz", "-s"])
        subprocess.run(["tar", "-xzf", "project.tar.gz"])
        subprocess.run(["mv", "Yale-LILY-SummEval-7e4330d/evaluation/summ_eval/ROUGE-1.5.5/", os.path.join(os.getcwd(), "ROUGE-1.5.5")])
        subprocess.run(["rm", "project.tar.gz"])
        subprocess.run(["rm", "-rf", "Yale-LILY-SummEval-7e4330d/"])
        
        # Setup for METEOR
        meteor_url = 'https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor-1.5.jar?raw=true'
        r = requests.get(meteor_url)
        with open(os.path.join(os.getcwd(), "meteor-1.5.jar"), "wb") as outputf:
            outputf.write(r.content)
        
        # Run the command to download the en_core_web_sm model for SummaQA; should be 3.7.1
        logging.info("Downloading 'en_core_web_sm' model.")
        result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        if result.returncode == 0:
            logging.info("Model 'en_core_web_sm' downloaded successfully.")
        else:
            logging.error(f"Failed to download 'en_core_web_sm': {result.stderr.decode()}")
            return
        
        logging.info("Downloading 'en_core_web_md' model.")
        result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"], check=True)
        if result.returncode == 0:
            logging.info("Model 'en_core_web_md' downloaded successfully.")
        else:
            logging.error(f"Failed to download 'en_core_web_md': {result.stderr.decode()}")
            return

        # Install submodule (GEMBA)
        os.system("git submodule update --init --recursive")
        os.system("git submodule update")
        

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metametrics",
    version="0.0.1",
    author="Genta Indra Winata",
    author_email="gentaindrawinata@gmail.com",
    description="MetaMetrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/meta-metrics/meta-metrics",
    project_urls={
        "Bug Tracker": "https://github.com/meta-metrics/meta-metrics/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "accelerate",
        "bitsandbytes",
        "black",
        "datasets",
        "deepspeed",
        "einops",
        "flake8>=6.0",
        "fschat",
        "huggingface_hub",
        "hf_transfer",
        "isort>=5.12.0",
        "peft",
        "pytest",
        "tabulate",  # dependency for markdown rendering in pandas
        "tokenizers",
        "tiktoken==0.6.0",  # added for llama 3
        "transformers==4.43.4",  # pinned at llama 3
        "trl>=0.8.2",  # fixed transformers import error, for DPO
        "wandb",  # for loading model path / reivisions from wandb
        "bayesian-optimization",
        "evaluate==0.4.2",
        "scipy",
        "transformers==4.43.4", # pinned at llama 3
        "sentencepiece",
        "sacrebleu==2.4.2",
        "unbabel-comet==2.2.2",
        "pandas",
        "numpy",
        "requests",
        "spacy",
        "tf_keras==2.16.0",
        "tensorflow==2.16.1",
        "torch==2.3.1",
        "torchvision",
        "tf-slim>=1.1",
        "bert_score",
        "nltk==3.8.1",
        "rouge_score==0.1.2",
        'regex',
        'six',
        "gdown",
        "huggingface-hub",
        "scikit-learn",
        "openai-clip",
        # GEMBA Requirements
        "openai>=1.0.0",
        "termcolor",
        "pexpect",
        "ipdb",
        "absl-py",
        "tqdm"
    ],
    packages = ['metametrics'],
    python_requires=">=3.10",
)
