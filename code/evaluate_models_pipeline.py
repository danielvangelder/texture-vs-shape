import os
import sys
from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
from collections import OrderedDict
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils import model_zoo


def get_model(model_name, stylized=False):
    """Loads the pretrained model for the given model name. If stylized is set to true 
    """
    # alexnet 
    # vgg_16 
    # googlenet 
    # resnet_50 
    model = None
    model_name = str.lower(model_name)

    if not stylized:
        if model_name == "alexnet":
            model = models.alexnet(pretrained=True, progress=True)
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=True, progress=True)
        elif model_name == "googlenet":
            model = models.googlenet(pretrained=True, progress=True)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True, progress=True)
        else:
            raise NotImplementedError(f"No implemented model for name {model_name}")
    if stylized:
        model = load_stylized_model(model_name)
    return model


def load_stylized_model(model_name):
    """Taken from models/load_pretrained_models.py
    """

    model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
    }
    device = torch.device('cpu')

    if "resnet50" in model_name:
        print("Using the ResNet50 architecture.")
        assert model_name in model_urls, "You have not specified the resnet50 name correctly. Choose one of the following: resnet50_trained_on_SIN, resnet50_trained_on_SIN_and_IN, resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN"
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model)
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=device)
    elif "vgg16" in model_name:
        print("Using the VGG-16 architecture.")
       
        # download model from URL manually and save to desired location
        filepath = "./vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

        assert os.path.exists(filepath), "Please download the VGG model yourself from the following link and save it locally: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be downloaded automatically like the other models)"

        model = torchvision.models.vgg16(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        checkpoint = torch.load(filepath, map_location=device)


    elif "alexnet" in model_name:
        print("Using the AlexNet architecture.")
        model_name = "alexnet_trained_on_SIN" 
        model = torchvision.models.alexnet(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=device)
    else:
        raise ValueError("unknown model architecture.")

    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_test_set(path_to_test_set):
    """ Loads the image test set using a ImageFolder dataloader from pytorch. Returns 
    a dictionary with the class labels and the test set with batch size 1 and a list
    of the image urls
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_set = datasets.ImageFolder(path_to_test_set, transform=transform)
    test = DataLoader(test_set, batch_size=1, shuffle=False) # Load in batches of size 1
    class_labels = {v: k for k, v in test_set.class_to_idx.items()}
    return class_labels, test, test_set.imgs

def run_model_on_test_set(model, test_set, class_labels, imgs):
    """Runs the given model on the given test set and returns a list of tuples of: (predicted label, actual label, image url)
    """
    results = []
    for i, data in tqdm(enumerate(test_set), total=len(test_set)):
        images, labels = data
        # get softmax output
        softmax_output = torch.softmax(model(images),1) # replace with your favourite CNN
        # convert to numpy
        softmax_output_numpy = softmax_output.detach().numpy().flatten() # replace with conversion
        # create mapping
        mapping = ImageNetProbabilitiesTo16ClassesMapping()
        # obtain decision 
        prediction = mapping.probabilities_to_decision(softmax_output_numpy)
        img_url = imgs[i][0]
        actual = class_labels[labels.item()]
        results.append((prediction, actual, img_url))
    return results

def output_results_to_file(model_name, results, output_file_path, session = 1):
    # subj,session,trial,rt,object_response,category,condition,imagename
    # alexnet,1,1,NaN,bicycle,airplane,0,0001_s5n_dnn_0_airplane_00_airplane1-bicycle2.png
    # alexnet,1,29,NaN,airplane,airplane,0,0029_s5n_dnn_0_airplane_00_airplane3-oven1.png

    data = [[model_name, session, i+1, 0, prediction, actual, i+1, img_url] for i, (prediction, actual, img_url) in enumerate(results)]
    df = pd.DataFrame(data, columns=['subj','session','trial','rt','object_response','category','condition','imagename'])
    df.to_csv(output_file_path,sep=',',header=True,index=False)


def main(model_name, stylized, path_to_test_set, output_file_path):
    now = datetime.now() 
    log_name = now.strftime("Log_run_%Y-%m-%d-%H-%M")
    logging.basicConfig(filename=log_name, encoding='utf-8', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Starting test set evaluation for model: {model_name}")
    logging.info(f"This log file will be saved at: {log_name}")
    # if stylized: # Not necessary anymore as the model is mapped and loaded to cpu
        # logging.info("""Running the pretrained stylized models requires torch to be compiled with CUDA... 
        # If this does not work for you make sure to install the recommended version with cuda support:
        # pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html""")
    logging.info(f"Loading model with pretrained parameters. Is trained on stylized dataset?: {stylized}")
    model = get_model(model_name, stylized)
    logging.info(f"Loading test set on path {path_to_test_set}")
    class_labels, test_set, imgs = load_test_set(path_to_test_set)
    logging.info(f"Running model on test set")
    results = run_model_on_test_set(model, test_set, class_labels, imgs)
    logging.info(f"Writing results to output file: {output_file_path}")
    output_results_to_file(model_name, results, output_file_path)
    logging.info(f"Evaluation complete! Results can be viewed at {output_file_path}")
    return




if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4 or len(args) > 5:
        raise RuntimeError("""Specified an invalid ammount of arguments. Pass either 4 or 5 arguments:
        "python evaluate_models_pipeline.py [model name] [path to test set] [path for output file] [-s]"
        here "-s" can be added to load the model trained on the stylized imagenet dataset...
        """)
        exit(1)
    model = args[1]
    path_to_test_set = args[2]
    output_file_path = args[3]
    stylized = False
    if len(args) == 5 and str.lower(args[4]) == "-s":
        stylized = True
    main(model, stylized, path_to_test_set, output_file_path)
    exit(0)