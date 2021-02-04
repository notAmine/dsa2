"""

https://docs.bentoml.org/en/latest/quickstart.html




https://github.com/jjmachan/resnet-bentoml



"""

# iris_classifier.py
import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)




    else:
        model_state_dict, criterion, optimizer_ft, exp_lr_scheduler, epoch = utils.load_model(checkpoint)
        model_ft.load_state_dict(model_state_dict)
        model_ft = model_ft.to(device)
    try:
        model_ft = train_model(dataloaders, dataset_sizes, model_ft, criterion,
            optimizer_ft, exp_lr_scheduler, start_epoch=epoch, end_epoch=25)

        saveToBento = input("Save model to Bento ML (yes/no): ")

        if saveToBento.lower() == "yes":
            # Add it to BentoML
            bento_svc = AntOrBeeClassifier()
            bento_svc.pack('model', model_ft)

            # Save your Bento Service
            saved_path = bento_svc.save()
            print('Bento Service saved in ', saved_path)
    except KeyboardInterrupt:
        pass
    finally:




###########################################################################
from PIL import Image

import torch
from torchvision import transforms

import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.handlers import ImageHandler

classes = ['ant', 'bee']
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
cpu = torch.device('cpu')

@bentoml.env(pip_dependencies=['torch', 'torchvision'])
@bentoml.artifacts([PytorchModelArtifact('model')])
class AntOrBeeClassifier(bentoml.BentoService):

    @bentoml.api(ImageHandler)
    def predict(self, img):
        img = Image.fromarray(img)
        img = transform(img)

        self.artifacts.model.eval()
        outputs = self.artifacts.model(img.unsqueeze(0))
        _, idxs = outputs.topk(1)
        idx = idxs.squeeze().item()
        return classes[idx]









################# Save to Bento Space ###################################
import argparse

import torch.nn as nn
from torchvision import models

import utils
from classificationService import AntOrBeeClassifier

def saveToBento(checkpoint):
    model_state_dict, _, _, _, _ = utils.load_model(checkpoint)

    # Define the model
    model_ft =  models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # Load saved model
    model_ft.load_state_dict(model_state_dict)

    # Add model to Bento ML
    bento_svc = AntOrBeeClassifier()
    bento_svc.pack('model', model_ft)

    # Save Bento Service
    saved_path = bento_svc.save()
    print('Bento Service Saved in ', saved_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint to load the model')
    args = parser.parse_args()

    saveToBento(args.checkpoint)


