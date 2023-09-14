from AutoEncoderModels.ConvolutionalModels import *
import torchvision.transforms as trans
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def train(modelType : type, trainImageFolder : str, valImageFolder : str,name : str = "model", pretrained = None, startEpoch = 0, epochs = 20, startLr = 0.1, batch_size=48, transforms = None, normalization = True):
    model = modelType(normalization= normalization)
    numparams = sum(map(lambda a: a.numel(), model.parameters()))
    print(f"Number of params : {numparams}")
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    if transforms is None:
        transforms = trans.Compose([
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
        ])
    dataset = ImageFolder(root=trainImageFolder, transform=transforms)
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    datasetVal = ImageFolder(root=valImageFolder, transform=transforms)
    dataLoaderVal = DataLoader(dataset=datasetVal, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, model.parameters()), 
        lr=startLr,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=1, verbose=True)
    criterion = torch.nn.MSELoss()

    trainModel(model, dataloader=dataLoader, dataLoaderVal=dataLoaderVal, criterion=criterion, scheduler=scheduler,epochs=epochs, optimizer=optimizer, name=name, startEpoch=startEpoch)

def test(modelType : type, imagesPath:str, path : str, normalization = True):
    model = modelType(normalization = normalization)
    numparams = sum(map(lambda a: a.numel(), model.parameters()))
    print(f"Number of params : {numparams}")
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(path))
    transforms = trans.Compose([
        trans.ToTensor()
    ])
    dataset = ImageFolder(root=imagesPath, transform=transforms)
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)
    criterion = torch.nn.MSELoss()
    testModelpPrintEncoded(model, dataLoader, criterion)

def testMultiple(modelType : type, imagesPath : str, paths : list[str], shape=(3, 128,128), save = False, saveName = "", normalization = True):
    models = []
    for path in paths:
        model = modelType(normalization = normalization)
        numparams = sum(map(lambda a: a.numel(), model.parameters()))
        print(f"Number of params : {numparams} : model {path}")
        model.eval()
        model.cuda()
        model.load_state_dict(torch.load(path))
        transforms = trans.Compose([
            trans.Resize((shape[1], shape[2])),
            trans.ToTensor()
        ])
        models.append(model)
    dataset = ImageFolder(root=imagesPath, transform=transforms)
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)
    testModelMultiplePrintEncoded(models, dataLoader, shape=shape, save = save, saveName=saveName)

def testMultipleModels(modelTypes : list[type], normalizations : list[bool], imagesPath : str, paths : list[str], shape=(3, 128,128), save = False, saveName = ""):
    models = []
    for i in range(len(modelTypes)):
        model = modelTypes[i](normalization = normalizations[i])
        numparams = sum(map(lambda a: a.numel(), model.parameters()))
        print(f"Number of params : {numparams} : model {paths[i]}")
        model.eval()
        model.cuda()
        model.load_state_dict(torch.load(paths[i]))
        transforms = trans.Compose([
            trans.ToTensor()
        ])
        models.append(model)
    dataset = ImageFolder(root=imagesPath, transform=transforms)
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)
    testModelMultiple(models, dataLoader, shape=shape, save = save, saveName=saveName)

if __name__ == "__main__":
    # transforms = trans.Compose([
    #         trans.RandomCrop((128,128), pad_if_needed=True),
    #         trans.RandomVerticalFlip(),
    #         trans.RandomHorizontalFlip(),
    #         trans.ToTensor(),
    #     ])
    # train(modelType=AutoEncoderDecoderS,
    #     trainImageFolder="D:\\internship2023\\traindata",
    #     valImageFolder="D:\\internship2023\\valdata",
    #     name="3S-NF",
    #     transforms=transforms,
    #     startLr=0.1, startEpoch=0, epochs = 512, batch_size=128, normalization=False)
    
    # test(AutoEncoderDecoderXS, imagesPath="C:\\Users\\best_intern\\Documents\\imagesColorVal", path=".\\models\\1XS-15.mld")
    # testMultipleModels([AutoEncoderDecoder, AutoEncoderDecoderM],
    #                    [True, True],
    #                     imagesPath="C:\\Users\\best_intern\\Documents\\imagesColorVal",
    #                     paths=[".\\models\\1A-1.mld",".\\models\\1M-1.mld"])
    testMultiple(AutoEncoderDecoderS,imagesPath="C:\\Users\\best_intern\\Documents\\imagesColorVal", paths=[
        ".\\models\\3S-NF-30.mld",
        ".\\models\\3S-NF-40.mld",
        ".\\models\\3S-NF-60.mld",
        ".\\models\\3S-NF-80.mld",
        ".\\models\\3S-NF-110.mld",
        ".\\models\\3S-NF-120.mld",
        ], save = False, saveName="XS", normalization = False)
    print("Done")