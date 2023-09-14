from ImageAnalyzation import *
import cv2
from scipy.spatial.distance import cosine
import numpy as np
import time
from FileExplorer import FileExplorer
import os
import json
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
from ImageAutocoder import *

def generateVector():
    ai = ImageAnalyzation("yolov8s.pt", device="cuda")

    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\source\\repos\\ImageClassification\\YOLOv8\\imgs")
    paths = fe.search()
    data = []
    for img in paths:
        data.append(ai.getFeatureVector(cv2.imread(img)).features)
    vector = np.zeros((230400))
    for data1 in zip(data, paths):
        for data2 in zip(data,paths):
            d1 = data1[0]
            print(len(d1))
            d2 = data2[0]
            vector = vector + np.absolute(np.subtract(d1, d2))
            print(os.path.basename(data1[1]), os.path.basename(data2[1]), ":\n", cosine(d1, d2)*1000)
            print("-------------------------------------------")
    vlen = []
    m = (np.min(vector) + np.max(vector))/2
    print(m)
    for i in range(len(vector)):
        if vector[i] > m:
            vlen.append(i)
    with open("vector.json", "w") as f:
        json.dump(vlen, fp=f)

def objectComparisonTest():
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\source\\repos\\ImageClassification\\YOLOv8\\imgs")
    paths = fe.search()

    ai = ImageAnalyzation("yolov8s.pt", device="cuda")
    data: list[ImageData] = []
    for p in paths:
        data.append(ai.getImageData(cv2.imread(p), getClassesData=True, getImageFeatures= False, getObjectsFeatures=False, shouldReturnOriginalImage=True))

    for d in data:
        for cdata in d.classes:
            # if cdata.className.lower() != "dog":
            #     break
            img = d.orgImage[cdata.boundingBox.y1 : cdata.boundingBox.y2, cdata.boundingBox.x1 : cdata.boundingBox.x2]
            for md in data:
                if md == d:
                    break
                for mcdata in md.classes:
                    if mcdata.className != cdata.className:
                        break
                    img2 = md.orgImage[mcdata.boundingBox.y1 : mcdata.boundingBox.y2, mcdata.boundingBox.x1 : mcdata.boundingBox.x2]
                    dist = ai.compareImageClassificationData(cdata, mcdata, img, img2)
                    print(dist)
                    # cv2.putText(img2, "dist : " + str(dist), (5, 20), 1, 1, (255,0,0), 1, cv2.LINE_AA)
                    cv2.imshow("org", img)
                    cv2.imshow("f2", img2)
                    key = 1
                    while key != ord(' ') and key != 27:
                        key = cv2.waitKey(0)
                        time.sleep(0.4)
                    cv2.destroyAllWindows()
                    if key == 27:
                        break

def loadAndShowImageResize(path, size, windowName):
    img = cv2.imread(path)
    showImageResize(img, size, windowName)

def showImageResize(img, size, windowName):
    imgr = cv2.resize(img, size)
    cv2.imshow(windowName, imgr)
    
def comparisonTest():
    ia = ImageAnalyzation("yolov8s", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\models\\ConvModelColor4R5C-28.model")
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Downloads\\val2017")
    fe.getChanges()
    paths = fe.getLastSearch()
    dataDict = dict()
    for p1 in paths:
        if p1 not in dataDict:
            img1 = cv2.imread(p1)
            # imgdatatime = time.time()
            dataDict[p1] = ia.getImageData(img1, imageFeatures=True, objectsFeatures=True, returnOriginalImage=True, wholeVector=False)
            # print(f"img data time : {time.time() - imgdatatime}")
        data1 = dataDict[p1]
        arrayOfPaths = []
        for p2 in paths:
            if p2 == p1:
                continue
            if p2 not in dataDict:
                img2 = cv2.imread(p2)
                # imgdatatime = time.time()
                dataDict[p2] = ia.getImageData(img2, imageFeatures=True, objectsFeatures=True, returnOriginalImage=True, wholeVector=False)
                # print(f"img data time : {time.time() - imgdatatime}")
            data2 = dataDict[p2]
            # cv2.imshow("img1", img1c)
            # cv2.imshow("img2", img2c)
            comp = ia.compareImages(imgData1=data1, imgData2=data2, compareWholeImages=True)
            arrayOfPaths.append((comp, p2))
        
        arrayOfPaths.sort(reverse=True, key=lambda x: x[0])
        print(arrayOfPaths[0][0])
        for cl in data1.classes:
            print(cl.className)
        print("-----------------------")
        data2 = dataDict[arrayOfPaths[0][1]]
        for cl in data2.classes:
            print(cl.className)
        print("-----------------------")

        loadAndShowImageResize(p1, (600, 600), "org")
        loadAndShowImageResize(arrayOfPaths[0][1], (600, 600), "similar")
        # cv2.imshow("similar", cv2.imread(arrayOfPaths[0][1]))
        # cv2.imshow("org", cv2.imread(p1))
        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.3)
            
    cv2.destroyAllWindows()
    return
def vectorGenerationTest():
    ia = ImageAnalyzation("yolov8s.pt", "cuda")
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Downloads\\test2017\\test2017")
    paths = fe.search()
    ia.generateVector(paths)
    return

def autocoderDatasetTrain(modelToLoad : str = None, startEpoch = 0):
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    dataset = ImageFolder(root="C:\\Users\\best_intern\\Documents\\imagesColor", transform=t)
    datasetvalidate = ImageFolder(root="C:\\Users\\best_intern\\Documents\\imagesColorVal", transform=t)
    batchSize = 72
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=batchSize)
    dataLoaderValidate = DataLoader(dataset=datasetvalidate, shuffle=True, batch_size=batchSize)
    model = ImageAutoencoderConvColor4R5C()
    if modelToLoad is not None :
        model.load_state_dict(torch.load(modelToLoad))
    model.cuda()
    epochs = 35
    runningLoss = 0
    lastLoss = 0
    lossFunction = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.25, patience=1)
    for epoch in range(startEpoch, epochs):
        etime = time.time()
        model.train()
        ptime = time.time()
        runningLoss = 0.0
        for i, (img, labels) in enumerate(dataLoader):
            img = img.cuda()
            rec = model(img)
            loss = lossFunction(rec, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            if i % 25 == 0:
                print('[%d, %6d] loss: %2.5f diff : %s %2.5f' % (epoch + 1, (i + 1) * batchSize, runningLoss / 25,'+' if runningLoss > lastLoss else '-',abs((runningLoss - lastLoss)/25)))
                print(f"ptime : {time.time()-ptime}s")
                lastLoss = runningLoss
                runningLoss = 0.0
                ptime = time.time()
        model.eval()
        with torch.no_grad():
            runningLoss = 0
            for i, (img, labels) in enumerate(dataLoaderValidate):
                img = img.cuda()
                rec = model(img)
                loss = lossFunction(img, rec)
                runningLoss += loss.item()
        runningLoss /= (i * batchSize)
        print("Validate loss : %2.4f \ntime %3.2f" % (runningLoss, time.time()-etime))
        scheduler.step(metrics=runningLoss)
        torch.save(model.state_dict(), ".\\models\ImageAutoencoderConvColor4R5C\\FB-"+ str(epoch) + ".model")
    return

def loadAutocoderTest(modelToLoad : str):
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Documents\\imagesColor")
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    model = ImageAutoencoderConvColor4R5C()
    model.eval()
    model.load_state_dict(torch.load(modelToLoad))
    paths = fe.search()
    for i in range(100):
        p = fe.randomImage() 
        img  = Image.open(p)
        imgt = t(img)
        rez = model(imgt)

        org = imgt.detach().numpy()
        decImg = rez.detach().numpy()

        org = np.array(np.squeeze((org * 0.5 + 0.5))*255, dtype="uint8").transpose(1, 2, 0)
        decImg = np.array(np.squeeze(decImg *0.5 + 0.5)*255, dtype="uint8").transpose(1, 2, 0)
        
        rso = cv2.cvtColor(cv2.resize(org, (512, 512)), cv2.COLOR_BGR2RGB)
        rsd = cv2.cvtColor(cv2.resize(decImg, (512, 512)), cv2.COLOR_BGR2RGB)
        # print(org,"\n", org.shape, "\n" ,decImg, "\n" ,decImg.shape, "\n------------------------------\n")
        cv2.imshow("org", rso)
        cv2.imshow("rec", rsd)
        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.4)    
    return

def loadAutocoderTestMultipleModels(folderPath:str, modelName:str, extension:str, numOfModels : int, startModel : int = 0, modelType : type = ImageAutoencoderConv4R5C):
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Documents\\imagesColorVal\\")
    fe.search()
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5)),
    ])
    models = []
    for i in range(startModel, numOfModels):
        model = modelType()
        model.eval()
        model.load_state_dict(torch.load(folderPath + modelName +"-"+ str(i) + extension))
        models.append((model, i))
    lossFunction = torch.nn.MSELoss()
    for i in range(100):
        p = fe.randomImage() 
        img  = Image.open(p)
        imgt = t(img).view((-1,3,128,128))
        print(imgt.shape)
        rez = []
        for model in models:
            rez.append((model[0](imgt), model[1]))

        org = imgt.detach().numpy()
        decImg = []
        for r in rez:
            loss = lossFunction(imgt, r[0])
            cimg = r[0].detach().numpy()
            cimg = np.array(np.squeeze(cimg *0.5 + 0.5)*255, dtype="uint8").transpose(1, 2, 0)
            cimg = cv2.cvtColor(cv2.resize(cimg, (256, 256)), cv2.COLOR_BGR2RGB)
            decImg.append((cimg, r[1], loss))

        org = np.array(np.squeeze((org * 0.5 + 0.5))*255, dtype="uint8").transpose(1, 2, 0)
        
        rso = cv2.cvtColor(cv2.resize(org, (256, 256)), cv2.COLOR_BGR2RGB)
        # print(org,"\n", org.shape, "\n" ,decImg, "\n" ,decImg.shape, "\n------------------------------\n")
        cv2.imshow("org", rso)
        for dec in decImg:
            cv2.imshow(str(dec[1]),  dec[0])
            print(str(dec[1]) + " : " + str(dec[2].item()),)
            
        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.4)    
    return

def compareImagesCD():
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\imgs")
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    model = ImageAutoencoderConvColor4R5C()
    model.eval()
    model.load_state_dict(torch.load("YOLOv8\\models\\ConvModelColor4R5C-28.model"))
    model.cuda()
    paths = fe.search()
    with torch.no_grad():
        for path1 in paths:
            path1 = fe.randomImage()
            orgImg1 = cv2.imread(path1)
            img1 = cv2.resize(orgImg1, (128,128))
            vec1 = model(t(img1).to("cuda")).cpu().numpy()
            cv2.imshow("1", orgImg1)
            maxSim = 0
            maxSimImg = None
            modelTime = 0
            numberOfCalls = 0
            for path2 in paths:
                if path1 == path2:
                    continue
                orgImg2 = cv2.imread(path2)     
                img2 = cv2.resize(orgImg2, (128, 128))
                img2 = t(img2).to("cuda")

                ct = time.time()
                
                vec2 = model((img2)).cpu().numpy()
                
                modelTime += time.time() - ct
                numberOfCalls += 1

                simm = 1 - cosine(vec1.flatten(), vec2.flatten())
                if simm > maxSim:
                    maxSim = simm
                    maxSimImg = orgImg2
                    print('%1.5f %1.5f' % (simm, modelTime / numberOfCalls))
                    # cv2.imshow("2", orgImg2)
                    # while cv2.waitKey(0) != ord(' '):
                    #     time.sleep(0.4) 
            print(maxSim)
            cv2.imshow("2", maxSimImg)
            while cv2.waitKey(0) != ord(' '):
                time.sleep(0.4)    

def trainAutoEncoder(modelToLoad : str = None, startEpoch = 0):
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    dataset = ImageFolder(root="C:\\Users\\best_intern\\Documents\\imagesColor", transform=t)
    batchSize = 32
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=batchSize)
    model = AutoEncoder()
    print(model)
    if modelToLoad is not None :
        model.load_state_dict(torch.load(modelToLoad))
    model.cuda()
    model.train()
    lossFunction = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 35
    runningLoss = 0
    lastLoss = 0
    for epoch in range(startEpoch, epochs):
        for i, (img, labels) in enumerate(dataLoader):
            img = img.cuda()
            optimizer.zero_grad()
            rec = model(img)
            loss = lossFunction(img, rec)
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.item()
            if i % 5 == 0:
                print('[%d, %6d] loss: %2.5f diff : %s %2.5f' % (epoch + 1, (i + 1) * batchSize, runningLoss / 5,'+' if runningLoss > lastLoss else '-',abs((runningLoss - lastLoss)/5)))
                lastLoss = runningLoss
                runningLoss = 0.0
        torch.save(model.state_dict(), "AE-"+ str(epoch) + ".model")

    return

def testMultipleImagesIA():
    ia = ImageAnalyzation(model="yolov8s", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\models\\ConvModelColor4R5C-28.model")
    fe = FileExplorer("C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\imgs")
    paths = fe.search()
    imgs = [cv2.imread(p) for p in paths]

    s = time.time()
    data = ia.getImageDataList(images=imgs, imageFeatures=True, objectsFeatures=True, returnOriginalImage=True)
    print(f'list time : {time.time()-s}')
    s = time.time()
    data2 = []
    for img in imgs:
        d = ia.getImageData(img, classesData=True, imageFeatures=True, objectsFeatures=True, returnOriginalImage=True)
        data2.append(d)
    print(f'one by one time : {time.time()-s}')
    
    for i in range(len(data)):
        if data[i] != data2[i]:
            print("NOT EQUAL")
            break
    return

def autocoderFullImageTest():
    ia = ImageAnalyzation(model="yolov8s", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\models\\ConvModelColor4R5C-28.model")
    fe = FileExplorer("C:\\Users\\best_intern\\Downloads\\val2017")
    paths = fe.search()
    data = []
    for i in range(100):
        path = fe.randomImage()
        data.append((ia.getFeatureVector(cv2.imread(path)), path))
    for i in range(len(data)):
        d1 = data[i]
        cdata = []
        for j in range(len(data)):
            d2 = data[j]

            dist = 1 - cosine(d1[0], d2[0])
            cdata.append((dist, d2[1]))
        cdata.sort(key=lambda x: x[0], reverse=True)

        print(f"Comparing image : {d1[1]} :")
        for cd in cdata:
            print(f"Image : {cd[1]} : {cd[0]}")

    return

def ACCTrain(modelToLoad : str = None, startEpoch = 0):
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    dataset = ImageFolder(root="C:\\Users\\best_intern\\Documents\\imagesColor", transform=t)
    datasetvalidate = ImageFolder(root="C:\\Users\\best_intern\\Documents\\imagesColorVal", transform=t)
    batchSize = 128
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=batchSize)
    dataLoader.__len__()
    dataLoaderValidate = DataLoader(dataset=datasetvalidate, shuffle=True, batch_size=batchSize)
    model = ACC()
    if modelToLoad is not None :
        model.load_state_dict(torch.load(modelToLoad))
    model.cuda()
    epochs = 35
    runningLoss = 0
    lastLoss = 0
    lossFunction = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.25, patience=1)
    for epoch in range(startEpoch, epochs):
        etime = time.time()
        model.train()
        ptime = time.time()
        runningLoss = 0.0
        for i, (img, labels) in enumerate(dataLoader):
            img = img.cuda()
            rec = model(img)
            loss = lossFunction(rec, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            if i % 25 == 0:
                print('[%d, %6d] loss: %2.5f diff : %s %2.5f' % (epoch + 1, (i + 1) * batchSize, runningLoss / 25,'+' if runningLoss > lastLoss else '-',abs((runningLoss - lastLoss)/25)))
                print(f"ptime : {time.time()-ptime}s")
                lastLoss = runningLoss
                runningLoss = 0.0
                ptime = time.time()
        model.eval()
        with torch.no_grad():
            runningLoss = 0
            for i, (img, labels) in enumerate(dataLoaderValidate):
                img = img.cuda()
                rec = model(img)
                loss = lossFunction(img, rec)
                runningLoss += loss.item()
        runningLoss /= (i * batchSize)
        print("Validate loss : %2.4f \ntime %3.2f" % (runningLoss, time.time()-etime))
        scheduler.step(metrics=runningLoss)
        torch.save(model.state_dict(), ".\\models\ACC\\A-"+ str(epoch) + ".model")
    return
if __name__ == "__main__":
    # torch.set_default_device(torch.device("cuda"))
    # autocoderDatasetTrain3R4C()
    # loadAutocoderTest("C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\bad\\ConvModelColor4R5C-10.model")
    # compareImagesCD()
    # autocoderDatasetTrain()
    # trainAutoEncoder()
    # testMultipleImagesIA()
    # ACCTrain(None, 0)
    loadAutocoderTestMultipleModels(".\\models\\ACC\\", "A", ".model", 10, 0, modelType=ACC)
    # autocoderFullImageTest()
    # comparisonTest()

    print("Done")
