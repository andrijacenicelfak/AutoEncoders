import json
import time
import cv2
import os
import multiprocessing
from queue import Empty
path = 'D:\\internship2023\\annotations\\instances_train2017.json'
imagesPath = "D:\\internship2023\\train2017\\"
savePath = "D:\\internship2023\\imageClassTrain"


def cutImageAndSaveToFile(imageId : int, box : list[int], category_id : int):
    if box[2] < 32 or box[3] < 32:
        return
    fullId = str(imageId).zfill(12)
    filePath = savePath +"\\" +str(category_id) + "\\"+  fullId + str(category_id) + "b"
    for i in box:
        filePath = filePath + str(int(i)) + "_"
    filePath = filePath + ".jpg"
    loadPath = imagesPath + fullId + ".jpg"
    img = cv2.imread(loadPath)
    box = list(map(lambda x : int(x), box))
    cutImg = img[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
    cutImg = cv2.resize(cutImg, (128, 128))
    cv2.imwrite(filePath, cutImg)
    return
DONE = False
def worker_funciton(queue : multiprocessing.Queue, index):
    print(f"Worker {index} start!")
    while True:
            try:
                data = queue.get(timeout=1)
                if type(data) is not tuple:
                    break
                cutImageAndSaveToFile(data[0], data[1], data[2])
            except Empty:
                continue
    print(f"Worker {index} done!")
    return

def factory_function(queue, num_workers, index):
    print(f"Factory {index} start!")
    anns = None
    with open(path) as f:
        anns = json.load(f)
    annsMap = map(lambda x: x, anns["annotations"])
    for i, ann in enumerate(annsMap):
        if i % 100 == 0:
            print(i)
        queue.put((ann["image_id"] , ann['bbox'], ann["category_id"]))
    DONE = True
    for i in range(num_workers):
        queue.put("DONE")
    print(f"Factory {index} done!")
    return

def multi(num_proc : int = 4):
    queue = multiprocessing.Queue()

    factory = multiprocessing.Process(target=factory_function, args=(queue, num_proc, 0))
    factory.start()

    workers = []
    for i in range(num_proc):
        worker = multiprocessing.Process(target=worker_funciton, args=(queue, i))
        worker.start()
        workers.append(worker)
    
    factory.join()
    for w in workers:
        w.join()
    print("Done")
    
def annToParams(ann):
    cutImageAndSaveToFile(imageId=ann["image_id"] ,category_id=ann["category_id"], box=ann['bbox'])

def createFolders():
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    for i in range(90):
        p = savePath + "\\" + str(i+1)
        if not os.path.exists(p):
            os.mkdir(p)
def boxes():
    anns = None
    with open(path) as f:
        anns = json.load(f)
    annsMap = map(lambda x: x, anns["annotations"])

    for ann in annsMap:
        annToParams(ann)

if __name__ == "__main__":
    createFolders()
    multi(8)
    # boxes()


# numProcesses = 4
# pool = multiprocessing.Pool(processes=numProcesses)
# results = pool.imap(annToParams, annsMap)
# pool.close()
# pool.join()


# for ann in annsInter:
#     cutImage(imageId=ann["image_id"] ,category_id=ann["category_id"], box=ann['bbox'])
    # print(ann['bbox'], " : ", ann['category_id'])

# for i in range(90):
#     os.mkdir(savePath + str(i+1))