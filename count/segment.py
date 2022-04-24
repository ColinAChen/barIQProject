import cv2
import numpy as np
#import matplotlib.pyplot as plt
import math
from pyzbar.pyzbar import decode
import os

import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred, {'databaseURL':'https://bariq-jia1356-default-rtdb.firebaseio.com/'})
import json
'''
try rough image segmentation based on flood fill

this might work because if a light is shined on the pallet flags, they should be close in color


can try to do first pass grouping with localized histogram?
look for peaks with similar intensities/colors?

'''
GREEN = (119, 171, 92)
TARGET = GREEN #(207, 227, 228)#(215, 232, 232)#(183, 212, 232)#(1,0,122)#(215, 232, 232)
THRESHOLD = 20
COUNT_THRESHOLD = 1000
PATH = 'gap4' 

def main():

    ## j is start
    ## want to see if the breakpoint is the same for non 255 starts
    #for j in range(0, 255, 40):
    #
    #    x = []
    #    y = []
    #    div = []
    #    for i in range(1,255):
    #        x.append(i)
    #        y.append(j/i)
    #        div.append(-j/(i*i))
    #    plt.figure(1)
    #    plt.subplot(211)
    #    plt.plot(x,y)
    #
    #    plt.figure(212)
    #    plt.plot(x,div)
    #    
    #plt.show()
    b,g,r = TARGET
    print(b/g, b/r, g/r)
    #sendToDB(None)
    
    for path in os.listdir(PATH):
        if 'green' in path or 'bq' in path or 'groups' in path:
            #print('skip: ', path)
            continue
        
        imagePath = os.path.join(PATH, path)
        #print(imagePath)
        #image = cv2.imread(imagePath)
        #green = segmentGreen(image)
        evalImage(imagePath)
        #saveName = path.split('.')[0] + '_green.png'
        #savePath = os.path.join(PATH, saveName)
        #cv2.imwrite(savePath, green)
        #green = cv2.resize(green, (500, 500))
        #showImage(green.astype(np.uint8))
        #detectRect(green)

    '''
Find barcodes and count tags
'''
def evalImage(path):
    image = cv2.imread(path)
    green = segmentGreen(image)
    #showGreen = cv2.resize(green, (500, 500))
    #showImage(showGreen.astype(np.uint8))
    #detectRect(green)
    # test sim color

    #g = cv2.imread(PATH, 0)
    #x = list(range(0,255))
    #y = []
    #for i in range(0,255):
    #    
    #    ret, thresh = cv2.threshold(g, i, 255, cv2.THRESH_BINARY)
    #    #cv2.imwrite('gray.png', thresh)
    #    # decode barcode info
    #    decodeList = decode(thresh)
    #    y.append(len(decodeList))
    #plt.plot(x,y)
    #plt.show()
    
    '''
    rows, cols, c = image.shape
    out = np.zeros(image.shape)
    for r in range(rows):
        if r%100 == 0:
            print('r:', r)
        for c in range(cols):
            #if distance(image[r][c], TARGET) < 1:
            #    print(image[r][c])
            #    print('r:', r)
            #    print('c:', c)
            p = simRatio(image[r][c], TARGET)
            #if(p[0] == 0 and p[1] == 0 and p[2] == 0):
            #    print('r: ',r, ' c:', c)
            if max(p) < 1:
                out[r][c] = p
            else:
                out[r][c] = (1,1,1)
    cv2.imwrite('simRatio.png', out*600)
    '''
    
    #g = cv2.imread('barcode.jpg', cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(path)

    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bcDict, bq = barcodeSearch(path)
    if len(bcDict.keys()) > 0:
        saveName = path.split('.')[0] + '_bq.png'
        cv2.imwrite(saveName, bq)
    '''
    ret, thresh = cv2.threshold(green.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY)

    showThresh = cv2.resize(thresh, (1000,1000))
    showImage(showThresh)
    cv2.imwrite('gray.png', thresh)
    # decode barcode info
    #decodeList = decode(thresh)
    decodeList = decode(thresh)

    print('num barcodes: ', len(decodeList))
    nameSet = parseBarcodeList(decodeList)
    
    for barcode in decodeList: 

        # Locate the barcode position in image
        (x, y, w, h) = barcode.rect
         
        # Put the rectangle in image using
        # cv2 to heighlight the barcode
        cv2.rectangle(color, (x-10, y-10),
                      (x + w+10, y + h+10),
                      (0, 0, 255), 7)
        #cv2.rectangle(thresh, (x-10, y-10),
        #          (x + w+10, y + h+10),
        #          0, 2)


        if barcode.data!="":
           
            # Print the barcode data
            print(barcode.data)
            print(barcode.type)
    
    print('brands detected: ', nameSet)
    cv2.imwrite('detect.png', color)
    image = cv2.resize(image, (100,100))
    img = cv2.imread('aisle1.jpeg', -1)
    '''

    #normalize(img)
    

    #segments = segment(image, target=TARGET)
    #print(segments)
    #for seg in segments:
    
    #mask = preprocess(image)
    #showImage(green)
    saveName = path.split('.')[0] + '_green.png'
    cv2.imwrite(saveName, green)
    mask = np.where(green>100, 255,0)
    #showImage(mask.astype(np.uint8))
    groups = group(mask.astype(np.uint8))
    cThresh = 200
    out = np.zeros(green.shape)
    tags = countGroups(groups,thresh=cThresh, out=out)
    #print('num tags:', tags)
    
    #showImage(out)
    saveName = path.split('.')[0] + '_groups.png'
    cv2.imwrite(saveName, out) 
    outDict = {}
    if (len(bcDict.keys()) == 0):
        outDict['none'] = tags
    else:
        split = len(bcDict.keys())
        for b in bcDict:
            outDict[b] = int(tags/split)

    #count = 0
    #out = np.zeros(mask.shape)
    
    
    '''
    #numFlags = count(image)
    
    out*=600
    out = np.where(out>255, 255, out)
    #cv2.imwrite('over.png', out)
    numFlags = count(out, target=(0,0,0))
    print('flags of color', TARGET, ': ', numFlags)

    '''
    print(outDict)
    return outDict

def count(image, target=None):
    mask = preprocess(image, target)
    groups = group(mask)
    
    count = 0
    out = np.zeros(mask.shape)
    for g in groups:
        # do more refined counting
        # if we can expect that mask will return squares we can use line detection or something to count the number of rectangles
        if len(groups[g]) > COUNT_THRESHOLD:
            count += 1
            for p in groups[g]:
                r,c = p
                out[r][c] = 255
    cv2.imwrite('count.png', out)
    print('flags counted: ', count)
    return count
def preprocess(image, target=None):
    if target is None:
        sub = abs(image - TARGET)
    else:
        print('target not None')
        sub = abs(image-target)
    
    cv2.imwrite('sub.png', sub)
    #dist = sub[:,:,0] + sub[:,:,1] + sub[:,:,2]
    dist = np.sum(sub, 2)
    dist = np.where(dist>255, 255, dist)
    print(dist.shape)
    print(np.amin(dist))
    cv2.imwrite('dist.png', dist)
    mask = np.where(dist < THRESHOLD, 255, 0)
    #mask = np.where(distance(image, TARGET) < THRESHOLD, image, (0,0,0))
    cv2.imwrite('mask.png', mask)
    return mask
'''
Flood fill for grayscale

'''
def group(image):
    rows, cols = image.shape
    visitedSet = set()
    group = 0
    groups = {}
    for row in range(rows):
        for col in range(cols):
            if image[row][col] == 255:
                # found a white pixel
                # this means it was close to the target color in the original image
                # do flood fill to create groups
                queue = [(row,col)]
                groups[group] = [(row,col)]
                while (len(queue) > 0):
                    p = queue.pop(0)
                    if p in visitedSet:
                        continue
                    # add to visited set and current group
                    visitedSet.add(p)
                    groups[group].append(p)

                    # add neighbors to the queue
                    n = getNeighbors(image, p)
                    queue.extend(n)
            group += 1
    return groups
def countGroups(groups, thresh=50, out=None):
    count = 0
    for g in groups:
        # do more refined counting
        # if we can expect that mask will return squares we can use line detection or something to count the number of rectangles
        if len(groups[g]) > thresh and checkSquare(groups[g]):
            #checkSquare(groups[g])
            count += 1
            if out is not None:
                for p in groups[g]:
                    r,c = p
                    out[r][c] = 255
    #cv2.imwrite('count.png', out)
    #print('flags counted: ', count)
    return count
'''
Flood fill for color
'''
def segment(image, target=(255,255,255), threshold=20):
    visitedSet = set()
    segments = {}
    print('shape:', image.shape)
    ds = cv2.resize(image, (100,100))

    cv2.imwrite('100x100.png', ds)

    rows, cols, channels = ds.shape

    segment = 0
    for row in range(rows):
        for col in range(cols):
            if (row, col) in visitedSet:
                continue
            #print(distance(ds[row][col], target))
            if distance(ds[row][col], target) < threshold:
                # flood fill/search
                segments[segment] = [(row, col)]
                queue = [(row, col)]
                while len(queue) > 0:
                    p = queue.pop(0)
                    # check for visited set
                    if p in visitedSet:
                        continue
                    
                    # add current to visited set and segment map
                    visitedSet.add(p)
                    r,c = p
                    segments[segment].append(p)

                    # add neighbors to the queue
                    # check colors here even though it's redundant
                    n = getNeighbors(image, p, target=target, threshold=threshold)
                    queue.extend(n)
                segment += 1
    return segments

def distance(c1, c2):
    return sum( [abs(x-y) for x,y in zip(c1,c2)])

'''
Identify similar colors under different illumination by looking at the relative ratios between color channels
'''
def simRatio(c1, c2, thresh=.1):
    #r1,b1,g1 = c1
    #r2,b2,g2 = c2

    # 10 loosely based off of when 255/x starts to change slowly, not a precise calculation
    # can consider some non linear scales?

    # maybe rescale from 0-255 to 10-255?

    c1N = [max(10, x) for x in c1]
    c2N = [max(10, x) for x in c2]
    c1N = np.array(c1N, dtype=np.uint8)
    c2N = np.array(c2N, dtype=np.uint8)
    # technically reversed because openCV reads as BGR but not important
    r1,b1,g1 = c1N
    r2,b2,g2 = c2N

    # shouldn't have any zeros by now
    
    # maybe 
    ratio1 = [r1/b1, r1/b1, b1/g1]
    ratio2 = [r2/b2, r2/b2, b2/g2]
    #print(ratio1)
    #print(ratio2)
    out = []
    for i in range(3):
        # don't do absolute value later?
        # won't matter when we make a distance call
        out.append( abs(ratio1[i] - ratio2[i]) )
    #return distance(ratio1, ratio2)
    '''
    if (c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2]):
        print(c1)
        print(c2)
        print(c1N)
        print(c2N)
        print(ratio1)
        print(ratio2)
        print(out)
        print('\n')
    '''
    return out
    
def getNeighbors(image, p, target=None, threshold=20, sim=False):
    r,c = p
    rows, cols = image.shape[:2]
    neighbors = [   (-1,-1), (-1, 0), (-1, 1),
                    ( 0,-1),          ( 0, 1),
                    ( 1,-1), ( 1, 0), ( 1, 1)]
    ret = []
    for n in neighbors:
        ar, ac = n
        # check row bounds
        if r + ar < 0 or r + ar > rows-1:
            continue
        # check col bounds
        if c + ac < 0 or c + ac > cols-1:
            continue
        # check color
        if target is not None and not sim and distance(image[r+ar][c+ac], target) > threshold:
            continue
        if target is not None and sim and simRatio(image[r+ar][c+ac], target):
            continue
        if target is None and image[r+ar][c+ac] == 0:
            continue
        ret.append((r+ar, c+ac))
    return ret

def normalize(img):
    
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    cv2.imwrite('shadows_out.png', result)
    cv2.imwrite('shadows_out_norm.png', result_norm)

def barcode(image, draw=False):
    
    pass


'''
Try multiple threshold values to find any barcodes in the image
'''
def barcodeSearch(path):
    color = cv2.imread(path)

    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(path)
    totalList = []
    totalSet = set()
    bcDict = {}
    for t in range(10, 240, 3):
        ret, thresh = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
        decodeList = decode(thresh)
        for barcode in decodeList:
            d = barcode.data
            r = barcode.rect
            if d in bcDict:
                bcDict[d].append(r)
            else:
                bcDict[d] = [r]
            #totalList.append(barcode.rect)
            #totalSet.add(barcode.
        #print('num barcodes: ', len(decodeList))
        #nameSet = parseBarcodeList(decodeList)
        #print('t: ',t, 'nameSet: ', nameSet)
        
        continue
        if (len(nameSet) == 0):
            continue
        
        for barcode in decodeList: 

            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect
             
            # Put the rectangle in image using
            # cv2 to heighlight the barcode
            cv2.rectangle(color, (x-10, y-10),
                          (x + w+10, y + h+10),
                          (0, 0, 255), 7)
            #cv2.rectangle(thresh, (x-10, y-10),
            #          (x + w+10, y + h+10),
            #          0, 2)


            if barcode.data!="":
               
                # Print the barcode data
                print(barcode.data)
                print(barcode.type)
        
        #print('brands detected: ', nameSet)
        #cv2.imwrite('detect.png', color)
        #image = cv2.resize(thresh, (100,100))
        #cv2.showImage(image)
    #print(bcDict)
    c = [(255,0,0), (0,0,255)]
    for i, bc in enumerate(bcDict):
        for rect in bcDict[bc]:
            (x, y, w, h) = rect
             
            # Put the rectangle in image using
            # cv2 to heighlight the barcode
            cv2.rectangle(color, (x-10, y-10),
                          (x + w+10, y + h+10),
                          c[i], 7)
    #cv2.imwrite('detect.png', color)
    return bcDict, color
def parseBarcodeList(bq):
    nameSet = set()
    for b in bq:
        #print(b.data)
        nameSet.add(b.data)
        #print(b)
    return nameSet
'''
Send data to the database

https://www.freecodecamp.org/news/how-to-get-started-with-firebase-using-python/
'''
def sendToDB(counts):
    ref = db.reference("/")
    test = {"Michelob Ultra": 4} 
    data = json.dumps(test)
    ref.set(data)    

'''
plant is likely where 2*b - g - r > 0

might work here with green tags
'''
def segmentGreen(image):
    # opencv image is bgr
    mask = (2*image[:,:,1]) - image[:,:,0] - image[:,:,2]
    
    #showMask = cv2.resize(mask, (500,500))
    #showImage(showMask)
    out = np.where(mask>0, mask, 0)
    out = np.where(out < 170, out, 0)
    #out = np.where(2*image[1] - image[0] - image[2] > 0, image, (0,0,0))
    return out

'''
do hough transform, filter out lines that are too far from 90deg, find interesectiosn (corners), try to count rectnagles based on this


'''
def detectRect(image):
    #src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src = image
    dst = cv2.Canny(src, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 100, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            #print(theta)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    cv2.imwrite('lines.png', cdst)
    show = cv2.resize(cdst, (500,500))
    showImage(show)
    show = cv2.resize(cdstP, (500,500))
    showImage(show)
'''
center of mass
just get xbar and ybar
this should get center of mass assuming each pixel has equal mass
'''
def com(group):
    rb = []
    cb = []
    minR = 10000
    minC = 10000
    maxR = -1
    maxC = -1
    for p in group:
        r,c = p
        if r < minR:
            minR = r
        if r > maxR:
            maxR = r
        if c < minC:
            minC = c
        if c > maxC:
            maxC = c
        rb.append(r)
        cb.append(c)
    rm = np.mean(rb)
    cm = np.mean(cb)
    return (rm,cm), (minR, minC), (maxR, maxC)

'''
find center of mass
find farthest vertical pixel
find farthest horizontal pixel
find difference
if diff < c * width or height accept otherwise reject
'''
def checkSquare(group):
    #gImage,center = createGroupImage(group)
    center, minP, maxP = com(group)
    maxR, maxC = maxP
    minR, minC = minP

    rB = maxR - minR
    cB = maxC - minC

    #rc, cc = center
    #rc = int(rc)
    #cc = int(cc)


    
    # sample in a few directions, see if the range is reasonable considering the 

    # use simple heuristic for now, width > height
    return rB < cB
def createGroupImage(group):
    center, minP, maxP = com(group)
    
    minR, minC = minP
    maxR, maxC = maxP
    rB = maxR - minR
    cB = maxC - minC
    
    out = np.zeros((rB+1, cB+1))
    for p in group:
        r,c = p
        out[r-minR][c-minC] = 255
    print('rB:', rB)
    print('cB:', cB)
    showImage(out)

    return rB, cB

def showImage(image, title='image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
