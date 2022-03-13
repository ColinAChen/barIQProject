import cv2
import numpy as np
import matplotlib.pyplot as plt

from pyzbar.pyzbar import decode

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
PATH = 'hallway.jpg'#'aisle1.jpeg'#'barcode.jpg'#'chess.jpg'#'red.jpeg'
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



    #print(distance((1,2,3), (1,2,4)))
    image = cv2.imread(PATH)
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
    
    
    #g = cv2.imread('barcode.jpg', cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('gray.png', thresh)
    # decode barcode info
    decodeList = decode(thresh)
    print('num barcodes: ', len(decodeList))
    parseBarcodeList(decodeList)
    for barcode in decodeList: 

        # Locate the barcode position in image
        (x, y, w, h) = barcode.rect
         
        # Put the rectangle in image using
        # cv2 to heighlight the barcode
        #cv2.rectangle(thresh, (x-10, y-10),
        #              (x + w+10, y + h+10),
        #              (255, 0, 0), 2)
        cv2.rectangle(thresh, (x-10, y-10),
                  (x + w+10, y + h+10),
                  0, 2)


        if barcode.data!="":
           
            # Print the barcode data
            print(barcode.data)
            print(barcode.type)
    cv2.imwrite('detect.png', thresh)
    image = cv2.resize(image, (100,100))
    img = cv2.imread('aisle1.jpeg', -1)
    

    #normalize(img)
    

    #segments = segment(image, target=TARGET)
    #print(segments)
    #for seg in segments:
    
    #mask = preprocess(image)
    #groups = group(mask)
    
    #count = 0
    #out = np.zeros(mask.shape)
    #for g in groups:
    #    # do more refined counting
    #    # if we can expect that mask will return squares we can use line detection or something to count the number of rectangles
    #    if len(groups[g]) > COUNT_THRESHOLD:
    #        count += 1
    #        for p in groups[g]:
    #            r,c = p
    #            out[r][c] = 255
    #cv2.imwrite('count.png', out)
    #print('flags counted: ', count)
    

    #numFlags = count(image)
    
    out*=600
    out = np.where(out>255, 255, out)
    #cv2.imwrite('over.png', out)
    numFlags = count(out, target=(0,0,0))
    print('flags of color', TARGET, ': ', numFlags)
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

def barcode(image):
    pass
def parseBarcodeList(bq):
    for b in bq:
        print(b)

if __name__ == '__main__':
    main()
