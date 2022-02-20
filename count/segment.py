import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
try rough image segmentation based on flood fill

this might work because if a light is shined on the pallet flags, they should be close in color


can try to do first pass grouping with localized histogram?
look for peaks with similar intensities/colors?

'''
TARGET = (215, 232, 232)
THRESHOLD = 20
COUNT_THRESHOLD = 200
def main():
    #print(distance((1,2,3), (1,2,4)))
    image = cv2.imread('aisle1.jpeg')
    img = cv2.imread('aisle1.jpeg', -1)
    normalize(img)
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
    numFlags = count(image)
    print('flags of color', TARGET, ': ', numFlags)
def count(image, target=None):
    mask = preprocess(image)
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
        sub = abs(image-target)
    cv2.imwrite('sub.png', sub)

    dist = np.sum(sub, 2)

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

def getNeighbors(image, p, target=None, threshold=20):
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
        if target is not None and distance(image[r+ar][c+ac], target) > threshold:
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
if __name__ == '__main__':
    main()
