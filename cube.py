import cv2
import numpy as np


def is_present(points, c):
    for pts in points:
        if max(pts[0]) > c[0] > min(pts[0]):
            if max(pts[1]) > c[1] > min(pts[1]):
                return True
    return False


def create_mask(image, hsv, lower, upper, center, key):
    pts = []
    for lv, uv in zip(lower, upper):
        mask = cv2.inRange(hsv, lv, uv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5), iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0.2*image.shape[0]*image.shape[1]
        for cnt in contours:
            if 600 < cv2.contourArea(cnt) < max_area:
                points = cv2.minAreaRect(cnt)
                points = np.array(cv2.boxPoints(points), dtype=np.int32)
                dim = abs(points[0][0]-points[2][0]), abs(points[1][1]-points[3][1])
                c, _ = cv2.minEnclosingCircle(points)
                if 0.8 < dim[0]/dim[1] < 1.2:
                    if len(pts) == 0 or not is_present(pts, c):
                        image = cv2.fillPoly(image, [points], 255)
                        cv2.circle(image, np.array(c, np.int32), 3, (255, 255, 255), -1)
                        center[0].append(c[0]), center[1].append(c[1]), center[2].append(key)
                        pts.append([[points[x][0] for x in range(4)], [points[x][1] for x in range(4)]])
    return image, center


def counting_order(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 25, minLineLength=70, maxLineGap=40)
    slopes = np.zeros((int(len(lines)),))
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        slopes[i] = np.arctan(abs(y1-y2)/abs((x1-x2)+0.0000001))
    if np.sum((slopes > (np.pi*5/18))+(slopes < np.pi*2/9)) > 0.9*len(slopes):
        return True
    return False


images = ['data/cube1.jpeg', 'data/cube2.jpeg', 'data/cube3.jpeg', 'data/cube4.jpeg', 'data/cube5.jpeg',
          'data/cube6.jpeg', 'data/cube7.jpeg', 'data/cube8.jpeg', 'data/cube9.jfif', 'data/cube10.jfif',
          'data/cube11.PNG']
for link in images:
    img = cv2.imread(link)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)   # HSV image
    lower_dict = {
        'blue': [np.array([100, 140, 0], np.uint8)],
        'green': [np.array([36, 150, 0], np.uint8)],
        'white': [np.array([0, 0, 185], np.uint8)],
        'orange': [np.array([5, 110, 125], np.uint8)],
        'red': [np.array([160, 70, 50], np.uint8), np.array([0, 70, 50], np.uint8)],
        'yellow': [np.array([21, 110, 117], np.uint8)]
    }
    upper_dict = {
        'blue': [np.array([140, 255, 255], np.uint8)],
        'green': [np.array([86, 255, 255], np.uint8)],
        'white': [np.array([255, 130, 255], np.uint8)],
        'orange': [np.array([18, 255, 255], np.uint8)],
        'red': [np.array([180, 255, 255], np.uint8), np.array([5, 255, 210], np.uint8)],
        'yellow': [np.array([45, 255, 255], np.uint8)]
    }
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    center = [[], [], []]
    key_list = [x for x in lower_dict.keys()]
    for key in key_list:
        mask, center = create_mask(mask, hsv_img, lower_dict[key], upper_dict[key], center, key)
    mask = cv2.bitwise_and(img, img, mask=mask)
    center = [(x, y, k) for x, y, k in zip(center[0], center[1], center[2])]
    datatype = [('x', float), ('y', float), ('k', np.dtype('<U16'))]
    center = np.array(center, dtype=datatype)
    if counting_order(mask):
        center = np.sort(center, order='y')
    else:
        center = sorted(center, key=lambda x: x['x']**2+x['y']**2)
    n = np.uint8(np.sqrt(len(center)))
    for i in range(n):
        center[i*n:(i+1)*n] = np.sort(center[i*n:(i+1)*n], order='x')
    appeared = []
    order = np.ones((len(center)//1,))
    for i, c in enumerate(center):
        try:
            ind = appeared.index(c['k'])
            order[i] = str(ind+1)
        except:
            appeared.append(c['k'])
            order[i] = str(appeared.index(c['k'])+1)
    print(order.reshape(n, n))
    for i, c in zip(order, center):
        img = cv2.putText(img, str(int(i)), (np.int32(c['x'] - 5), np.int32(c['y'] + 5)), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()