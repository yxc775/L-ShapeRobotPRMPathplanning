import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

#Paramers
NUMSAMPLE = 500
NUMEDGE = 10
MAXEDGELEN = 35

ANIMATIONON = True
DRAWEDGE = False
DRAWMAP = False
ISPLANNING = True



class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

def dijkstra_planning(sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    """

    nstart = Node(sx, sy, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    while True:
        if not openset:
            print("Cannot find path")
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        # show graph
        if ANIMATIONON and len(closedset.keys()) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    # generate final course
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind

    return rx, ry


class KDtree:
    def __init__(self,source):
        self.tree = scipy.spatial.cKDTree(source)

    def search(self,data, k = 1):
        """search neighbor"""
        if len(data.shape) >= 2:
            indices = []
            distances = []
            for i in data.T:
                distance, index = self.tree.query(i,k = k)
                indices.append(index)
                distances.append(distance)
            return indices,distances
        distances, indices = self.tree.query(data, k=k)
        return indices, distances

def PRM(sourcex, sourcey, goalx, goaly, obstaclex, obstacley, pointRadius, edgelength):
    obstacleTree = KDtree(np.vstack((obstaclex, obstacley)).T)
    samplerawX, samplerawY = samplePoints(sourcex, sourcey, goalx, goaly, pointRadius, edgelength, obstaclex, obstacley, obstacleTree)
    sampleX, sampleY = reFormatarrays(samplerawX,samplerawY)
    if ANIMATIONON:
        plt.plot(sampleX[0], sampleY[0], ".b",markersize = 0.95)
        plt.plot(sampleX[1], sampleY[1], ".r")
        plt.plot(sampleX[2], sampleY[2], ".m",markersize = 0.95)
        if DRAWEDGE:
            for i in range(0,len(sampleX[2])):
                plt.plot([sampleX[1][i], sampleX[2][i]], [sampleY[1][i], sampleY[2][i]], 'k-', linewidth=0.8)
                plt.plot([sampleX[0][i], sampleX[1][i]], [sampleY[0][i], sampleY[1][i]], 'k-', linewidth=0.8)

    roadMap = createMap(sampleX, sampleY, pointRadius, obstacleTree)
    if ANIMATIONON:
        if DRAWMAP:
            drawRoadMap(roadMap,sampleX[1],sampleY[1])

    rx = []
    ry = []
    if ISPLANNING:
        rx, ry = dijkstra_planning(sourcex[1], sourcey[1], goalx[1], goaly[1], obstaclex, obstacley, pointRadius, roadMap, sampleX[1], sampleY[1])
    return rx, ry

def reFormatarrays(sX,sY):
    sampleX1 = []
    sampleY1 = []

    sampleX2 = []
    sampleY2 = []

    sampleX3 = []
    sampleY3 = []
    for i in range(0,len(sX)):
        sampleX1.append(sX[i][0])
        sampleX2.append(sX[i][1])
        sampleX3.append(sX[i][2])
        sampleY1.append(sY[i][0])
        sampleY2.append(sY[i][1])
        sampleY3.append(sY[i][2])


    return [sampleX1,sampleX2,sampleX3], [sampleY1,sampleY2,sampleY3]



def createMap(sampleX, sampleY, pointRadius, obstacleTree):
    road_map = []
    nsample = len(sampleX[1])
    skdtree =KDtree(np.vstack((sampleX[1], sampleY[1])).T)
    codelist = []
    for (i, ix, iy) in zip(range(nsample), sampleX[1], sampleY[1]):
        index, dists = skdtree.search(
            np.array([ix, iy]).reshape(2, 1), k=nsample)
        inds = index[0]
        edge_id = []

        for j in range(1, len(inds)):
            nx = [sampleX[0][inds[j]],sampleX[1][inds[j]],sampleX[2][inds[j]]]
            ny = [sampleY[0][inds[j]],sampleY[1][inds[j]],sampleY[2][inds[j]]]

            if not collideLShape([sampleX[0][i],sampleX[1][i],sampleX[2][i]],[sampleY[0][i],sampleY[1][i],sampleY[2][i]] , nx, ny, pointRadius, obstacleTree):
                edge_id.append(inds[j])

            if len(edge_id) >= NUMEDGE:
                break
        road_map.append(edge_id)
    #  plot_road_map(road_map, sample_x, sample_y)
    return road_map

def collideLShape(ix, iy, gx, gy, pointRadius, obstacleTree):
    dx, dy, angle, d = getDistanceAndAngle(ix[1], iy[1], gx[1], gy[1])
    if d >= MAXEDGELEN:
        return True
    nstep = round(d/pointRadius)
    cx,cy = ix,iy

    for i in range(nstep):
        idxs0, dist0 = obstacleTree.search(np.array([cx[0], cy[0]]).reshape(2, 1))
        idxs1, dist1 = obstacleTree.search(np.array([cx[1], cy[1]]).reshape(2, 1))
        idxs2, dist2 = obstacleTree.search(np.array([cx[2], cy[2]]).reshape(2, 1))
        if dist1[0] <= pointRadius:
            return True #collide
        else:
            if dist0[0] > pointRadius and dist2[0] > pointRadius and satisfiedEdgePointConstrained([cx[0],cy[0]],[cx[1],cy[1]],[cx[2],cy[2]],pointRadius,obstacleTree):
                for i in range(0,3):
                    cx[i] += pointRadius * math.cos(angle)
                    cy[i] += pointRadius * math.sin(angle)
            else:
                ##we try to rotate the body to find a better pose, if there exist a valid pose, we pass this step:
                ##otherwise, we continue to rotate to search for valid pose until we reach all possible rotation
                ##Then, we return False
                tempx = cx
                tempy = cy
                #Searching better pose
                isCollide = True
                for i in range(0,360):
                    ang = i * math.pi/180
                    tempx[0],tempy[0] = rotate([cx[1],cy[1]],[cx[0],cy[0]],ang)
                    tempx[2], tempy[2] = rotate([cx[1],cy[1]],[cx[2],cy[2]], ang)
                    idxs0, dist0 = obstacleTree.search(np.array([tempx[0], tempy[0]]).reshape(2, 1))
                    idxs1, dist1 = obstacleTree.search(np.array([tempx[1], tempy[1]]).reshape(2, 1))
                    idxs2, dist2 = obstacleTree.search(np.array([tempx[2], tempy[2]]).reshape(2, 1))
                    if dist0[0] > pointRadius and dist1[0] > pointRadius and dist2[0] > pointRadius and satisfiedEdgePointConstrained(
                            [tempx[0], tempy[0]], [tempx[1], tempy[1]], [tempx[2], tempy[2]], pointRadius, obstacleTree):
                        for i in range(0, 3):
                            tempx[i] += pointRadius * math.cos(angle)
                            tempy[i] += pointRadius * math.sin(angle)
                            cx[i] = tempx[i]
                            cy[i] = tempy[i]
                            isCollide = False
                            if not isCollide:
                                break
                    if not isCollide:
                        break

    # goal point check
    idxs, dist = obstacleTree.search(np.array([gx[1], gy[1]]).reshape(2, 1))
    if dist[0] <= pointRadius:
        return True  # collision

    return False


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def isCollide(x, y, gx, gy, pointRadius, obstacleTree):
    dx,dy,angle,d = getDistanceAndAngle(x,y,gx,gy)
    if d >= MAXEDGELEN:
        return True
    D = pointRadius
    nstep = round(d/D)

    for i in range(nstep):
            idxs, dist = obstacleTree.search(np.array([x, y]).reshape(2, 1))
            if dist[0] <= pointRadius:
                return True  # collision
            x += D * math.cos(angle)
            y += D * math.sin(angle)

        # goal point check
    idxs, dist = obstacleTree.search(np.array([gx, gy]).reshape(2, 1))
    if dist[0] <= pointRadius:
        return True
    return False



def checkEdgePoint(x,y,pointRadius,obstacleTree):
    if satisfiedEdgePointConstrained([x[0],y[0]],[x[1],y[1]],[x[2],y[2]], pointRadius, obstacleTree):
        return True
    else:
        return False


def getDistanceAndAngle(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy,dx)
    d = math.sqrt(dx**2 + dy**2)
    return dx,dy,angle,d

def drawRoadMap(road_map,sampleX,sampleY):
    codelist = []
    if ANIMATIONON:
        for i in range(0,len(road_map)):
                for j in range(0, len(road_map[i])):
                    code = encodeEdgePair(sampleX[i], sampleY[i], j, sampleX, sampleY)
                    if code not in codelist:
                        codelist.append(code)
                        connectpoints(sampleX[i], sampleY[i], road_map[i][j], sampleX, sampleY)

def encodeEdgePair(ix,iy,id,sx,sy):
    code = ix * 3 + iy * 7 + sx[id] * 3 + sy[id] * 7
    return code

def connectpoints(ix,iy,id,sampleX,sampleY):
    x2 = sampleX[id]
    y2 = sampleY[id]
    plt.plot([ix,x2],[iy,y2],'k-',linewidth= 1.00)




def samplePoints(sourcex, sourcey, goalx, goaly, pointradius, edgelength, obstaclex, obstacley, obstacleTree):
    #use obstacle as limit to the sampling to avoid over-sampling or sampling too spread
    maxx = max(obstaclex)
    maxy = max(obstacley)
    minx = min(obstaclex)
    miny = min(obstacley)

    sampleXstate,sampleYstate = [],[]
    while len(sampleXstate) <= NUMSAMPLE:
        pickedX = (random.random() - minx) * (maxx - minx)
        pickedY = (random.random() - miny) * (maxy - miny)
        X, Y  = generateBenchPoints(pickedX, pickedY, edgelength)
        if satisfiedEdgePointConstrained([X[0],Y[0]], [X[1],Y[1]], [X[2],Y[2]], pointradius, obstacleTree):
            index, distance = obstacleTree.search(np.array([X[0],Y[0]]).reshape(2, 1))
            index2, distance2 = obstacleTree.search(np.array([X[1],Y[1]]).reshape(2, 1))
            index3, distance3 = obstacleTree.search(np.array([X[2],Y[2]]).reshape(2, 1))

            if distance[0] >= pointradius and  distance2[0] >= pointradius and distance3[0] >= pointradius:
                sampleXstate.append(X)
                sampleYstate.append(Y)

    #Add goal point and source point into sampling data
    sampleXstate.append(sourcex)
    sampleYstate.append(sourcey)
    sampleXstate.append(goalx)
    sampleYstate.append(goaly)

    return sampleXstate,sampleYstate


def generateBenchPoints(x,y,e):
    # radius of the circle
    # center of the circle (x, y)
    circle_x = x
    circle_y = y
    # random angle
    alpha = 2 * math.pi * random.random()
    beta = alpha + 0.5 * math.pi
    # calculating coordinates
    x1 = e * math.cos(alpha) + circle_x
    y1 = e * math.sin(alpha) + circle_y

    x2 = e * math.cos(beta) + circle_x
    y2 = e * math.sin(beta) + circle_y

    return [x1,x,x2],[y1,y,y2]

def createMapBound(ox,oy,startxy,endxy):
    for i in range(int(endxy)):
        ox.append(i)
        oy.append(startxy)
    for i in range(int(endxy)):
        ox.append(endxy)
        oy.append(i)
    for i in range(int(endxy+1)):
        ox.append(i)
        oy.append(endxy)
    for i in range(int(endxy+1)):
        ox.append(startxy)
        oy.append(i)

    return ox,oy

def addObstables(ox,oy,row,tocolumn):
    for i in range(int(tocolumn)):
        ox.append(row)
        oy.append(i)

    return ox,oy

def addObstaclesUpDown(ox,oy,row,length):
    for i in range(int(length)):
        ox.append(row)
        oy.append(60 - i)

    return ox,oy



def satisfiedEdgePointConstrained(p1, p2, p3, pointradius, obstacleTree):
    obstacleConstraint = (not isCollide(p1[0], p1[1], p2[0], p2[1], pointradius, obstacleTree)) and (not isCollide(p2[0], p2[1], p3[0], p3[1], pointradius, obstacleTree))

    if obstacleConstraint:
        return True
    else:
        return False

def pointd(p1,p2):
    dx = p1[0] -p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)

def testCase1():
    ox,oy = [],[]
    ox,oy = createMapBound(ox,oy,0.0,60.0)
    ox,oy = addObstables(ox,oy,20.0,40.0)
    ox,oy = addObstaclesUpDown(ox,oy,40.0,40.0)
    ox,oy = addObstables(ox,oy,40.0,15.0)
    return ox,oy

def testCase2():
    ox,oy = [],[]
    ox,oy = createMapBound(ox,oy,0.0,60.0)
    ox,oy = addObstables(ox,oy,20.0,40.0)
    ox,oy = addObstaclesUpDown(ox,oy,40.0,40.0)
    ox,oy = addObstables(ox,oy,40.0,16.0)
    ox,oy = addLine(ox,oy)
    return ox,oy

def testCase3():
    ox,oy = [],[]
    ox,oy = createMapBound(ox,oy,0.0,60.0)
    ox,oy = addObstables(ox,oy,20.0,55.0)
    ox,oy = addObstaclesUpDown(ox,oy,40.0,55)
    return ox,oy


def addLine(ox,oy):

    for i in range(35,40):
        ox.append(i)
        oy.append(16)

    for i in range(40,45):
        ox.append(i)
        oy.append(16)

    for i in range(35, 40):
        ox.append(i)
        oy.append(20)

    for i in range(40, 45):
        ox.append(i)
        oy.append(20)

    return ox,oy




def main():
    edgelength = 5.0
    pointradius = 2.0
    x = 10.0
    y = 10.0
    startX,startY = generateBenchPoints(x,y,edgelength)
    print("X " + str(startX))
    print("Y " + str(startY))

    gx = 50.0
    gy = 50.0
    goalX,goalY = generateBenchPoints(gx,gy,edgelength)
    ox,oy = testCase1()

    if ANIMATIONON:
        plt.plot(ox, oy, ".k")
        plt.plot(startX[0], startY[0], "^m")
        plt.plot(startX[1], startY[1], "^m")
        plt.plot(startX[2], startY[2], "^m")

        plt.plot(goalX[0], goalY[0], "^m")
        plt.plot(goalX[1], goalY[1], "^m")
        plt.plot(goalX[2], goalY[2], "^m")
        plt.grid(True)
        plt.axis("equal")

    rx,ry = PRM(startX, startY, goalX, goalY, ox, oy, pointradius, edgelength)
    if ISPLANNING:
        assert rx, 'Cannot found path'
    if ANIMATIONON:
        plt.plot(rx, ry, "-r", linewidth=3.0)
        plt.show()

def experiments():
    edgelength = 5.0
    pointradius = 2.0
    x = 10.0
    y = 10.0
    startX, startY = generateBenchPoints(x, y, 5)
    print("X " + str(startX))
    print("Y " + str(startY))

    ox,oy = testCase2()

    gx = 50.0
    gy = 50.0
    goalX, goalY = generateBenchPoints(gx, gy, 5)

    rx, ry = PRM(startX, startY, goalX, goalY, ox, oy, pointradius, edgelength)
    passed = 0
    for i in range(0,10):
        if len(rx) != 1:
            passed += 1

        rx, ry = PRM(startX, startY, goalX, goalY, ox, oy, pointradius, edgelength)

    print(float(passed)/float(10))



if __name__ == '__main__':
    main()
