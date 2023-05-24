import os

import sys
import numpy as np
import torch


import ternary
from scipy.stats import chi2
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.stats import dirichlet
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import ConvexHull, convex_hull_plot_2d
plt.style.use('plot_style.txt')



import ternary
import pandas as pd
import math
import itertools
import numpy as np
from pymatgen.core import Element, Composition
from matplotlib import cm
import matplotlib.pyplot as plt

def permute_point(p, permutation=None):
    """
    Permutes the point according to the permutation keyword argument. The
    default permutation is "012" which does not change the order of the
    coordinate. To rotate counterclockwise, use "120" and to rotate clockwise
    use "201"."""
    if not permutation:
        return p
    return [p[int(permutation[i])] for i in range(len(p))]

def unzip(l):
    """[(a1, b1), ..., (an, bn)] ----> ([a1, ..., an], [b1, ..., bn])"""
    return list(zip(*l))

def project_point(p, permutation=None):
    """
    Maps (x,y,z) coordinates to planar simplex.
    Parameters
    ----------
    p: 3-tuple
    The point to be projected p = (x, y, z)
    permutation: string, None, equivalent to "012"
    The order of the coordinates, counterclockwise from the origin
    """
    permuted = permute_point(p, permutation=permutation)
    a = permuted[0]
    b = permuted[1]
    x = a + b/2.
    y = (np.sqrt(3)/2) * b
    return np.array([x, y])

def fill_region(ax, color, points, pattern=None, zorder=-1000, alpha=None):
    """Draws a triangle behind the plot to serve as the background color
    for a given region."""
    vertices = map(project_point, points)
    xs, ys = unzip(vertices)
    poly = ax.fill(xs, ys, facecolor=color, edgecolor=color, hatch=pattern, zorder=zorder, alpha=alpha)
    return poly



def color_point(x, y, z, mean, cov,delta):
        p = np.array([x,y,z]) - mean
        if validate_bcr(p, delta, ns, cov) == True:
            r=0.5
            g=0.5
            b=0.5
            return (r, g, b, 1.)
        elif validate_var(p,delta,cov)== True:
            r = 0
            g = 1.0
            b = 0
            return (r,g,b,1.)
        else:
            r = 1.0
            g =1.0
            b = 1.0
            return (r,g,b,1.)




def generate_heatmap_data(mean,cov,delta,scale=1):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, mean,cov,delta)
    return d


def compute_dist(z,M):
    M = np.linalg.pinv(M)
    temp1 = np.matmul(z,M)
    temp2 = np.matmul(temp1,z)
    r = np.sqrt(temp2).flatten()[0]
    return r

def validate_bcr(z,delta,ns,cov):
    r = compute_dist(z,cov)
    rad = np.sqrt(chi2.ppf(delta, ns, loc=0, scale=1))
    if np.abs(r)<= np.abs(rad):
        return True
    else:
        return False


def validate_var(z,delta,cov):
    r = compute_dist(z, cov)
    rad = norm.ppf(delta)
    if np.abs(r)<= np.abs(rad):
        return True
    else:
        return False


def get_cartesian_from_barycentric(bs, t):
    pts=[]
    for pt in bs:
        pts.append(t.dot(pt).flatten())
    return np.array(pts)


def compute_covariance(alpha):
    sum_a = np.sum(alpha)
    norm_a = alpha/sum_a
    arr=[]
    l = len(alpha)
    for idx in range(l):
        arr_temp = []
        for jdx in range(l):
            if idx==jdx:
                k = (norm_a[idx]-norm_a[idx]*norm_a[jdx])/(sum_a+1)
            else:
                k = (-norm_a[idx] * norm_a[jdx]) / (sum_a + 1)
            arr_temp.append(k)

        arr.append(arr_temp)
    return np.array(arr)

def plot(alpha):
    variables = dirichlet.rvs(alpha, size=500, random_state=1)
    mean = dirichlet.mean(alpha)
    cov = dirichlet.var(alpha)
    t= np.transpose(np.array([[0.5,np.sqrt(3)*0.5],[1,0],[0,0]]))


    points_bcr=[]
    delta=0.8
    cov =compute_covariance(alpha)
    points_bcr_ = []
    for pt in variables:
        p = pt - mean
        if (validate_var(p, delta,cov)==False and validate_bcr(p,delta,ns, cov)==True):
            points_bcr.append(pt)

        if (validate_bcr(p,delta,ns, cov)==True):
            points_bcr_.append(pt)

    points_var=[]
    for pt in variables:
        p = pt - mean
        if validate_var(p, delta,cov)==True:
            points_var.append(pt)


    plt.clf()
    plt.figure(figsize=(8, 8))

    figure, tax = ternary.figure(scale=1.0)



    tax.boundary(linewidth=1.0)
    tax.scatter(variables, linewidth=0.01, label="Dirichlet samples",marker='.', color='grey',alpha=0.5)

    points_b = get_cartesian_from_barycentric(points_bcr_, t)
    # plt.fill(points_b[:,0],points_b[:,1],"blue",alpha=0.5)

    bcr_points = []
    hull = ConvexHull(points_b)

    plt.fill(points_b[hull.vertices, 0], points_b[hull.vertices, 1], 'b', alpha=0.2)

    for simplex in hull.simplices:
        x = np.array(points_bcr_)[simplex, 0]
        y = np.array(points_bcr_)[simplex, 1]
        z = np.array(points_bcr_)[simplex, 2]

        point = np.array([x, y, z]).transpose()
        bcr_points.append((x, y, z))



    plt.axis('off')
    points = get_cartesian_from_barycentric(points_var, t)
    hull = ConvexHull(points)

    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'magenta', alpha=0.3)


    var_points = []
    for simplex in hull.simplices:


        x = np.array(points_var)[simplex,0]
        y = np.array(points_var)[simplex,1]
        z= np.array(points_var)[simplex,2]


        point = np.array([x,y,z]).transpose()
        var_points.append((x,y,z))


    tax.clear_matplotlib_ticks()
    plt.annotate('$s_1$', xy=(0, -0.02), xytext=(-0.09, 0), fontsize=36)
    plt.annotate('$s_2$', xy=(0.5, np.sqrt(3) * 0.5), xytext=(0.5-0.02, 0.02+np.sqrt(3) * 0.5), fontsize=36)
    plt.annotate('$s_3$', xy=(1, 0), xytext=(1.02,0) ,fontsize=36)


    tax.savefig("plots_all/" + str(alpha[0]) + "_fig_var.png",format='png',bbox_inches='tight')



def plot_all(alpha):
    variables = dirichlet.rvs(alpha, size=500, random_state=1)
    mean = dirichlet.mean(alpha)
    cov = dirichlet.var(alpha)
    t= np.transpose(np.array([[0.5,np.sqrt(3)*0.5],[1,0],[0,0]]))


    deltas = [0.6,0.7,0.8,0.9,0.99]

    plt.clf()
    plt.figure(figsize=(8, 8))

    for delta in deltas:
        points_bcr=[]
        # delta=0.8
        cov =compute_covariance(alpha)
        points_bcr_ = []
        for pt in variables:
            p = pt - mean
            if (validate_var(p, delta,cov)==False and validate_bcr(p,delta,ns, cov)==True):
                points_bcr.append(pt)

            if (validate_bcr(p,delta,ns, cov)==True):
                points_bcr_.append(pt)

        points_var=[]
        for pt in variables:
            p = pt - mean
            if validate_var(p, delta,cov)==True:
                points_var.append(pt)



        # f, (ax1, ax2) = plt.subplots(1, 2)
        figure, tax = ternary.figure(scale=1.0)



        # figure, tax = ternary.figure(scale=1.0)
        tax.boundary(linewidth=1.0)
        tax.scatter(variables, linewidth=0.01, label="Dirichlet samples",marker='.', color='grey',alpha=0.5)
        # tax.scatter(points_bcr, linewidth=0.01, label="BCR",marker='.', color='blue',alpha=0.01)

        points_b = get_cartesian_from_barycentric(points_bcr_, t)
        # plt.fill(points_b[:,0],points_b[:,1],"blue",alpha=0.5)

        bcr_points = []
        hull = ConvexHull(points_b)

        plt.fill(points_b[hull.vertices, 0], points_b[hull.vertices, 1], 'b', alpha=0.3)

        for simplex in hull.simplices:
            x = np.array(points_bcr_)[simplex, 0]
            y = np.array(points_bcr_)[simplex, 1]
            z = np.array(points_bcr_)[simplex, 2]

            point = np.array([x, y, z]).transpose()
            bcr_points.append((x, y, z))
            # tax.plot(point,color='purple',linewidth=2.0,alpha=0.8)


        plt.axis('off')
        points = get_cartesian_from_barycentric(points_var, t)
        hull = ConvexHull(points)

        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'r', alpha=0.3)

        # cpoints_var=[]
        var_points = []
        for simplex in hull.simplices:

            x = np.array(points_var)[simplex,0]
            y = np.array(points_var)[simplex,1]
            # z = 1-x-y
            z= np.array(points_var)[simplex,2]



            point = np.array([x,y,z]).transpose()
            var_points.append((x,y,z))
            # tax.plot(point,color='red',linewidth=2.0,alpha=0.8)

        tax.clear_matplotlib_ticks()
        # tax.legend(loc='upper left', fontsize=16, frameon=False)
        plt.annotate('$s_1$', xy=(0, 0), xytext=(-0.08, 0), fontsize=36)
        plt.annotate('$s_2$', xy=(0.5, np.sqrt(3) * 0.5), xytext=(0.5+0.01, np.sqrt(3) * 0.5), fontsize=36)
        plt.annotate('$s_3$', xy=(1, 0), xytext=(1.01,0) ,fontsize=36)



        tax.savefig("plots_all/" + str(alpha[0]) + "_new_var.png",format='png',bbox_inches='tight')



if __name__=='__main__':

    path = "plots_all/"
    if os.path.exists(path) is False:
        os.mkdir(path)
    ns = 3
    alpha = np.ones(ns) * 1
    plot(alpha)

    alpha = np.ones(ns) * 5
    plot(alpha)

    alpha = np.ones(ns) * 30
    plot(alpha)

    alpha = np.ones(ns) * 50
    plot(alpha)


