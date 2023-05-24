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

#
# corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
# AREA = 0.5 * 1 * 0.75**0.5
# triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
#
# refiner = tri.UniformTriRefiner(triangle)
# # trimesh = refiner.refine_triangulation(subdiv=4)
#
# plt.figure(figsize=(8, 4))
#
# # plt.subplot(1, 2, i+ 1)
# plt.triplot(triangle)
# plt.axis('off')
# plt.axis('equal')
#
# plt.savefig('fig1')

# # For each corner of the triangle, the pair of other corners
# pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# # The area of the triangle formed by point xy and another pair or points
# tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))
#
# def xy2bc(xy, tol=1.e-4):
#     '''Converts 2D Cartesian coordinates to barycentric.'''
#     coords = np.array([tri_area(xy, p) for p in pairs]) / AREA
#     return np.clip(coords, tol, 1.0 - tol)
#
#

#

def compute_dist(z,M):
    M = np.linalg.inv(M)
    temp1 = np.matmul(z,M)
    temp2 = np.matmul(temp1,z)
    r = np.sqrt(temp2).flatten()[0]
    return r

def validate_bcr(z,delta,ns,cov):
    r = compute_dist(z,cov)
    rad = np.sqrt(chi2.ppf(delta, ns-1, loc=0, scale=1))
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




def plot(alpha):
    variables = dirichlet.rvs(alpha, size=1000, random_state=1)
    mean = dirichlet.mean(alpha)
    cov = dirichlet.var(alpha)
    t= np.transpose(np.array([[0.5,np.sqrt(3)*0.5],[1,0],[0,0]]))


    points_bcr=[]
    delta=0.8
    cov = np.diag(cov)
    for pt in variables:
        p = pt - mean
        if validate_bcr(p,delta,ns, cov)==True:
            points_bcr.append(pt)

    points_var=[]
    for pt in variables:
        p = pt - mean
        if validate_var(p, delta,cov)==True:
            points_var.append(pt)

    # tax.scatter(variables, linewidth=0.02, label="samples",marker='.', color='red')
    # tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
    # tax.legend()
    # tax.clear_matplotlib_ticks()
    #
    # tax.savefig("plots/" +str(alpha[0]) + "_fig_plain.png")
    plt.clf()
    plt.figure(figsize=(9, 9))

    # f, (ax1, ax2) = plt.subplots(1, 2)
    figure, tax = ternary.figure(scale=1.0)

    # tax = ternary.TernaryAxesSubplot(ax=ax1,scale=1.0)
    # figure, tax = ternary.figure(ax=ax1,scale=1.0)
    # ax1.xaxis.set_tick_params(labelsize=20)

    # figure.set_size_inches(12,8)
    # ax2.xaxis.set_tick_params(labelsize=24)
    # ax2.yaxis.set_tick_params(labelsize=24)


    # figure, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=1.0)
    tax.scatter(variables, linewidth=0.01, label="Dirichlet samples",marker='.', color='grey',alpha=0.5)
    tax.scatter(points_bcr, linewidth=0.01, label="BCR",marker='.', color='blue',alpha=0.5)
    points = get_cartesian_from_barycentric(points_bcr, t)
    hull = ConvexHull(points)
    # for simplex in hull.simplices:
    #     # ax1.plot(points[simplex, 0], points[simplex, 1], 'k-')
    #     plt.plot(points[simplex, 0], points[simplex, 1],color='blue',linewidth=2.0,alpha=0.8)


    # tax.ticks(axis='lbr', multiple=1.0, linewidth=2, tick_formats="%.1f",fontsize=14)
    # plt.tick_params(axis='both', pad=20)

    tax.legend(loc='upper left',fontsize=16)
    plt.annotate('(1,0,0)', xy=(0, 0),xytext=(-0.1,0))
    plt.annotate('(0,1,0)', xy=(0.5, np.sqrt(3) * 0.5))
    plt.annotate('(0,0,1)', xy=(1, 0))

    tax.clear_matplotlib_ticks()
    plt.axis('off')

    tax.savefig("plots_all/" +str(alpha[0]) + "_fig_bcr.pdf",format='pdf')
    plt.clf()
    plt.figure(figsize=(9, 9))

    # figure, tax = ternary.figure(ax=ax2,scale=1.0)
    # figure.set_size_inches(12, 8)
    # ax2.xaxis.set_tick_params(labelsize=20)

    figure, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=1.0)
    tax.scatter(variables, linewidth=0.01, label="Dirichlet samples", marker='.', color='grey', alpha=0.5)
    tax.scatter(points_bcr, linewidth=0.01, label="VaR", marker='.', color='red', alpha=0.5)
    plt.axis('off')
    points = get_cartesian_from_barycentric(points_var, t)
    # hull = ConvexHull(points)
    # for simplex in hull.simplices:
    #     plt.plot(points[simplex, 0], points[simplex, 1], color='red',linewidth=2.0,alpha=0.8)

    plt.annotate('(1,0,0)', xy=(0, 0), xytext=(-0.1,0))
    plt.annotate('(0,1,0)', xy=(0.5, np.sqrt(3)*0.5))
    plt.annotate('(0,0,1)', xy=(1, 0))


                    # tax.ticks(axis='lbr', multiple=1.0, linewidth=2, tick_formats="%.1f",fontsize=14)
    # plt.tick_params(axis='both', pad=20)

    tax.legend(loc='upper left',fontsize=16)
    tax.clear_matplotlib_ticks()

    tax.savefig("plots_all/" + str(alpha[0]) + "_fig_var.pdf",format='pdf')



if __name__=='__main__':
    path = "plots_all/"
    if os.path.exists(path) is False:
        os.mkdir(path)
    ns = 3
    alpha = np.ones(ns) * 1
    plot(alpha)

    alpha = np.ones(ns) * 5
    plot(alpha)

    alpha = np.ones(ns) * 50
    plot(alpha)

    alpha = [10,10,1]
    print("mean", dirichlet.mean(alpha))
    plot(alpha)