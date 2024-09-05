from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import Ellipse, Rectangle

from .plot_funs import colorize, zoom_image


# Adding patches

def add_aperture(ax,sh,loc,rad, color='white',lw=2):
    ax.add_patch(Ellipse(np.array(sh)/2-loc*np.array(sh), 
                            width=2*rad*sh[0],
                            height=2*rad*sh[1],
                            edgecolor=color,
                            facecolor='none',
                            linewidth=lw))


def add_linesNcircs(sh, angs, rads, ax=None, sc=1, ls_rad=['--','-'], ls_ang=['--','-']):
    if ax is None:
        ax = plt.gca()
    ny, nx = sh
    for ia, angle in enumerate(angs):
        ax.plot([nx//2,nx//2+nx*np.cos(angle)*sc/2], 
                [ny//2,ny//2+ny*np.sin(angle)*sc/2],
                c='white',
                linewidth=2,
                ls=ls_ang[ia])

    for ir, rad in enumerate(rads):
        ax.add_patch(Ellipse(np.array((nx//2,ny//2)), 
                                width=rad*nx*sc,
                                height=rad*ny*sc,
                                edgecolor='white',
                                facecolor='none',
                                linewidth=2,
                                ls=ls_rad[ir]))

# plotting funs

def plot_holoWzoom(holo, ax=None, rz=0.1, lw=2, xyzoom=[0.5,0.5], interpolation=None, cmap='gray'):
    if ax is None:
        ax = plt.gca()
    ax.imshow(holo, cmap=cmap, interpolation=interpolation)
    ax.axis('off')
    sh = holo.shape

    x1, x2, y1, y2 = sh[0]*(xyzoom[0]-rz), sh[0]*(xyzoom[0]+rz), \
        sh[1]*(xyzoom[1]-rz), sh[1]*(xyzoom[1]+rz)  # subregion of the original image
    
    axins = ax.inset_axes(
        [0.58, 0.58, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y2, y1), xticklabels=[], yticklabels=[])
    axins.imshow(holo, cmap=cmap, interpolation=interpolation)
    axins.set_xticks([])
    axins.set_yticks([])    
    
    _patch, pp1, pp2 = mark_inset(ax, axins, loc1=2, loc2=4, lw=lw, ec='w')
    pp1.loc1, pp1.loc2 = 2, 3  # inset corner 1 to origin corner 4 (would expect 1)
    pp2.loc1, pp2.loc2 = 4, 1  # inset corner 3 to origin corner 2 (would expect 3)

    for axis in ['top','bottom','left','right']:
        axins.spines[axis].set_linewidth(lw)
        axins.spines[axis].set_color('w')

def plot_four(four, ax=None, log=True, zoom=1, aperture=True, ap_loc=[0,0], ap_rad=1/4, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.axis('off')

    afour = np.abs(four)
    if log:
        afour = np.log(afour)
    
    ax.imshow(zoom_image(afour,zoom), cmap='inferno', **kwargs)
    if aperture:
        add_aperture(ax, np.array(four.shape)/zoom, ap_loc, zoom*ap_rad)


def plot_fieldWzoom(field, ax=None, rz=0.1, lw=2, xyzoom=[0.5,0.5]):
    if ax is None:
        ax = plt.gca()
    ax.imshow(colorize(field))
    ax.axis('off')
    sh = field.shape

    x1, x2, y1, y2 = sh[0]*(xyzoom[0]-rz), sh[0]*(xyzoom[0]+rz), \
        sh[1]*(xyzoom[1]-rz), sh[1]*(xyzoom[1]+rz)  # subregion of the original image
    
    axins = ax.inset_axes(
        [0.58, 0.58, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y2, y1), xticklabels=[], yticklabels=[])
    axins.imshow(colorize(field))
    axins.set_xticks([])
    axins.set_yticks([])    
    
    _patch, pp1, pp2 = mark_inset(ax, axins, loc1=2, loc2=4, lw=lw, ec='w')
    pp1.loc1, pp1.loc2 = 2, 3  # inset corner 1 to origin corner 4 (would expect 1)
    pp2.loc1, pp2.loc2 = 4, 1  # inset corner 3 to origin corner 2 (would expect 3)

    for axis in ['top','bottom','left','right']:
        axins.spines[axis].set_linewidth(lw)
        axins.spines[axis].set_color('w')

# Tick formatting

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


def get_seg_mask(X, Y, phi,st=0.01):
    Xr = np.cos(phi)*X+np.sin(phi)*Y
    Yr = -np.sin(phi)*X+np.cos(phi)*Y
    return np.logical_and(np.logical_and(np.abs(Yr)<st , Xr>=0), Xr<1)

def get_circle_mask(X, Y, r, st=0.01):
    return np.logical_and((r-st/2)**2 < X**2+Y**2 , X**2+Y**2 < (r+st/2)**2)

def plot_amp_phase_proj(X, Y, target, fields, angs, rads, st_ang=.001,st_rad=.001, sp_lbl='SP'):
    nx, ny = target.shape
    circ_mask = get_circle_mask(X, Y, rads[0], st=st_ang)
    seg_mask = get_seg_mask(X, Y, angs[0], st=st_rad)

    fig, axs = plt.subplots(1,2, figsize=(0.8*12,0.8*5))
    col_list = ['r', 'b', 'teal']
    ls_list = ['--','-']
    target_phases = np.angle(target[circ_mask])+np.pi
    lns = []
    ln, = axs[0].plot(np.abs(target[seg_mask]),np.abs(target[seg_mask]), c='gray')
    axs[1].plot(target_phases,target_phases,c='gray')
    lns += [ln]

    for i in range(len(fields)):

        for ia, angle in enumerate(angs):
            seg_mask = get_seg_mask(X, Y, angle, st=st_ang)
            axs[0].plot(np.abs(target[seg_mask]),np.pi*np.abs(fields[i][seg_mask]), 
                        c=col_list[i],ls=ls_list[ia])
            # lns += [ln]

        for ir, rad in enumerate(rads):
            circ_mask = get_circle_mask(X, Y, rad, st=st_rad)
            target_phases = np.angle(target[circ_mask])+np.pi
            inds = np.argsort(target_phases)
            target_phases = target_phases[inds]
            unwrapped_phases = np.unwrap(np.angle(fields[i][circ_mask])[inds])
            if unwrapped_phases[0] >np.pi/2:
                unwrapped_phases -= np.pi
            else:
                unwrapped_phases += np.pi
            
            unwrapped_phases = np.mod(unwrapped_phases,2*np.pi)
            unwrapped_phases[np.logical_and(unwrapped_phases>2*np.pi-np.pi/8, target_phases<np.pi/2)] \
             -= 2*np.pi
            unwrapped_phases[np.logical_and(unwrapped_phases<np.pi/8, target_phases>3*np.pi/2)] \
             += 2*np.pi

            ln, = axs[1].plot(target_phases,
                        unwrapped_phases,
                        # np.mod(unwrapped_phases+0.0*np.pi,2*np.pi)-0.0*np.pi,
                        c=col_list[i],ls=ls_list[ir])
            lns += [ln]
    axs[0].grid(True)
    axs[0].set_xlabel('Target amplitude')
    axs[0].set_ylabel('Shaped amplitude')
    axs[1].set_xlabel('Target phase')
    axs[1].set_ylabel('Shaped phase')
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    axs[1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    axs[1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axs[1].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    axs[1].yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axs[1].legend([lns[0],lns[2],lns[4],lns[6]],['Target', r'$\parallel$ Lee', r'$\perp$ Lee',sp_lbl])
    fig.tight_layout()