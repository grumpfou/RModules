import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def general_title(text,fontsize=25,**kargs):
    """ Will plot a general title to the figure, usefull expecially if there is
    several subplots.
    - text: the title to display
    - fontsize: the size of the font used in the title (a little bit bigger
        that the one of the subplots
    - **kargs: other matplotlib.pyplot.text arguments
    """
    if fontsize!=None:
        kargs['fontsize']=fontsize

    return plt.text(0.5, 0.95, text, transform=plt.gcf().transFigure,
                                        horizontalalignment='center',**kargs)


def plot_with_se(X,Y=None,se=None,alpha_se=0.5,color_se=None,style='fill',
                                                    kargs_se=None,*args,**kargs):
    """Plot the an array with its standard-error given in se.
    X :    - if Y is None then it will be the data to plot.
        - if Y is not None, then it will be the abscisses of the data
    Y : The mean of the data to plot (If None, then we will take X and the
        abscisse will be arange(X.shape[0])
    se : the standard error that will give the area place from both side of the
        mean. It will plot it in the same color as the line.
    alpha_se : the amount of transparency to put to the standard error area.
    style : str in ['fill','line']
        if 'fill': will be in fill the space between X and se
        if 'line': will plot a line at se
    kargs_se: the kargs specific to the sdandard-error
    *args,**kargs : argument that will be put in the mean ploting arguments
        (see matplotlib.pyplot.plot options)
    """
    if kargs_se==None: kargs_se={}
    assert style in ['fill','line']
    if type(X)==list: X = np.array(X)
    if type(Y)==list: Y = np.array(Y)
    if type(se)==list: se = np.array(se)
    if type(Y)==type(None):
        Y=X.copy()
        X=np.arange(Y.shape[0])
    mainline,=plt.plot(X,Y,*args,**kargs)
    if color_se==None:
        c = mainline.get_color()
    else:
        c = color_se
    if  type(se)!= type(None):
        assert Y.shape==se.shape
        mean_up=Y+se
        mean_dn=Y-se
        if style == 'fill':
            se_obj = plt.fill_between(X,mean_dn,mean_up,linewidth=0,
                            facecolor=c,alpha=alpha_se,**kargs_se)
        else:
            if 'color' in kargs_se:
                kargs_se.pop('color')

            se_line_dn = plt.plot(X,mean_dn, color=c,alpha=alpha_se,**kargs_se)
            se_line_up = plt.plot(X,mean_up, color=c,alpha=alpha_se,**kargs_se)
            se_obj = [se_line_dn[0],se_line_up[0]]

    handler = Handler_plot_with_se(mainline,se_obj,style)
    ### LEGEND #########################################################################################

    class AnyObject(object):pass



    return handler,AnyObject

from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class Handler_plot_with_se(HandlerBase):
    def __init__(self,mainline,se_obj,style,*args,**kargs):
        HandlerBase.__init__(self,*args,**kargs)
        self.style = style
        self.mainline = mainline
        self.se_obj = se_obj

    def getLineKargs(self,line):
        return dict(color         = line.get_color(),
                    alpha         = line.get_alpha(),
                    linestyle     = line.get_linestyle(),
                    linewidth     = line.get_linewidth(),
                    marker         = line.get_marker(),
                    )
    def getPatchKargs(self,patch):
        return dict(facecolor     = patch.get_facecolor()[0],
                    alpha         = patch.get_alpha(),
                    linestyle     = patch.get_linestyle()[0],
                    linewidth     = patch.get_linewidth(),
                    )

    def create_artists(self, legend, orig_handle,
                                        x0, y0, width, height, fontsize, trans):
        # x0, y0 = handlebox.xdescent, handlebox.ydescent
        # width, height = handlebox.width, handlebox.height
        patch_line = mlines.Line2D([x0,x0+width], [y0+height/2,y0+height/2],
            **self.getLineKargs(self.mainline)
            )
        if     self.style == 'fill':
            patch_rect = mpatches.Rectangle([x0, y0], width, height,
                **self.getPatchKargs(self.se_obj))
            # handlebox.add_artist(patch_rect)
            # handlebox.add_artist(patch_line)
            return [patch_line,patch_rect]

        else:
            patch_line1 = mlines.Line2D([x0,x0+width],[y0+height,y0+height],
                **self.getLineKargs(self.se_obj[1]))
            patch_line2 = mlines.Line2D([x0,x0+width],[0,0],
                **self.getLineKargs(self.se_obj[0]))

            # handlebox.add_artist(patch_line1)
            # handlebox.add_artist(patch_line2)
            # handlebox.add_artist(patch_line)
            return [patch_line,patch_line1,patch_line2]

    ####################################################################################################



class color_normalization:
    def __init__(self,rge,plt_cm=None,type_norm=None):
        """
        Will give a convient way to use color maps of matplotlib :
        rge : the range of values it has to take :
            - if float : colors will be in [0,rge]
            - if array_like : colors will be in [min(rge),max[rge]]
        plt_cm : the matplotlib color map to use. By default plt.cm.jet.
        type_norm : in ['linear','lognorm'] if lognorm, the Normalization will
            be log. If None, will try to guess with the array
        """
        assert type_norm in [None,'linear','lognorm']
        if plt_cm==None : plt_cm=plt.cm.jet
        self.plt_cm = plt_cm
        self.type_norm = type_norm
        rge_new = np.array(rge).ravel()
        if len(rge_new)==1:
            self.lim = [0,rge_new[0]]
        else:
            self.lim = [rge_new.min(),rge_new.max()]
        self.rge=rge_new

        if type_norm is None:
            type_norm = 'linear'
            diff_ = np.diff(np.log(rge_new))
            if len(diff_>2) and (np.abs(diff_/diff_[0]-1)<1e-9).all():
                type_norm = 'lorgnorm'

        if type_norm == 'linear':

            self.norm = matplotlib.colors.Normalize(self.lim[0],self.lim[1])
        else:
            self.norm = matplotlib.colors.LogNorm(self.lim[0],self.lim[1])

    def __getitem__(self,key):
        """
        Will give the color corresponding to 'key'.
        """
        if self.lim[0]<=key<=self.lim[1] :
            return self.plt_cm(self.norm(key))
        elif self.lim[0]>key:
            return self.plt_cm(self.norm(self.lim[0]))
        elif self.lim[1]<key:
            return self.plt_cm(self.norm(self.lim[1]))
        else:
            raise ValueError("This the key "+str(key)+"does not seem a proper value inside the limits: "+str(self.lim))

    def get_colorbar(self,cb_array=None,additional_array=None,additional_label=None,*args,**kargs):
        """
        To call to have the colorbar :
        - cb_array : the array to display to the colorbar :
            - if None : will display the self.rge
            - if array-like : will display the array-like
        - additional_array: is not None
            will plot a second array on the left
        - *args,**kargs : argument of the intense colorbar
        """
        m = plt.cm.ScalarMappable(norm=self.norm,cmap=self.plt_cm)
        if cb_array==None: cb_array = self.rge
        m.set_array(cb_array)
        return plt.colorbar(m,*args,**kargs)

def addArrayToColorbar(cbar,add_array,add_label=None,type_norm='linear'):
        ax = cbar.ax
        pos = cbar.ax.get_position()
        cbar.ax.set_aspect('auto')

        #### create a second axes instance and set the limits you need
        ax = cbar.ax.twinx()
        if type_norm == 'lognorm':
            ax.set_yscale('log')
        ax.set_ylim([add_array[0],add_array[-1]])


        #### resize the colorbar (otherwise it overlays the plot)
        pos.x0 +=0.05
        cbar.ax.set_position(pos)
        ax.set_position(pos)

        if not add_label  is None:
            cbar.ax.yaxis.set_label_position("left")
            t = cbar.ax.yaxis.get_label()
            properties = {s:t.properties()[s] for s in ['fontfamily','fontfamily', 'fontname', 'fontproperties', 'fontsize', 'fontstyle', 'fontvariant', 'fontweight']}
            cbar.ax.set_ylabel(add_label,**properties)





def pmf_histogram(a,bins=None,range=None, normed=False, weights=None,
        density=None):
    """ Will return a bar histogram (especially usefull to describe the resulte
    of discreate simulations).
    - a: array_like, the array from which the histogram is made.
    - bins: if None --> will display for all integer from a.min() to a.max()+1
            if int --> will display for all integer in a range from 0 to bins.
            if (int,int) --> will display for all integer in a range from
                bins[0] to bins[1].
            if array --> will display for all integer in the array.
    - range, normes, weights, density : numpy.histogram options.

    Returns (Y array, X array)
    """
    if not isinstance(a,np.ndarray):
        a=np.array(a)
    if bins is None:
        bins=np.arange(a.min(),a.max()+1)
    elif type(bins)==int:
        bins=np.arange(bins)
    elif type(bins)==tuple and len(bins)==2:
        bins=np.arange(bins[0],bins[1])
    Y,X = np.histogram(a,bins=bins,range=range, normed=normed, weights=weights,
            density=density)
    return Y,bins[:-1]

def pmf_hist(a,bins=None,range=None, normed=False, weights=None, density=None,\
            shift=0.,style='bar', *arg,**kargs):
    """ Will plot a bar histogram (especially usefull  to describe the resulte
    of discreate simulations).
    - a: array_like, the array from which the histogram is made.
    - bins: if None --> will display for all integer from a.min() to a.max()+1
            if int --> will display for all integer in a range from 0 to bins.
            if (int,int) --> will display for all integer in a range from
                bins[0] to bins[1].
            if array --> will display for all integer in the array.
    - range, normes, weights, density : numpy.histogram options.
    - shift: the shift to apply to the x axis (usefull when several bar
            histograms are ploted).
    - style: the style of representation : in ['bar','plot']
    - *arg,**kargs : matplotlib.pylab.bar or matplotlib.pylab.plot options
            (depending on the style chosen).

    Returns (Y array, X array, matplotlib.collections.LineCollection object)
    """
    Y,bins = pmf_histogram(
            a,
            bins=bins,
            range=range,
            normed=normed,
            weights=weights,
            density=density)
    if len(arg)==0 and (not 'color' in kargs):
        cc = plt.gca()._get_lines.prop_cycler
        kargs['color'] = next(cc)['color']
    if style=='bar':
        Z = plt.bar(bins+shift,Y,*arg,**kargs)
    elif style=='plot':
        Z = plt.plot(bins+shift,Y,*arg,**kargs)
    else:
        raise ValueError('option '+style+'is not recognized')

    return Y,bins,Z

def pmf_plot(a,bins=None,range=None, normed=False, weights=None, density=None,\
            shift=0., *arg,**kargs):
    """ Will plot a function histogram (especially usefull  to describe the
    resulte of discreate simulations).
    - a: array_like, the array from which the histogram is made.
    - bins: if None --> will display for all integer from a.min() to a.max()+1
            if int --> will display for all integer in a range from 0 to bins.
            if (int,int) --> will display for all integer in a range from
                bins[0] to bins[1].
            if array --> will display for all integer in the array.
    - range, normes, weights, density : numpy.histogram options.
    - shift: the shift to apply to the x axis (usefull when several bar
            histograms are ploted).
    - *arg,**kargs : matplotlib.pylab.vlines options.

    Returns (Y array, X array, matplotlib.collections.LineCollection object)
    """
    Y,bins = pmf_histogram(
            a,
            bins=bins,
            range=range,
            normed=normed,
            weights=weights,
            density=density)
    if len(arg)==0 and (not 'color' in kargs):
        cc = plt.gca()._get_lines.color_cycle
        kargs['color'] = next(cc)

    Z = plt.plot(bins+shift,Y,*arg,**kargs)

    return Y,bins,Z


def qqplot(a,b,bins=None,**kargs):
    """
    Will compare the distributions of a and b by plotling them one agains the
    other using a Quantil-Quantil-Plot
    - a: array-like, first set of data that from which the first distribution
        will be compute
    - b: array-like, second set of data that from which the second
        distribution will be compute
    - bins: the array of quantil on which we should cut the distribution
        (default: np.arange(0,1.1,.1))
    - **kargs: other arguments in plt.plot (color etc.)
    """
    if bins==None:
        bins = np.arange(0,1.1,.1)
    else:
        bins=np.array(bins)

    perc1 = np.percentile(a,bins*100)
    perc2 = np.percentile(b,bins*100)
    if 'marker' not in kargs.keys():
        # kargs['marker']='circle'
        kargs['marker']='o'

    Z = plt.plot(perc1,perc2,**kargs)
    return perc1,perc2,Z

def rimshow(a,aspect='auto',xlim=None,ylim=None,xlog=False,ylog=False,**kargs):
    """Will show the 2dim array a in a color map (with a the good orientation).
    - aspect: see matplotlib.imshow description
    - xlim and ylim: list of values that represent the x-axis (respect the
        y-axis)
        - if None: the extend will be from 0 to a.shape[0] (to a.shape[1] if
             it is the y-axis).
        - if a number: the extend will be from 0 to xlim (or ylim)
        - if a list/tuple: the extend will be from xlim[0] to xlim[-1] (or
            from ylim[0] to xlim[-1])

    - xlog and ylog: bool
        if True, set to the axis to be log
    """
    def get_lim(lim,shape):
        if lim is None: return [0,shape]
        elif type(lim)==float or type(lim)==int: return [0,lim]
        else: return [lim[0],lim[-1]]

    extent = get_lim(xlim,a.shape[0])+get_lim(ylim,a.shape[1])
    plt.imshow(a.transpose(),origin='lower',aspect='auto',extent=extent,**kargs)
    if xlog:plt.xscale('log')
    if ylog:plt.yscale('log')




def text_axes(x,y,s,trans_x=True,trans_y=True,ax=None,*args,**kargs):
    import matplotlib.transforms as mtransforms
    if ax==None: ax=plt.gca()

    trans = mtransforms.blended_transform_factory(
            ax.transAxes if trans_x else ax.transData,
            ax.transAxes if trans_y else ax.transData,
            )
    plt.text(x=x,y=y,s=s,transform=trans,*args,**kargs)


def rsubplots(nrows=1, ncols=1,figsize=None,*args,**kargs):
    if figsize==None:
        x,y = plt.rcParams['figure.figsize']
        figsize = (ncols*x,nrows*y)
    return plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize,*args,**kargs)
