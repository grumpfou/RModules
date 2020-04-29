try:
    import cv2
except ImportError as e:
    cv2 = False
import matplotlib.pyplot as plt
import os

class VideoCreator:
    def __init__(self,anim_tmp_dirname = 'anim_tmp',anim_name = 'Video.avi',FPS = 10):
        if not cv2:
            raise ImportError('cv2 not installed')
        self.anim_tmp_dirname = anim_tmp_dirname
        self.anim_name = anim_name
        self.FPS = FPS
        if not os.path.isabs(self.anim_tmp_dirname):
            self.anim_tmp_dirname = os.path.abspath(self.anim_tmp_dirname)
            # if '__file__' in locals():
            #     self.anim_tmp_dirname = os.path.join(os.path.split(__file__)[0], self.anim_tmp_dirname)
            # else:
            #     self.anim_tmp_dirname = os.path.join(os.curdir, self.anim_tmp_dirname)
        if os.path.splitext(self.anim_name)[1]=='':
            self.anim_name += '.avi'
        self.idx_display = 0
        self.n_zeros = 10
        self.imagesList = []


    def saveFig(self,f=None):
        if f is None:
            f = plt.gcf()
        if not os.path.exists(self.anim_tmp_dirname):
                os.mkdir(self.anim_tmp_dirname)
        file = os.path.join(self.anim_tmp_dirname,
                'fig_'+ str(self.idx_display).zfill(self.n_zeros)+ '.png')
        f.savefig(file)
        self.imagesList.append(file)
        self.idx_display += 1

    def createVid(self,figsize=None):
        if len( self.imagesList)==0:
            raise FileNotFoundError('No images has yet been created')
        output=self.anim_name
        if figsize is None:
            f = plt.gcf()
            figsize=f.get_size_inches()

        figsize = (figsize[0]/(figsize[0]+figsize[1]),figsize[1]/(figsize[0]+figsize[1]))
        shape = int(2000*figsize[0]),int(2000*figsize[1])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output, fourcc, self.FPS, shape)
        idx=0
        for im in self.imagesList:
            assert os.path.exists(im)
            image = cv2.imread(im)
            resized=cv2.resize(image,shape)
            # cv2.putText(img = resized, text = 't= '+str(idx)+ " days", org = (80,120), fontFace = cv2.FONT_HERSHEY_SIMPLEX,  fontScale = 3, color = (0, 0, 0))
            video.write(resized)
            idx+=1
        video.release()
        print('animation created')
        for f in self.imagesList:
            os.remove(f)
        self.imagesList = []
        # os.remove(self.anim_tmp_dirname)
