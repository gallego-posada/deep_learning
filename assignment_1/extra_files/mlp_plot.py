import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.6)
sns.set_style("white")

def smooth(x,window_len=11,window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

tr_stats = pickle.load(open("./numpy_mlp_training.pkl","rb"))

window = 25
stats = np.reshape(np.array(tr_stats), (-1,5))


def plot_smoothed(x, y, name):
    smooth_x = smooth(x, window)
    plt.plot(x, alpha = 0.5)
    plt.plot(smooth_x[window-1:], color = 'blue', label='Train ')

    smooth_y = smooth(y, window)
    plt.plot(y, alpha = 0.5)
    plt.plot(smooth_y[window-1:], color = 'green', label='Test ')
    plt.ylabel(name)
    plt.xlim(xmax = len(x))
    plt.xlabel('Iteration')
    plt.legend(bbox_to_anchor=(0.2, 1.02, 0.7, .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
    plt.show()

#plot_smoothed(stats[:,1], stats[:,3], 'Loss')
#plot_smoothed(stats[:,2], stats[:,4], 'Accuracy')

print(stats[-5:,:])
