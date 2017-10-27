import numpy as np
import matplotlib.pyplot as plt

def __check_type(y_pre,y_true):
    if type(y_pre)!=np.ndarray:
        y_pre=np.array(y_pre)
    if type(y_true)!=np.ndarray:
        y_true=np.array(y_true)
    return y_pre,y_true

def __check_y(y_pre,y_true):
    y_pre,y_true=__check_type(y_pre,y_true)
    if y_pre.shape!=y_true.shape:
        if y_true.ndim!=1:
            y_pre=y_pre.reshape([y_true.shape[0],y_true.shape[1]])
        else:
            y_pre=y_pre.reshape([-1,y_true.shape[0]])
    return y_pre,y_true

def accuracy_score(y_pre,y_true):
    y_pre,y_true=__check_y(y_pre,y_true)
    return (y_pre==y_true).sum()/max(y_pre.shape)

def r2_score(y_pre,y_true):
    y_pre,y_true=__check_y(y_pre,y_true)
    return 1-((y_true-y_pre)**2).sum()/((y_true-y_true.mean())**2).sum()
    
def MSE(y_pre,y_true):
    y_pre,y_true=__check_y(y_pre,y_true)
    return ((y_pre-y_true)**2).mean()

def p_score(y_pre,y_true,zh=1):
    y_pre,y_true=__check_y(y_pre,y_true)
    true=y_true==zh
    pre=y_pre==zh
    tp=true&pre
    fp=~true&pre
    if tp.sum()==0:
        return 0
    return tp.sum()/(tp.sum()+fp.sum())
    
def r_score(y_pre,y_true,zh=1):
    y_pre,y_true=__check_y(y_pre,y_true)
    true=y_true==zh
    pre=y_pre==zh
    tp=true&pre
    fn=true&~pre
    if tp.sum()==0:
        return 0
    return tp.sum()/(tp.sum()+fn.sum())
    
def f1_score(y_pre,y_true):
    p=p_score(y_pre,y_true)
    r=r_score(y_pre,y_true)
    if (p+r)==0:
        return 0
    return 2*p*r/(p+r)
    
def macro_f1_score(y_pre,y_true):
    p=[]
    r=[]
    for i in np.unique(y_true):
        p.append(p_score(y_pre,y_true,i))
        r.append(r_score(y_pre,y_true,i))
    print(p,r)
    if np.mean(p)==0 or np.mean(r)==0:
        return 0
    return 2*np.mean(p)*np.mean(r)/(np.mean(p)+np.mean(r))
        
def f_bete(y_pre,y_true,bete):
    p=p_score(y_pre,y_true)
    r=r_score(y_pre,y_true)
    if (p+r)==0:
        return 0
    return (1+bete**2)*p*r/(bete**2*p+r)
        
def Roc(y_pre,y_true):
    y_pre,y_true=__check_y(y_pre,y_true)
    Tpr=[]
    Fpr=[]
    for i in np.arange(0,1.01,0.01):
        pre=y_pre>=i
        true=y_true>=i
        tp=true&pre
        fn=true&~pre
        fp=pre&~true
        tn=~true&~pre
        tpr=tp.sum()/(tp.sum()+fn.sum())
        fpr=fp.sum()/(tn.sum()+fp.sum())
        if tn.sum()+fp.sum()==0:
            fpr=1.0
        Tpr.append(tpr)
        Fpr.append(fpr)
    plt.plot(Fpr,Tpr)
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.show()
    pass




