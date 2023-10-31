import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.misc import electrocardiogram


def find_peaks(data,idx):
    data = data.detach().cpu().numpy().squeeze()
    pred = np.sum(data)
    print("pred:",pred)
    x = (data * 100000 ).astype(int)
    sorted_arr = np.sort(x)
    # 获取最大三个值和最小三个值
    max_three = sorted_arr[-3:]
    min_three = sorted_arr[:3]
    # 计算最大三个值和最小三个值的差值
    diff = np.max(max_three) - np.min(min_three)
    threshold  = diff / 3
    print("threshold:",threshold)

    # [1 ,2, 4 , 8]
    if(pred >= 9):
        order = 1
    elif(pred >= 4 and pred < 9):
        order = 2
    elif(pred >= 2 and pred < 4):
        order = 4
    else:
        order = 8
    print("order:",order)
    peaks = scipy.signal.argrelextrema(x, np.greater, order=order)
    # 根据阈值筛选极大值点
    selected_maxima_indices = peaks[0][x[peaks[0]] > threshold ]
    peaks = peaks[0]
    # plt.plot(x)
    # plt.plot(selected_maxima_indices, x[selected_maxima_indices], "o")
    # plt.text(0, 0, "count = %d" % len(selected_maxima_indices))
    # plt.title("scipy.signal.argrelextrema")
    # plt.savefig('./plot/findpeak_'+str(idx))
    # plt.clf()
    return len(selected_maxima_indices)
    

