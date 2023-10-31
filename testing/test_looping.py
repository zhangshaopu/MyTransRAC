""" test of TransRAC """
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.my_tools import paint_smi_matrixs, plot_inference, density_map
from tools.findpeaks import find_peaks

torch.manual_seed(1)


def test_loop(n_epochs, model, test_set, inference=True, batch_size=1, lastckpt=None, paint=False, device_ids=[0]):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=10)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)

    if lastckpt is not None:
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        testOBO = []
        testMyOBO = []
        testMAE = []
        testMyMAE = []
        predCount = []
        Count = []
        if inference:
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(testloader, total=len(testloader))
                for input, target in pbar: #input处理过后的视频帧 target[1,64] timepoint的标准化后结果
                    model.eval()
                    acc = 0
                    input = input.to(device)
                    count = torch.sum(target, dim=1).round().to(device)
                    output, sim_matrix = model(input) # output(1,64) sim_matrix[1,12,64,64]

                    mycount = find_peaks(output ,batch_idx )
                    # paint_smi_matrixs(sim_matrix) # sim_matrix.shape = 
                    
                    predict_count = torch.sum(output, dim=1).round()
                    # plot_inference(predict_count , count)

                    mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                          predict_count.flatten().shape[0]  # mae
                    mymae = torch.sum(torch.div(torch.abs(mycount - count), count + 1e-1)) / \
                          mycount  # mae
                    # if predict_count.item() == count.item() and MAE < 0.2:
                    # np.savez('./result_'+ str(batch_idx),output = output.detach().cpu().numpy() , sim_matrix = sim_matrix.detach().cpu().numpy())
                    # density_map(output, count.item(batch_idx), batch_idx,batch_idx)

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    MyOBO = acc / mycount
                    testOBO.append(OBO)
                    testMyOBO.append(MyOBO)
                    MAE = mae.item()
                    MYMAE = mymae.item()
                    testMAE.append(MAE)
                    testMyMAE.append(MYMAE)


                    predCount.append(predict_count.item())
                    Count.append(count.item())
                    print('predict count :{0}, groundtruth :{1} , mycount :{2}'.format(predict_count.item(), count.item() , mycount))
                    batch_idx += 1

        print("MAE:{0},OBO:{1},MyMAE:{2},MyOBO:{3}".format(np.mean(testMAE), np.mean(testOBO) , np.mean(testMyMAE) , np.mean(testMyOBO)))
        # plot_inference(predict_count, count)
