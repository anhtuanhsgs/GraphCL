import numpy as np

def GetDices(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: BestDice: best possible Dice score
#         FgBgDice: dice score for joint foreground
#
# We assume that the background is labelled with 0
#
# For the original Dice score, labels corresponding to each other need to
# be known in advance. Here we simply take the best matching label from 
# gtLabel in each comparison. We do not make sure that a label from gtLabel
# is used only once. Better measures may exist. Please enlighten me if I do
# something stupid here...

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        print('Shapes of label images not identical.')
        return 0, 0
    
    inLabels = np.unique(inLabel)
    maxInLabel = np.amax(inLabels)
    maxGtLabel = np.amax(gtLabel)

    if(len(inLabels)<=1): # trivial solution
        print('Only one label given, assuming all background.')
        return 0, 0

    # calculate Dice between all labels using 2d histogram
    xedges = np.linspace( - 0.5, maxInLabel+0.5,maxInLabel+2) # interval edges of histogram 
    yedges = np.linspace( - 0.5, maxGtLabel+0.5,maxGtLabel+2) # interval edges of histogram 
    
    # histograms
    H2D, xedges, yedges = np.histogram2d(inLabel.flatten(), gtLabel.flatten(), bins=(xedges, yedges))
    inH1D, edges = np.histogram(inLabel.flatten(), bins=xedges)
    gtH1D, edges = np.histogram(gtLabel.flatten(), bins=yedges)
    
    # reshape 1d histograms for broadcast
    inH1D = np.reshape(inH1D,[len(inH1D),1])
    gtH1D = np.reshape(gtH1D,[1,len(gtH1D)])
    
    # best Dice is (2*overlap(A,B)/(size(A)+size(B)))
    perCombinationDice = 2*H2D/(inH1D + gtH1D + 1e-16)
    sMax = np.amax(perCombinationDice[1:,1:],1)
    bestDice = np.mean(sMax)
    
    # FgBgDice
    Overlap = np.sum(H2D[1:,1:])
    inFG = np.sum(inH1D[1:])
    gtFG = np.sum(gtH1D[:,1:])
    if ((inFG + gtFG)>1e-16):
        FgBgDice = 2*Overlap/(inFG + gtFG)
    else:
        FgBgDice = 1 # gt is empty and in has it found correctly
      
    return bestDice, FgBgDice

def DiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return -1

    inLabels = np.unique(inLabel)
    gtLabels = np.unique(gtLabel)
    maxInLabel = np.int(np.max(inLabels)) # maximum label value in inLabel
    minInLabel = np.int(np.min(inLabels)) # minimum label value in inLabel
    maxGtLabel = np.int(np.max(gtLabels)) # maximum label value in gtLabel
    minGtLabel = np.int(np.min(gtLabels)) # minimum label value in gtLabel

    return  (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel)