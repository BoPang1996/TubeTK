import os
import shutil
import numpy as np
from scipy.optimize import linear_sum_assignment
from easydict import EasyDict as edict
import configparser

INF = 1e8


def linear_sum_assignment_with_inf(cost_matrix):
    """
    This is a workaround from 'https://github.com/scipy/scipy/issues/6900'
    """
    cost_matrix = np.asarray(cost_matrix).copy()
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix)


def parseSequences2(seqmapFile):
    assert (os.path.exists(seqmapFile)), 'seqmap file {} does not exist'.format(seqmapFile)
    with open(seqmapFile) as f:
        allseq = [x.strip() for x in f.readlines()[0:]]
    return allseq


def boxiou(x1, y1, w1, h1, x2, y2, w2, h2):
    def boxIntersect(bboxleft1, bboxright1, bboxbottom1, bboxup1, bboxleft2, bboxright2, bboxbottom2, bboxup2):
        hor = np.max((0, np.min((bboxright1, bboxright2)) - np.max((bboxleft1, bboxleft2))))
        if hor < 1e-8:
            return 0.0
        ver = np.max((0, np.min((bboxbottom1, bboxbottom2)) - np.max((bboxup1, bboxup2))))
        if ver < 1e-8:
            return 0.0
        return hor * ver

    def boxUnion(bboxleft1, bboxright1, bboxbottom1, bboxup1, bboxleft2, bboxright2, bboxbottom2, bboxup2, isect=None):
        a1 = bboxright1 - bboxleft1
        b1 = bboxbottom1 - bboxup1
        a2 = bboxright2 - bboxleft2
        b2 = bboxbottom2 - bboxup2
        union = a1 * b1 + a2 * b2
        if isect is not None:
            bisect = isect
        else:
            bisect = boxIntersect(bboxleft1, bboxright1, bboxbottom1, bboxup1, bboxleft2, bboxright2, bboxbottom2,
                                  bboxup2)
        return union - bisect

    bisect = boxIntersect(x1, x1 + w1, y1 + h1, y1, x2, x2 + w2, y2 + h2, y2)
    if bisect < 1e-8:
        return 0.0
    bunion = boxUnion(x1, x1 + w1, y1 + h1, y1, x2, x2 + w2, y2 + h2, y2, bisect)
    assert bunion > 0, 'something wrong with union computation'
    iou = bisect / bunion
    return iou


def bbox_overlap(bbox1, bbox2):
    return boxiou(bbox1[0], bbox1[1], bbox1[2], bbox1[3], bbox2[0], bbox2[1], bbox2[2], bbox2[3])
    # return boxiou(bbox1[0], bbox1[1], bbox1[2] - bbox1[0], bbox1[3] - bbox1[1], bbox2[0], bbox2[1], bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])


def classIDToString(classID):
    labels = ['ped',
              'person_on_vhcl',
              'car',
              'bicycle',
              'mbike',
              'non_mot_vhcl',
              'static_person',
              'distractor',
              'occluder',
              'occluder_on_grnd',
              'occluder_full',
              'reflection',
              'crowd']
    if classID < 1 or classID > len(labels):
        return 'unknown'
    return labels[classID - 1]


def preprocessResult(resFile, seqName, dataDir=None, force=True, minvis=0.0):
    def cleanRequired(seqFolder):
        return 'CVPR19' in seqFolder or 'MOT16' in seqFolder or 'MOT17' in seqFolder

    # assert cleanRequired(seqName), 'preproccessing should only be done for MOT15/16/17 and CVPR 19'

    if not os.path.exists(resFile):
        print('Results file does not exist')
        return

    p = os.path.dirname(resFile)
    f, e = os.path.splitext(os.path.basename(resFile))
    cleanDir = os.path.join(p, 'clean')
    if not os.path.exists(cleanDir):
        os.makedirs(cleanDir)
    resFileClean = os.path.join(cleanDir, f + e)
    if not force and os.path.exists(resFileClean):
        print('skipping...')
        return

    tf_ = os.path.getsize(resFile)
    if tf_ == 0:
        print('Results file empty')
        shutil.copy(resFile, resFileClean)
        return

    def getSeqInfoFromFile(seq, dataDir):
        seqFolder = os.path.join(dataDir, seqName)
        seqInfoFile = os.path.join(dataDir, seqName, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seqInfoFile)

        imgFolder = config.get('Sequence', 'imDir')
        frameRate = config.getint('Sequence', 'frameRate')
        F = config.getint('Sequence', 'seqLength')
        imWidth = config.getint('Sequence', 'imWidth')
        imHeight = config.getint('Sequence', 'imHeight')
        imgExt = config.get('Sequence', 'imExt')

        return seqName, seqFolder, imgFolder, frameRate, F, imWidth, imHeight, imgExt

    seqName, seqFolder, imgFolder, frameRate, F, imWidth, imHeight, imgExt \
        = getSeqInfoFromFile(seqName, dataDir)

    resRaw = np.loadtxt(resFile, delimiter=',')

    gtFolder = os.path.join(dataDir, seqName, 'gt')
    gtFile = os.path.join(gtFolder, 'gt.txt')
    gtRaw = np.loadtxt(gtFile, delimiter=',')

    assert np.shape(gtRaw)[1] == 9, 'unknown GT format'

    if 'CVPR19' in seqName:
        distractors = ['person_on_vhcl', 'static_person', 'distractor', 'reflection', 'non_mot_vhcl']
    else:
        distractors = ['person_on_vhcl', 'static_person', 'distractor', 'reflection']

    keepBoxes = np.ones((np.shape(resRaw)[0],), dtype=bool)

    td = 0.5
    for t in range(1, F + 1):
        resInFrame = np.where(resRaw[:, 0] == t)[0]
        N = len(resInFrame)
        resInFrame = np.reshape(resInFrame, (N,))

        GTInFrame = np.where(gtRaw[:, 0] == t)[0]
        Ngt = len(GTInFrame)
        GTInFrame = np.reshape(GTInFrame, (Ngt,))

        allisects = np.zeros((Ngt, N))
        g = 0
        for gg in GTInFrame:
            g = g + 1
            r = 0
            bxgt, bygt, bwgt, bhgt = gtRaw[gg, 2:6]
            for rr in resInFrame:
                r = r + 1
                bxres, byres, bwres, bhres = resRaw[rr, 2:6]

                if bxgt + bwgt < bxres or bxgt > bxres + bwres:
                    continue
                if bygt + bhgt < byres or bygt > byres + bhres:
                    continue

                allisects[g - 1, r - 1] = boxiou(bxgt, bygt, bwgt, bhgt, bxres, byres, bwres, bhres)

        tmpai = allisects
        tmpai = 1 - tmpai
        # tmpai[tmpai > td] = np.inf
        # mGT, mRes = linear_sum_assignment_with_inf(tmpai)
        tmpai[tmpai > td] = INF
        mGT, mRes = linear_sum_assignment(tmpai)
        Mtch = np.zeros_like(tmpai)
        Mtch[mGT, mRes] = 1
        nMtch = len(mGT)

        for m in range(nMtch):
            g = GTInFrame[mGT[m]]
            r = resInFrame[mRes[m]]
            if (tmpai[mGT[m]][mRes[m]] == INF):
                continue

            gtClassID = gtRaw[g, 7].astype(np.int)
            gtClassString = classIDToString(gtClassID)

            if gtClassString in distractors:
                keepBoxes[r] = False

            if gtRaw[g, 8] < minvis:
                keepBoxes[r] = False

    resNew = resRaw
    resNew = resRaw[keepBoxes, :]
    np.savetxt(resFileClean, resNew)
    return resFileClean


def clear_mot_hungarian(gtDB, stDB, threshold, world, VERBOSE=False):
    # TODO: This function comes from https://github.com/shenh10/mot_evaluation/blob/master/utils/measurements.py
    # TO BE reimplemented

    """
    compute CLEAR_MOT and other metrics
    [recall, precision, FAR, GT, MT, PT, ML, falsepositives, false negatives, idswitches, FRA, MOTA, MOTP, MOTAL]
    """
    # st_frames = np.unique(stDB[:, 0])
    gtDB = gtDB.astype(np.int)
    stDB = stDB.astype(np.int)
    gt_frames = np.unique(gtDB[:, 0])
    st_ids = np.unique(stDB[:, 1])
    gt_ids = np.unique(gtDB[:, 1])
    # f_gt = int(max(max(st_frames), max(gt_frames)))
    # n_gt = int(max(gt_ids))
    # n_st = int(max(st_ids))
    f_gt = len(gt_frames)
    n_gt = len(gt_ids)
    n_st = len(st_ids)

    mme = np.zeros((f_gt,), dtype=float)  # ID switch in each frame
    c = np.zeros((f_gt,), dtype=float)  # matches found in each frame
    fp = np.zeros((f_gt,), dtype=float)  # false positives in each frame
    missed = np.zeros((f_gt,), dtype=float)  # missed gts in each frame

    g = np.zeros((f_gt,), dtype=float)  # gt count in each frame
    d = np.zeros((f_gt, n_gt), dtype=float)  # overlap matrix
    Mout = np.zeros((f_gt, n_gt), dtype=float)
    allfps = np.zeros((f_gt, n_st), dtype=float)

    gt_inds = [{} for i in range(f_gt)]
    st_inds = [{} for i in range(f_gt)]
    M = [{} for i in range(f_gt)]  # matched pairs hashing gid to sid in each frame

    # hash the indices to speed up indexing
    for i in range(gtDB.shape[0]):
        frame = np.where(gt_frames == gtDB[i, 0])[0][0]
        gid = np.where(gt_ids == gtDB[i, 1])[0][0]
        gt_inds[frame][gid] = i

    gt_frames_list = list(gt_frames)
    for i in range(stDB.shape[0]):
        # sometimes detection missed in certain frames, thus should be assigned to groundtruth frame id for alignment
        frame = gt_frames_list.index(stDB[i, 0])
        sid = np.where(st_ids == stDB[i, 1])[0][0]
        st_inds[frame][sid] = i

    for t in range(f_gt):
        g[t] = len(gt_inds[t].keys())

        # preserving original mapping if box of this trajectory has large enough iou in avoid of ID switch
        if t > 0:
            mappings = list(M[t - 1].keys())
            sorted(mappings)
            for k in range(len(mappings)):
                if mappings[k] in gt_inds[t].keys() and M[t - 1][mappings[k]] in st_inds[t].keys():
                    row_gt = gt_inds[t][mappings[k]]
                    row_st = st_inds[t][M[t - 1][mappings[k]]]
                    dist = bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])
                    if dist >= threshold:
                        M[t][mappings[k]] = M[t - 1][mappings[k]]
        # mapping remaining groundtruth and estimated boxes
        unmapped_gt, unmapped_st = [], []
        unmapped_gt = [key for key in gt_inds[t].keys() if key not in M[t].keys()]
        unmapped_st = [key for key in st_inds[t].keys() if key not in M[t].values()]
        if len(unmapped_gt) > 0 and len(unmapped_st) > 0:
            square_size = np.max((len(unmapped_gt), len(unmapped_st)))
            overlaps = np.zeros((square_size, square_size), dtype=float)
            for i in range(len(unmapped_gt)):
                row_gt = gt_inds[t][unmapped_gt[i]]
                for j in range(len(unmapped_st)):
                    row_st = st_inds[t][unmapped_st[j]]
                    dist = 1 - bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])
                    if dist <= threshold:
                        overlaps[i][j] = dist
            overlaps[overlaps == 0.0] = 1e8
            matched_indices = linear_sum_assignment(overlaps)

            for matched in zip(*matched_indices):
                if overlaps[matched[0], matched[1]] == 1e8:
                    continue
                M[t][unmapped_gt[matched[0]]] = unmapped_st[matched[1]]

        # compute statistics
        cur_tracked = list(M[t].keys())
        fps = [key for key in st_inds[t].keys() if key not in M[t].values()]
        for k in range(len(fps)):
            allfps[t][fps[k]] = fps[k]
        # check miss match errors
        if t > 0:
            for i in range(len(cur_tracked)):
                ct = cur_tracked[i]
                est = M[t][ct]
                last_non_empty = -1
                for j in range(t - 1, 0, -1):
                    if ct in M[j].keys():
                        last_non_empty = j
                        break
                if ct in gt_inds[t - 1].keys() and last_non_empty != -1:
                    mtct, mlastnonemptyct = -1, -1
                    if ct in M[t]:
                        mtct = M[t][ct]
                    if ct in M[last_non_empty]:
                        mlastnonemptyct = M[last_non_empty][ct]

                    if mtct != mlastnonemptyct:
                        mme[t] += 1
        c[t] = len(cur_tracked)
        fp[t] = len(st_inds[t].keys())
        fp[t] -= c[t]
        missed[t] = g[t] - c[t]
        for i in range(len(cur_tracked)):
            ct = cur_tracked[i]
            est = M[t][ct]
            row_gt = gt_inds[t][ct]
            row_st = st_inds[t][est]
            d[t][ct] = 1 - bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])
        for k in M[t].keys():
            Mout[t][k] = M[t][k] + 1;
    return mme, c, fp, missed, g, d, Mout, allfps


def CLEAR_MOT_HUN(gtMat, resMat, threshold, world):
    metricsInfo = edict()
    metricsInfo.names = edict()
    metricsInfo.names.long = ['Recall', 'Precision', 'False Alarm Rate',
                              'GT Tracks', 'Mostly Tracked', 'Partially Tracked', 'Mostly Lost',
                              'False Positives', 'False Negatives', 'ID Switches', 'Fragmentations',
                              'MOTA', 'MOTP', 'MOTA Log']
    metricsInfo.names.short = ['Rcll', 'Prcn', 'FAR',
                               'GT', 'MT', 'PT', 'ML',
                               'FP', 'FN', 'IDs', 'FM',
                               'MOTA', 'MOTP', 'MOTAL']

    metricsInfo.widths = edict()
    metricsInfo.widths.long = [6, 9, 16, 9, 14, 17, 11, 13, 15, 15, 11, 14, 5, 5, 8]
    metricsInfo.widths.short = [5, 5, 5, 3, 3, 3, 3, 2, 4, 4, 3, 3, 5, 5, 5]

    metricsInfo.format = edict()
    metricsInfo.format.long = {'.1f', '.1f', '.2f',
                               'i', 'i', 'i', 'i',
                               'i', 'i', 'i', 'i', 'i',
                               '.1f', '.1f', '.1f'}
    metricsInfo.format.short = metricsInfo.format.long
    additionalInfo = edict()

    _, ic = np.unique(gtMat[:, 1], return_inverse=True)
    gtMat[:, 1] = ic
    _, ic2 = np.unique(resMat[:, 1], return_inverse=True)
    resMat[:, 1] = ic2

    VERBOSE = False
    mme, c, fp, m, g, d, alltracked, allfalsepos = clear_mot_hungarian(gtMat, resMat, threshold, VERBOSE)
    # ! Caution: alltracked is 0-indexed
    Fgt = np.max(gtMat[:, 0])
    Ngt = len(np.unique(gtMat[:, 1]))
    F = np.max(resMat[:, 0])
    missed = np.sum(m)
    falsepositives = np.sum(fp)
    idswitches = np.sum(mme)

    MOTP = (1.0 - np.sum(np.sum(d)) / np.sum(c)) * 100
    if world:
        MOTP = MOTP / threshold
    if np.isnan(MOTP):
        MOTP = 0.0

    MOTAL = (1 - ((np.sum(m) + np.sum(fp) + np.log10(np.sum(mme) + 1)) / np.sum(g))) * 100
    MOTA = (1 - ((np.sum(m) + np.sum(fp) + (np.sum(mme))) / np.sum(g))) * 100
    recall = np.sum(c) / np.sum(g) * 100
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100
    if np.isnan(precision):
        precision = 0.0
    FAR = np.sum(fp) / Fgt

    MTstatsa = np.zeros((Ngt,))
    for i in range(Ngt):
        gtframes = gtMat[gtMat[:, 1] == i, 0]
        gttotallength = gtframes.size
        trlengtha = np.where(alltracked[gtframes.astype(np.int) - 1, i] > 0)[0].size
        if trlengtha / gttotallength < 0.2:
            MTstatsa[i] = 3
        elif F >= np.nonzero(gtMat[gtMat[:, 1] == i, 0])[0][-1] + 1 and trlengtha / gttotallength <= 0.8:
            MTstatsa[i] = 2
        elif trlengtha / gttotallength >= 0.8:
            MTstatsa[i] = 1

    MT = (np.where(MTstatsa == 1))[0].size
    PT = (np.where(MTstatsa == 2))[0].size
    ML = (np.where(MTstatsa == 3))[0].size

    fr = np.zeros((Ngt,))
    for i in range(Ngt):
        beg = np.where(alltracked[:, i] > 0)[0]
        end = np.where(alltracked[:, i] > 0)[0]
        if (beg.size > 0 and end.size > 0):
            b = alltracked[beg[0]:end[-1] + 1, i]
            b[np.where(b > 0)] = 1
            fr[i] = np.where(np.diff(b) == -1)[0].size
    FRA = np.sum(fr)

    metrics = [recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA, MOTA, MOTP, MOTAL]

    additionalInfo = edict()
    additionalInfo.alltracked = alltracked
    additionalInfo.allfalsepos = allfalsepos
    additionalInfo.m = m
    additionalInfo.fp = fp
    additionalInfo.mme = mme
    additionalInfo.g = g
    additionalInfo.c = c
    additionalInfo.Fgt = Fgt
    additionalInfo.Ngt = Ngt
    additionalInfo.d = d
    additionalInfo.MT = MT
    additionalInfo.PT = PT
    additionalInfo.ML = ML
    additionalInfo.FRA = FRA
    additionalInfo.td = threshold
    return metrics, metricsInfo, additionalInfo


def IDmeasures(gtDB, stDB, threshold):
    # TODO: This function comes from https://github.com/shenh10/mot_evaluation/blob/master/utils/measurements.py
    # TO BE reimplemented
    """
    compute MTMC metrics
    [IDP, IDR, IDF1]
    """

    def corresponding_frame(traj1, len1, traj2, len2):
        """
        Find the matching position in traj2 regarding to traj1
        Assume both trajectories in ascending frame ID
        """
        p1, p2 = 0, 0
        loc = -1 * np.ones((len1,), dtype=int)
        while p1 < len1 and p2 < len2:
            if traj1[p1] < traj2[p2]:
                loc[p1] = -1
                p1 += 1
            elif traj1[p1] == traj2[p2]:
                loc[p1] = p2
                p1 += 1
                p2 += 1
            else:
                p2 += 1
        return loc

    def compute_distance(traj1, traj2, matched_pos):
        """
        Compute the loss hit in traj2 regarding to traj1
        """
        distance = np.zeros((len(matched_pos),), dtype=float)
        for i in range(len(matched_pos)):
            if matched_pos[i] == -1:
                continue
            else:
                iou = bbox_overlap(traj1[i, 2:6], traj2[matched_pos[i], 2:6])
                distance[i] = iou
        return distance

    def cost_between_trajectories(traj1, traj2, threshold):
        [npoints1, dim1] = traj1.shape
        [npoints2, dim2] = traj2.shape
        # find start and end frame of each trajectories
        start1 = traj1[0, 0]
        end1 = traj1[-1, 0]
        start2 = traj2[0, 0]
        end2 = traj2[-1, 0]

        ## check frame overlap
        has_overlap = max(start1, start2) < min(end1, end2)
        if not has_overlap:
            fn = npoints1
            fp = npoints2
            return fp, fn

        # gt trajectory mapping to st, check gt missed
        matched_pos1 = corresponding_frame(traj1[:, 0], npoints1, traj2[:, 0], npoints2)
        # st trajectory mapping to gt, check computed one false alarms
        matched_pos2 = corresponding_frame(traj2[:, 0], npoints2, traj1[:, 0], npoints1)
        dist1 = compute_distance(traj1, traj2, matched_pos1)
        dist2 = compute_distance(traj2, traj1, matched_pos2)
        # FN
        fn = sum([1 for i in range(npoints1) if dist1[i] < threshold])
        # FP
        fp = sum([1 for i in range(npoints2) if dist2[i] < threshold])
        return fp, fn

    def cost_between_gt_pred(groundtruth, prediction, threshold):
        n_gt = len(groundtruth)
        n_st = len(prediction)
        cost = np.zeros((n_gt, n_st), dtype=float)
        fp = np.zeros((n_gt, n_st), dtype=float)
        fn = np.zeros((n_gt, n_st), dtype=float)
        for i in range(n_gt):
            for j in range(n_st):
                fp[i, j], fn[i, j] = cost_between_trajectories(groundtruth[i], prediction[j], threshold)
                cost[i, j] = fp[i, j] + fn[i, j]
        return cost, fp, fn

    st_ids = np.unique(stDB[:, 1])
    gt_ids = np.unique(gtDB[:, 1])
    n_st = len(st_ids)
    n_gt = len(gt_ids)
    groundtruth = [gtDB[np.where(gtDB[:, 1] == gt_ids[i])[0], :] for i in range(n_gt)]
    prediction = [stDB[np.where(stDB[:, 1] == st_ids[i])[0], :] for i in range(n_st)]
    cost = np.zeros((n_gt + n_st, n_st + n_gt), dtype=float)
    cost[n_gt:, :n_st] = INF
    cost[:n_gt, n_st:] = INF

    fp = np.zeros(cost.shape)
    fn = np.zeros(cost.shape)
    # cost matrix of all trajectory pairs
    cost_block, fp_block, fn_block = cost_between_gt_pred(groundtruth, prediction, threshold)

    cost[:n_gt, :n_st] = cost_block
    fp[:n_gt, :n_st] = fp_block
    fn[:n_gt, :n_st] = fn_block

    # computed trajectory match no groundtruth trajectory, FP
    for i in range(n_st):
        cost[i + n_gt, i] = prediction[i].shape[0]
        fp[i + n_gt, i] = prediction[i].shape[0]

    # groundtruth trajectory match no computed trajectory, FN
    for i in range(n_gt):
        cost[i, i + n_st] = groundtruth[i].shape[0]
        fn[i, i + n_st] = groundtruth[i].shape[0]

    matched_indices = linear_sum_assignment(cost)

    nbox_gt = sum([groundtruth[i].shape[0] for i in range(n_gt)])
    nbox_st = sum([prediction[i].shape[0] for i in range(n_st)])

    IDFP = 0
    IDFN = 0
    for matched in zip(*matched_indices):
        IDFP += fp[matched[0], matched[1]]
        IDFN += fn[matched[0], matched[1]]
    IDTP = nbox_gt - IDFN
    assert IDTP == nbox_st - IDFP
    IDP = IDTP / (IDTP + IDFP) * 100  # IDP = IDTP / (IDTP + IDFP)
    IDR = IDTP / (IDTP + IDFN) * 100  # IDR = IDTP / (IDTP + IDFN)
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100  # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    measures = edict()
    measures.IDP = IDP
    measures.IDR = IDR
    measures.IDF1 = IDF1
    measures.numGT = nbox_gt
    measures.numPRED = nbox_st
    measures.IDTP = IDTP
    measures.IDFP = IDFP
    measures.IDFN = IDFN

    return measures


def evaluateBenchmark(allMets, world):
    MT, PT, ML, FRA, falsepositives, missed, idswitches, Fgt, distsum, \
    Ngt, sumg, Nc, numGT, numPRED, IDTP, IDFP, IDFN = (0,) * 17
    for ind in range(len(allMets)):
        if allMets[ind].m is None:
            print('Results missing for sequence #{}'.format(ind))
            continue
        numGT = numGT + allMets[ind].IDmeasures.numGT
        numPRED = numPRED + allMets[ind].IDmeasures.numPRED
        IDTP = IDTP + allMets[ind].IDmeasures.IDTP
        IDFN = IDFN + allMets[ind].IDmeasures.IDFN
        IDFP = IDFP + allMets[ind].IDmeasures.IDFP

        MT = MT + allMets[ind].additionalInfo.MT
        PT = PT + allMets[ind].additionalInfo.PT
        ML = ML + allMets[ind].additionalInfo.ML
        FRA = FRA + allMets[ind].additionalInfo.FRA
        Fgt = Fgt + allMets[ind].additionalInfo.Fgt
        Ngt = Ngt + allMets[ind].additionalInfo.Ngt
        Nc = Nc + np.sum(allMets[ind].additionalInfo.c)
        sumg = sumg + np.sum(allMets[ind].additionalInfo.g)
        falsepositives = falsepositives + np.sum(allMets[ind].additionalInfo.fp)
        missed = missed + np.sum(allMets[ind].additionalInfo.m)
        idswitches = idswitches + np.sum(allMets[ind].additionalInfo.mme)
        dists = allMets[ind].additionalInfo.d
        td = allMets[ind].additionalInfo.td
        distsum = distsum + np.sum(np.sum(dists))

    IDPrecision = IDTP / (IDTP + IDFP)
    IDRecall = IDTP / (IDTP + IDFN)
    IDF1 = 2 * IDTP / (numGT + numPRED)
    if numPRED == 0:
        IDPrecision = 0
    IDP = IDPrecision * 100
    IDR = IDRecall * 100
    IDF1 = IDF1 * 100

    FAR = falsepositives / Fgt
    MOTP = (1 - distsum / Nc) * 100
    if world:
        MOTP = MOTP / td
    if np.isnan(MOTP):
        MOTP = 0
    MOTAL = (1 - (missed + falsepositives + np.log10(idswitches + 1)) / sumg) * 100
    MOTA = (1 - (missed + falsepositives + idswitches) / sumg) * 100
    recall = Nc / sumg * 100
    precision = Nc / (falsepositives + Nc) * 100

    metsBenchmark = [IDF1, IDP, IDR, recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA,
                     MOTA, MOTP, MOTAL]

    return metsBenchmark


def evaluateTracking(seqmap, resDir, gtDataDir, benchmark):
    world = 0
    threshold = 0.5
    multicam = False

    if benchmark == 'MOT15':
        pass
    elif benchmark == 'MOT15_3D':
        world = 1
        threshold = 1
    elif benchmark == 'MOT16':
        pass
    elif benchmark == 'MOT17':
        pass
    elif benchmark == 'CVPR19':
        pass
    elif benchmark == 'PETS2017':
        pass
    elif benchmark == 'DukeMTMCT':
        multicam = True
    else:
        raise ValueError('Benchmark {} Not Implemented'.format(benchmark))

    sequenceListFile = os.path.join('seqmaps', seqmap)
    try:
        allSequences = parseSequences2(sequenceListFile)
    except:
        sequenceListFile = os.path.join('../seqmaps', seqmap)
        allSequences = parseSequences2(sequenceListFile)
    gtMat = []
    resMat = []

    allMets = []
    metsBenchmark = []
    MetsMultiCam = []

    for ind in range(len(allSequences)):

        if not multicam:
            sequenceName = allSequences[ind]
            sequenceFolder = os.path.join(gtDataDir, sequenceName)
            assert os.path.exists(sequenceFolder) and os.path.isdir(sequenceFolder), \
                'Sequence folder {} missing'.format(sequenceFolder)

            gtFilename = os.path.join(gtDataDir, sequenceName, 'gt', 'gt.txt')
            gtdata = np.loadtxt(gtFilename, delimiter=',')
            gtdata = gtdata[gtdata[:, 6] != 0]
            gtdata = gtdata[gtdata[:, 0] > 0]
            if benchmark == 'MOT16' or benchmark == 'MOT17' or benchmark == 'CVPR19':
                gtdata = gtdata[gtdata[:, 7] == 1]

            if benchmark == 'MOT15_3D':
                gtdata[:, 6:7] = gtdata[:, 7:8]

            _, ic = np.unique(gtdata[:, 1], return_inverse=True)
            gtdata[:, 1] = ic
            gtMat += [gtdata]

        else:
            raise NotImplementedError('Duke Format Not Implemented')

        if not multicam:

            resFilename = os.path.join(resDir, sequenceName + '.txt')
            if benchmark == 'MOT16' or benchmark == 'MOT17' or benchmark == 'CVPR19' or benchmark == 'MOT15':
                resFilename = preprocessResult(resFilename, sequenceName, gtDataDir)

            assert os.path.exists(resFilename), 'Invalid submission. Result for sequence {} not available!'.format(
                sequenceName)

            if os.path.exists(resFilename):
                if os.path.getsize(resFilename) > 0:
                    print(resFilename)
                    resdata = np.loadtxt(resFilename)
                else:
                    resdata = np.zeros((0, 9))

            resdata = resdata[resdata[:, 0] > 0, :]
            if benchmark == 'MOT15_3D':
                resdata[:, 6:8] = resdata[:, 7:9]

            resdata = resdata[resdata[:, 0] <= np.max(gtMat[ind][:, 0]), :]
            resMat += [resdata]
        else:
            raise NotImplementedError('Duke Format Not implemented')

        frameIdPairs = resMat[ind][:, 0:2]
        u, I = np.unique(frameIdPairs, return_index=True, axis=0)
        hasDuplicates = len(u) < len(frameIdPairs)
        assert not hasDuplicates, 'Invalid submission: Found duplicated ID/Frame pairs in sequence'

        metsCLEAR, mInf, additionalInfo = CLEAR_MOT_HUN(gtMat[ind], resMat[ind], threshold, world)
        metsID = IDmeasures(gtMat[ind], resMat[ind], threshold)
        mets = [metsID.IDF1, metsID.IDP, metsID.IDR] + metsCLEAR
        allMets.append(edict())
        allMets[ind].name = sequenceName
        allMets[ind].m = mets
        allMets[ind].IDmeasures = metsID
        allMets[ind].additionalInfo = additionalInfo
        evalFile = os.path.join(resDir, 'eval_{}.txt'.format(sequenceName))
        print(sequenceName)
        print('IDF1\tIDP\t\tIDR|\tRcll\tPrcn\tFAR|\tGT\tMT\tPT\tML|\tFP\t\tFN\t\tIDs\tFM|\t\tMOTA\tMOTP\tMOTAL')
        print('%.1f\t%.1f\t%.1f|\t%.1f\t%.1f\t%.1f|\t%d\t%d\t%d\t%d|\t%d\t%d\t%d\t%d|\t%.1f\t%.1f\t%.1f' %
              (mets[0], mets[1], mets[2], mets[3], mets[4], mets[5], mets[6], mets[7], mets[8], mets[9], mets[10],
               mets[11], mets[12], mets[13], mets[14], mets[15], mets[16]))

        # print(mets)
        # np.savetxt(evalFile, mets)

    metsBenchmark = evaluateBenchmark(allMets, world)
    evalFile = os.path.join(resDir, 'eval.txt')
    np.savetxt(evalFile, metsBenchmark)

    if multicam:
        raise NotImplementedError('Duke Format Not Implemented')

    return allMets, metsBenchmark, MetsMultiCam


# seq_map = 'MOT17_train.txt'
# evaluateTracking(seq_map, os.path.join('output', 'res'), os.path.join('/ssd/ssd0/MOT17', 'train'), 'MOT17')
# evaluateTracking('MOT17_train.txt', '/home/pb/3DTracking/MatchingRes/', '/ssd/ssd0/MOT17/train/', 'MOT17')
