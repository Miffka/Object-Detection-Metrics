from copy import deepcopy

import tqdm

from .BoundingBox import BoundingBox
from .BoundingBoxes import BoundingBoxes
from .Evaluator import Evaluator
from .utils import CoordinatesType, BBFormat, BBType, MethodAveragePrecision


def getBoundingBoxes(
    df,
    isGT,
    bbFormat=BBFormat.XYX2Y2,
    coordType=CoordinatesType.Absolute,
    allBoundingBoxes=None,
    allClasses=None,
    imgSize=(0, 0),
):
    """
    Read dataframe containing bounding boxes (ground truth and detections).

    Required columns are: 
        image_id
        class_id
        x_min
        y_min
        x_max
        y_max
        im_h
        im_w
        scores (if isGT is 'False')
    """
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read GT detections from dataframe
    for row_idx, row in tqdm.tqdm(df.iterrows(), desc="Convert df to boxes", total=df.shape[0]):
        classId = row["class_id"]
        imgSize = (row["im_h"], row["im_w"])
        bbType = BBType.GroundTruth if isGT else BBType.Detected
        confidence = None if isGT else row["scores"]

        bb = BoundingBox(
            imageName=row["image_id"],
            classId=classId,
            x=row["x_min"],
            y=row["y_min"],
            w=row["x_max"],
            h=row["y_max"],
            typeCoordinates=coordType,
            imgSize=imgSize,
            bbType=bbType,
            classConfidence=confidence,
            format=bbFormat,
        )
        allBoundingBoxes.addBoundingBox(bb)
        if classId not in allClasses:
            allClasses.append(classId)
    return allBoundingBoxes, allClasses


def calculate_map(allBoundingBoxes, allClasses, category_id_to_name, iouThreshold=0.4):
    evaluator = Evaluator()

    detections = evaluator.GetPascalVOCMetrics(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
    )

    acc_AP = 0
    validClasses = 0
    allPositives = 0
    precision_dict = {}

    for metricsPerClass in detections:
        # Get metric values per each class
        class_id = metricsPerClass["class"]
        ap = metricsPerClass["AP"]
        totalPositives = metricsPerClass["total positives"]
        precision_dict[category_id_to_name[class_id]] = {
            "cat_id": class_id,
            "n_objs": totalPositives,
            "prec": ap,
        }

        if totalPositives > 0:
            validClasses += 1
            acc_AP += ap

    mAP = acc_AP / validClasses
    precision_dict["total"] = {
        "cat_id": None,
        "n_objs": allPositives,
        "prec": mAP,
    }

    return precision_dict


def calculate_map_from_dfs(
    df_pred, category_id_to_name, df_gt=None, gtBoundingBoxes=None, gtClasses=None, iouThreshold=0.4
):
    if gtBoundingBoxes is None:
        allBoundingBoxes, allClasses = getBoundingBoxes(df_gt, isGT=True)
    else:
        allBoundingBoxes = deepcopy(gtBoundingBoxes)
        allClasses = deepcopy(gtClasses)

    allBoundingBoxes, allClasses = getBoundingBoxes(
        df_pred, isGT=False, allBoundingBoxes=allBoundingBoxes, allClasses=allClasses
    )
    allClasses.sort()

    precision_dict = calculate_map(
        allBoundingBoxes, allClasses, category_id_to_name=category_id_to_name, iouThreshold=iouThreshold
    )

    return precision_dict
