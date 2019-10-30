import numpy as np
from pathlib import Path
from SlideRunner.dataAccess.database import Database
import openslide

from random import randint

from lib.object_detection_helper import *
from PIL import ImageFile


from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.data_block import *

class SlideContainer():

    def __init__(self, file: Path, annotations:dict, y, level: int=0, width: int=256, height: int=256, sample_func: callable=None):
        self.file = file
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.y = y
        self.annotations = annotations
        self.sample_func = sample_func
        self.classes = list(set(self.y[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int=0, y: int=0):
             return np.array(self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)),
                                          level=self.level, size=(self.width, self.height)))[:, :, :3]


    @property
    def shape(self):
        return (self.width, self.height)

    def __str__(self):
        return 'SlideContainer with:\n sample func: '+str(self.sample_func)+'\n slide:'+str(self.file)

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "annotations" : self.annotations,
                                               "level": self.level, "container" : self})

        # use default sampling method
        class_id = np.random.choice(self.classes, 1)[0]
        ids = self.y[1] == class_id
        xmin, ymin, _, _ = np.array(self.y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
        return int(xmin - self.shape / 2), int(ymin - self.height / 2)

def bb_pad_collate_min(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    samples = [s for s in samples if s[1].data[0].shape[0] > 0] # check that labels are available

    max_len = max([len(s[1].data[1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        bbs, lbls = s[1].data
        bboxes[i,-len(lbls):] = bbs
        labels[i,-len(lbls):] = torch.from_numpy(lbls)
    return torch.cat(imgs,0), (bboxes,labels)

class SlideLabelList(LabelList):


    def __getitem__(self,idxs:Union[int,np.ndarray])->'LabelList':
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            if self.item is None:
                slide_container = self.x.items[idxs]

                xmin, ymin = slide_container.get_new_train_coordinates()

                x = self.x.get(idxs, xmin, ymin)
                y = self.y.get(idxs, xmin, ymin)
            else:
                x,y = self.item ,0
            if self.tfms or self.tfmargs:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            if y is None: y=0
            return x,y
        else:
            return self.new(self.x[idxs], self.y[idxs])



PreProcessors = Union[PreProcessor, Collection[PreProcessor]]
fastai_types[PreProcessors] = 'PreProcessors'

class SlideItemList(ItemList):

    def __init__(self, items:Iterator, path:PathOrStr='.', label_cls:Callable=None, inner_df:Any=None,
                 processor:PreProcessors=None, x:'ItemList'=None, ignore_empty:bool=False):
        self.path = Path(path)
        self.num_parts = len(self.path.parts)
        self.items,self.x,self.ignore_empty = items,x,ignore_empty
        self.sizes = [None] * len(self.items)
        if not isinstance(self.items,np.ndarray): self.items = array(self.items, dtype=object)
        self.label_cls,self.inner_df,self.processor = ifnone(label_cls,self._label_cls),inner_df,processor
        self._label_list,self._split = SlideLabelList,ItemLists
        self.copy_new = ['x', 'label_cls', 'path']

    def __getitem__(self,idxs: int, x: int=0, y: int=0)->Any:
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            return self.get(idxs, x, y)
        else:
            return self.get(*idxs)

    def label_from_list(self, labels:Iterator, label_cls:Callable=None, **kwargs)->'LabelList':
        "Label `self.items` with `labels`."
        labels = array(labels, dtype=object)
        label_cls = self.get_label_cls(labels, label_cls=label_cls, **kwargs)
        y = label_cls(labels, path=self.path, **kwargs)
        res = SlideLabelList(x=self, y=y)
        return res


class SlideImageItemList(SlideItemList):
    pass

class SlideObjectItemList(SlideImageItemList, ImageList):

    def get(self, i, x: int, y: int):
        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res

class ObjectItemListSlide(SlideObjectItemList):

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        return Image(pil2tensor(fn.get_patch(x, y) / 255., np.float32))


class SlideObjectCategoryList(ObjectCategoryList):

    def get(self, i, x: int=0, y: int=0):
        h, w = self.x.items[i].shape
        bboxes, labels = self.items[i]
        if x > 0 and y > 0:
            bboxes = np.array(bboxes)
            labels = np.array(labels)

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

            bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
            bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

            ids = ((bboxes[:, 0] + bb_widths) > 0) \
                  & ((bboxes[:, 1] + bb_heights) > 0) \
                  & ((bboxes[:, 2] - bb_widths) < w) \
                  & ((bboxes[:, 3] - bb_heights) < h)

            bboxes = bboxes[ids]
            bboxes = np.clip(bboxes, 0, x)
            bboxes = bboxes[:, [1, 0, 3, 2]]

            labels = labels[ids]
            if len(labels) == 0:
                labels = np.array([0])
                bboxes = np.array([[0, 0, 1, 1]])

            return ImageBBox.create(h, w, bboxes, labels, classes=self.classes, pad_idx=self.pad_idx)
        else:
            return ImageBBox.create(h, w, bboxes[:10], labels[:10], classes=self.classes, pad_idx=self.pad_idx)


def slide_object_result(learn: Learner, anchors, detect_thresh:float=0.2, nms_thresh: float=0.3,  image_count: int=5):
    with torch.no_grad():
        img_batch, target_batch = learn.data.one_batch(DatasetType.Train, False, False, False)
        prediction_batch = learn.model(img_batch)
        class_pred_batch, bbox_pred_batch = prediction_batch[:2]
        regression_pred_batch = prediction_batch[3].view(-1) if len(prediction_batch) > 3 \
            else [None] * class_pred_batch.shape[0]
        bbox_regression_pred_batch = prediction_batch[4] if len(prediction_batch) > 4 \
            else [None] * bbox_pred_batch.shape[0]

        bbox_gt_batch, class_gt_batch = target_batch

        for img, bbox_gt, class_gt, clas_pred, bbox_pred, reg_pred, box_reg_pred in \
                list(zip(img_batch, bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch,
                         regression_pred_batch, bbox_regression_pred_batch))[:image_count]:
            img = Image(learn.data.denorm(img))

            out = process_output(clas_pred, bbox_pred, anchors, detect_thresh)
            bbox_pred, scores, preds = [out[k] for k in ['bbox_pred', 'scores', 'preds']]
            if bbox_pred is not None:
                to_keep = nms(bbox_pred, scores, nms_thresh)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()
                box_reg_pred = box_reg_pred[to_keep].cpu() if box_reg_pred is not None else None

            t_sz = torch.Tensor([*img.size])[None].cpu()
            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0] - 1
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))
            if bbox_pred is not None:
                bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
                # change from center to top left
                bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2

            pred_score_classes = f'{np.mean(to_np(preds)):.2f}' if preds is not None else '0.0'
            pred_score_classes_reg = f'{np.mean(to_np(box_reg_pred)):.2f}' if box_reg_pred is not None else '0.0'
            gt_score = f'{np.mean(to_np(class_gt)):.2f}' if class_gt.shape[0] > 0 else '0.0'

            pred_score = '' if reg_pred is None else f'Box:{pred_score_classes} \n Reg:{to_np(reg_pred):.2f}'

            if box_reg_pred is None:
                show_results(img, bbox_pred, preds, scores, list(range(0, learn.data.c))
                             , bbox_gt, class_gt, (15, 3), titleA=str(gt_score), titleB=str(pred_score), titleC='CAM', clas_pred=clas_pred, anchors=anchors)
            else:
                pred_score_reg = f'BoxReg:{pred_score_classes_reg} \n Reg:{to_np(reg_pred):.2f}'

                show_results_with_breg(img, bbox_pred, preds, box_reg_pred, scores, list(range(0, learn.data.c))
                                       , bbox_gt, class_gt, (15, 15), titleA=str(gt_score), titleB=str(pred_score),
                                       titleC=pred_score_reg)


def show_results_with_breg(img, bbox_pred, preds, scores, breg_pred, classes, bbox_gt, preds_gt, figsize=(5,5)
                 , titleA: str="", titleB: str="", titleC: str=""):

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    ax[0].set_title(titleA)
    ax[1].set_title(titleB)
    ax[2].set_title(titleC)

    # show gt
    img.show(ax=ax[0])
    for bbox, c in zip(bbox_gt, preds_gt):
        txt = str(c.item()) if classes is None else classes[c.item()]
        draw_rect(ax[0], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt}')

    # show prediction class
    img.show(ax=ax[1])
    if bbox_pred is not None:
        for bbox, c, scr in zip(bbox_pred, preds, scores):
            txt = str(c.item()) if classes is None else classes[c.item()]
            draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr.item():.1f}')

    # show prediction class
    img.show(ax=ax[2])
    if bbox_pred is not None:
        for bbox, c in zip(bbox_pred, breg_pred):
            draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{c.item():.1f}')

