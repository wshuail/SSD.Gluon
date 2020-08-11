import os
import sys
import time
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.block import SymbolBlock


__all__ = ['SSDDetector']


def change_nms_attr(sym, nms_thresh, nms_topk, force_suppress):
    for layer in sym.get_internals():
        if 'nms' in layer.name:
            layer._set_attr(overlap_thresh=str(nms_thresh))
            layer._set_attr(topk=str(nms_topk))
            layer._set_attr(force_suppress=str(force_suppress))
    return sym


class CustomSymbolBlock(SymbolBlock):
    @staticmethod
    def imports(symbol_file, input_names, param_file=None, ctx=None,
                nms_thresh=None, nms_topk=400, force_suppress=False):
        sym = mx.symbol.load(symbol_file)
        if nms_thresh:
            sym = change_nms_attr(sym, nms_thresh, nms_topk, force_suppress)
        if isinstance(input_names, str):
            input_names = [input_names]
        inputs = [mx.symbol.var(i) for i in input_names]
        ret = SymbolBlock(sym, inputs)
        if param_file is not None:
            ret.collect_params().load(param_file, ctx=ctx)
        return ret


def rbox2quad(xmin, ymin, xmax, ymax, deg):
    xc = (xmin + xmax) *0.5
    yc = (ymin + ymax) *0.5
    src = np.array([(xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax)], dtype=np.float32)
    src[:,0] -= xc
    src[:,1] -= yc

    theta = -deg /180 *np.pi
    mat_rot = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
    pts = np.dot(mat_rot, src.T).T

    pts[:, 0] += xc
    pts[:, 1] += yc
    return pts

def quad2rbox(q):
    def d_p2p(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.sqrt(dx*dx + dy*dy)

    def d_p2l(p0, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.abs(dy*p0[0] - dx*p0[1] + p2[0]*p1[1] - p2[1]*p1[0]) / np.sqrt(dx*dx + dy*dy)

    xc = (q[0][0] + q[1][0] + q[2][0] + q[3][0])/4
    yc = (q[0][1] + q[1][1] + q[2][1] + q[3][1])/4
    if d_p2p(q[0], q[1]) > d_p2p(q[0],q[3]):
        w = (d_p2p(q[0], q[1]) + d_p2p(q[2], q[3]))/2
        h = d_p2l((xc,yc), q[0], q[1]) + d_p2l((xc,yc), q[2], q[3])
        theta = np.arctan2(q[1][1]-q[0][1], q[1][0]-q[0][0]) *180 / np.pi
    else:
        w = d_p2l((xc,yc), q[0], q[3]) + d_p2l((xc,yc), q[1], q[2])
        h = (d_p2p(q[0], q[3]) + d_p2p(q[1], q[2]))/2
        theta = np.arctan2(q[3][1]-q[0][1], q[3][0]-q[0][0]) *180 / np.pi -90
    return (xc-w/2, yc-h/2, xc+w/2, yc+h/2, theta)


class SSDDetector(BaseDetector):
    def __init__(self, params_file, input_size=320,
                 gpu_id=0, nms_thresh=None, nms_topk=400,
                 force_suppress=False):
        super(JsonDetector, self).__init__()
        if isinstance(input_size, int):
            self.width, self.height = input_size, input_size
        elif isinstance(input_size, (list, tuple)):
            self.width, self.height = input_size
        else:
            raise ValueError('Expected int or tuple for input size')
        self.ctx = mx.gpu(gpu_id)

        self.transform_fn = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        symbol_file = params_file[:params_file.rfind('-')] + "-symbol.json"
        # self.net = gluon.nn.SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        self.net = CustomSymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx,
                                             nms_thresh=nms_thresh, nms_topk=nms_topk,
                                             force_suppress=force_suppress)
        self.net.hybridize()

    def detect(self, imgs, conf_thresh=0.5, batch_size=4):

        # self.net.set_nms(nms_thresh=nms_thresh, nms_topk=400)

        num_example = len(imgs)

        all_detections = []

        t0 = time.time()
        for i in range(0, num_example, batch_size):
            batch_raw_imgs = imgs[i: min(i+batch_size, num_example)]
            orig_sizes = []
            batch_img_lst = []
            for img in batch_raw_imgs:
                orig_sizes.append(img.shape)
                if not isinstance(img, mx.nd.NDArray):
                    img = mx.nd.array(img)
                img = self.transform_fn(img)
                batch_img_lst.append(img)
            batch_img = mx.nd.stack(*batch_img_lst)
            batch_img = batch_img.as_in_context(self.ctx)
            mx.nd.waitall()
            t1 = time.time()

            outputs = self.net(batch_img)
            assert len(outputs) == 3, 'Expected length of outputs == 3'
            ids, scores, bboxes = [outputs[i].asnumpy() for i in range(len(outputs))]
            t2 = time.time()

            for img_idx in range(len(batch_img_lst)):
                img_ids = ids[img_idx].flatten()
                img_scores = scores[img_idx].flatten()
                img_bboxes = bboxes[img_idx]

                x_scale = orig_sizes[img_idx][1] / float(self.width)
                y_scale = orig_sizes[img_idx][0] / float(self.height)

                positive_idx = (img_scores >= conf_thresh)

                positive_img_ids = img_ids[positive_idx]
                positive_img_scores = img_scores[positive_idx]
                positive_img_bboxes = img_bboxes[positive_idx, :]

                img_detection = []
                for box_idx in range(len(positive_img_ids)):
                    # rescale bbox
                    bbox = positive_img_bboxes[box_idx, :].tolist()
                    if len(bbox) == 5:
                        if abs(x_scale - y_scale) > 0.1:
                            quad = rbox2quad(*bbox)
                            quad[:, 0] *= x_scale
                            quad[:, 1] *= y_scale
                            rbox = quad2rbox(quad)
                            deg = rbox[4]
                            bbox = list(rbox[:4])
                        else:
                            deg = bbox[4]
                            bbox = [x * x_scale for x in bbox[:4]]
                    elif len(bbox) == 4:
                        bbox[0] *= x_scale
                        bbox[1] *= y_scale
                        bbox[2] *= x_scale
                        bbox[3] *= y_scale
                        deg = 0
                    else:
                        raise NotImplementedError('Expected bbox with length 4 or 5, got %d' % len(bbox))

                    img_info = {}
                    img_info['class'] = int(positive_img_ids[box_idx])
                    img_info['score'] = positive_img_scores[box_idx]
                    img_info['bbox'] = bbox
                    img_info['degree'] = float(deg)
                    img_detection.append(img_info)
                all_detections.append(img_detection)
            t3 = time.time()
            logging.info('batch-size: {} preparation: {:.3f} ms, forward: {:.3f} ms, post: {:.3f} ms'.format(batch_img.shape, (t1-t0)*1000, (t2-t1)*1000,(t3-t2)*1000))
            t0 = time.time()

        return all_detections



