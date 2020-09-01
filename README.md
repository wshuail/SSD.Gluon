# SSD
Minimal SSD implemented with MXNet/Gluon and nvidia DALI.  

## train
```
python train.py --cfg ./cfgs/ssd_mobilenet_1.0_coco_512x512.yaml
```
## requirement
- MXNet==1.5.0
- nvidia-dali==0.23.0  

## Performance
### COCO: 
backbone | input_size | mAP(0.5:0.95) 
--|:--|:------
mobilenet1.0 | 512 | 24.2 

more models will be trained if I have enough GPU ...

## Demo
![000000132116.jpg](samples/000000132116.jpg)
![000000321333.jpg](samples/000000321333.jpg)
![000000335328.jpg](samples/000000335328.jpg)
![000000346905.jpg](samples/000000346905.jpg)
![000000415238.jpg](samples/000000415238.jpg)

## Reference
- https://github.com/dmlc/gluon-cv
- https://github.com/NVIDIA/retinanet-examples