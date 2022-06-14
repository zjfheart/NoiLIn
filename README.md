# NoiLIn
The code for NoiLIn: Improving Adversarial Training and Correcting Stereotype of Noisy Labels (TMLR 22 accept)

## Requirements
+ Python (3.8)
+ PyTorch (1.11)
+ CUDA (11.3)
+ [AutoAttack](https://github.com/fra31/auto-attack) (This is a package for implementing AutoAttack. You can install it via ```pip install git+https://github.com/fra31/auto-attack```)
+ Numpy

## Conducting experiments
(1) SAT-NoiLIn/TRADES-NoiLIn using ResNet18/Wide ResNet on CIFAR10/CIAFR100/SVHN
```
python SAT-NoiLIn.py --dataset='dataset' --noise_type='noise_type' --net='net_name'
python TRADES-NoiLIn.py --dataset='dataset' --noise_type='noise_type' --net='net_name'
```

(2) TRADES-AWP-NoiLIn using WRN-34-10 on CIFAR10
```
cd TRADES-AWP-NoiLIn
python TRADES-AWP-NoiLIn.py
```

(3) SAT-NoiLIn with extra training data using WRN-28-10 on CIFAR10 <br/>
You need to download ```ti_500K_pseudo_labeled.pickle``` which contains 500K pseudo-labeled TinyImages from this [link](https://drive.google.com/file/d/1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi/view) (Auxillary data provided by Carmon et al. 2019). Then, store ```ti_500K_pseudo_labeled.pickle``` into the folder ```./data```. In addition, we recommend to using mutilple GPUs.

```
cd NoiLIn_ExtraData
python SAT-NoiLIn-ExtraData.py --gpu='0,1,2,3' --aux_data_filename='ti_500K_pseudo_labeled.pickle'
```

(4) Evaluate the performance of trained models <br/>
(4.1) Obtain the learning curve of natural and robust accuracy
``` 
python eval.py --all_epoch --start_epoch=1 --end_epoch=120 --model_dir='model_dir'
```
(4.2) Obtain natural and robust accuracy of a given model 
```
python eval.py --model_dir='model_dir' --pt_name='model_pt_name'
```
## Reference
```
@article{zhang2021noilin,
  title={NoiLIn: Do Noisy Labels Always Hurt Adversarial Training?},
  author={Zhang, Jingfeng and Xu, Xilie and Han, Bo and Liu, Tongliang and Niu, Gang and Cui, Lizhen and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:2105.14676},
  year={2021}
}
```

## Contact
Please contact jingfeng.zhang@riken.jp and xuxilie@comp.nus.edu.sg if you have any questions on the codes.

