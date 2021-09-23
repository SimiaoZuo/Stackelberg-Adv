# SALT: Stackelberg Adversarial Regularization

Code for [Adversarial Regularization as Stackelberg Game: An Unrolled Optimization Approach](https://arxiv.org/abs/2104.04886), 
EMNLP 2021.

## Run the code

### Dependency
* The most convenient way to run the code is to use this docker image: `tartarusz/adv-train:azure-pytorch-apex-v1.7.0`. 
  The image supports running on Microsoft Azure.
* We use the [Higher](https://github.com/facebookresearch/higher) package to implement unrolling.
  Use `pip install higher` to install the package.
* Our implementation is modified from the [Fairseq](https://github.com/pytorch/fairseq) code base.
  
### Instructions
* Please refer to the [Fairseq examples](https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md)
for dataset pre-processing.
* Use `bash run.sh` to run the code. 

### Note
* The major modification from the original Fairseq code base is the following.
  * `fairseq/criterions/adv_unroll_loss.py` is the main file that handles Stackelberg adversarial regularization.
  * `fairseq/models/transformer.py` modifies embedding to include adversarial perturbations.
  * `fairseq/tasks/fairseq_task.py` contains the adversarial training procedure.
* There are many variants of Stackelberg adversarial regularization. For example,
the projection step after updating the adversarial perturbations may be removed,
if the initialization and the inner learning rate are carefully chosen. 


## Reference

Please cite the following paper if you use this code.

```
@article{zuo2021adversarial,
  title={Adversarial Training as Stackelberg Game: An Unrolled Optimization Approach},
  author={Zuo, Simiao and Liang, Chen and Jiang, Haoming and Liu, Xiaodong and He, Pengcheng and Gao, Jianfeng and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2104.04886},
  year={2021}
}
```