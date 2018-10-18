## PyTorch-SSD

[//]: # (References)
[ssd-pytorch-repo]: https://github.com/amdegroot/ssd.pytorch
[cudnn]: https://developer.nvidia.com/cudnn
[cuda]: https://developer.nvidia.com/cuda-downloads
[pytorch-install]: https://pytorch.org/
[stopiteration-fix]: https://github.com/amdegroot/ssd.pytorch/issues/214#issuecomment-409851395

---

All code was taken from Max deGroot's & Ellis Brown's [ssd.pytorch repository][ssd-pytorch-repo] except the `object_detection.py` file. However, some modifications were done in order to make this project run on Windows 10 and Python 3.6 with PyTorch 0.4.1 for CUDA 9.2.

| Specs                          |
| :----------------------------- |
| Windows 10                     |
| NVIDIA GeForce GTX 850M        |
| [CUDA 10.0 (Download)][cuda]   |
| [cuDNN 10.0 (Download)][cudnn] |

Even though PyTorch 0.4.1 for CUDA 9.2 was installed, the library also works for CUDA 10.0.

## Installation 

**Before you start [please refer to the original repository][ssd-pytorch-repo] on how to use this code properly.** 

Basically, what you will need to do is:
1. Download the datasets and the pretrained VGG-16 base network (both described in the original repository).
   
2. [Install PyTorch by visiting the website and choosing your specifications.][pytorch-install]

3. Install OpenCV, NumPy & imageio by execute the following line in your Terminal/Command Prompt:
    ```sh
    pip install -r requirements.txt
    ```

## Modifications for PyTorch 0.4.1 for CUDA 9.2

The following modifications has been made to successfully execute `train.py`:

In `train.py` line 203 (line 165 in the original repo) was changed from:
```python
images, targets = next(batch_iterator)
```
to:
```python
try:
    images, targets = next(batch_iterator)
except StopIteration:
    batch_iterator = iter(data_loader)
    images, targets = next(batch_iterator)
```
The fix was copied from [this comment][stopiteration-fix]. 

Fixed naming of the saved model in `train.py` on line 239 & 244 (line 196 & 198).

In `layers/modules/multibox_loss.py` add `loss_c = loss_c.view(pos.size()[0], pos.size()[1])` on line 97 like so:
```python
# Hard Negative Mining
loss_c = loss_c.view(pos.size()[0], pos.size()[1])
loss_c[pos] = 0  # filter out pos boxes for now
loss_c = loss_c.view(num, -1)
```
and then change `N = num_pos.data.sum()` to `N = num_pos.data.sum().float()` on line 115.

## Training

If you are training on a Windows machine make sure to set the value of the `--num_workers` flag to `0` or you will get a `BrokenPipeError: [Errno 32] Broken pipe` error. On my machine, I also need to close all programs (except the Command Prompt of course) and set the batch size to 2 as well as the learning rate to 0.000006 in order to train the model otherwise I get a `RuntimeError: CUDA error: out of memory` error.

```sh
python train.py --num_workers 0 --batch_size 2 --lr 1e-6
```

TODO: train on EC2 instance with Deep Learning AMI (Ubuntu) Version 14.0

## Detection in a video

After you have trained the SSD model and you want to detect objects in a video execute the following line in your Terminal/Command Prompt.

```python
python object_detection.py path_to/your_ssd_model.pth path_to/your_video.mp4 -o path_to/name_of_your_output_video.mp4
```

If the `-o` flag is not specified the output video will simply have the name `output.mp4`