# PyTorch-SSD

[//]: # (References)
[ssd-pytorch-repo]: https://github.com/amdegroot/ssd.pytorch
[cudnn]: https://developer.nvidia.com/cudnn
[cuda]: https://developer.nvidia.com/cuda-downloads
[pytorch-install]: https://pytorch.org/
[stopiteration-fix]: https://github.com/amdegroot/ssd.pytorch/issues/214#issuecomment-409851395
[aws-login]: https://console.aws.amazon.com/
[aws-dlami-guide]: https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html
[aws-spot-instance]: ./imgs/aws-dlami-pytorch.jpg
[activate-pytorch-env]: ./imgs/activate-pytorch-env.jpg
[installation]: #installation
[anaconda-dist]: https://www.anaconda.com/download/
[my-ssd]: https://ln.sync.com/dl/74a3bbef0/8njcymdw-wm37r4u9-idyfiu3c-8uu2y2ss
[horse-detection]: ./videos/epic-horses-detected.mp4
[dog-detection]: ./videos/funny_dog-detected.mp4

---

All code was taken from Max deGroot's & Ellis Brown's [ssd.pytorch repository][ssd-pytorch-repo] except the `object_detection.py` file. However, some modifications were done in order to make this project run on Windows 10 and Python 3.6 with PyTorch 0.4.1 for CUDA 9.2.

| Local Machine Specs            |
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

3. Install OpenCV, NumPy & imageio by executing the following line in your Terminal/Command Prompt:
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

In `layers/functions/detection.py` line 62 was changed from:
```python
if scores.dim() == 0:
    continue
```
to:
```python
if scores.size(0) == 0:
    continue
```

## Training

### Local machine
If you are training on a Windows machine make sure to set the value of the `--num_workers` flag to `0` or you will get a `BrokenPipeError: [Errno 32] Broken pipe` error. On my machine, I also need to close all programs (except the Command Prompt of course) and set the batch size to 2 as well as the learning rate to 0.000006 in order to train the model otherwise I get a `RuntimeError: CUDA error: out of memory` error.

```sh
python train.py --num_workers 0 --batch_size 2 --lr 1e-6
```

### AWS spot instance
Since training on my local machine with the settings/flags above would take days (or even weeks) to get reasonable results I decided to train the SSD on an AWS spot instance.

To set up an AWS spot instance do the following steps:

1. [Login to your Amazon AWS Account][aws-login]
2. Navigate to **EC2 > Instances > Spot Requests > Request Spot Instances**
3. Under `AMI` click on `Search for AMI`, type `AWS Deep Learning AMI` in the search field, choose `Community AMIs` from the drop-down and select the `Deep Learning AMI (Ubuntu) Version 14.0`
3. Delete the default instance type, click on Select and select the p2.xlarge instance
4. Uncheck the `Delete` checkbox under EBS Volumes so your progress is not deleted when the instance gets terminated
5. Set Security Groups to default
6. Select your key pair under Key pair name (if you don't have one create a new key pair)
7. At the very bottom set `Request valid until` to about 10 - 12 hours and set `Terminate instances at expiration` as checked (You don't have to do this but keep in mind to receive a very large bill from AWS if you forget to terminate your spot instance because the default value for termination is set to 1 year.)
8. Click `Launch`, wait until the instance is created and then connect to your instance via ssh

There's also a detailed explanation from AWS about [AWS Deep Learning AMIs][aws-dlami-guide]. You might give it a shot as well.

![aws-spot-instance][aws-spot-instance]
_Spot instance setup_

When your spot instance is up and running AND you have connected to your spot instance you then need to activate the PyTorch environment like so:
```sh
source activate pytorch_p36
```

![activate-pytorch-env][activate-pytorch-env]
*Activate PyTorch environment on spot instance*

Lastly, clone this repository, [proceed with the installation process][installation] (except for PyTorch) and start training by executing: 
```sh
## you probably don't need to add any arguments here
python train.py
```

If you don't want to train an SSD model and want to try the detection only you can [download my trained SSD model][my-ssd]. I've trained the model with all default values/parameters from the original repository but stopped the training after 1500 iterations because the loss stagnated.

## Detection in a video

To detect objects in a video you first need to install `ffmpeg` by executing the following line:
```sh
conda install ffmpeg -c conda-forge
```

Note: This command only works if you have the [Anaconda Distribution][anaconda-dist] installed on your computer.


After you have trained the SSD model and you want to detect objects in a video execute the following line in your Terminal/Command Prompt.

```python
python object_detection.py path_to/your_ssd_model.pth path_to/your_video.mp4 -o path_to/name_of_your_output_video.mp4
```

If the `-o` flag is not specified the output video will simply have the name `output.mp4`

You can watch sample outputs from here:
* [Horse detection][horse-detection]
* [Person & dog detection][dog-detection]