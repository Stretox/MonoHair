
# MonoHair:High-Fidelity Hair Modeling from a Monocular Video  <font color='red'> ( CVPR 2024 Oral ) </font> [[Projectpage](https://keyuwu-cs.github.io/MonoHair/ "Projectpage")] #
This repository was slightly modified by me @Stretox as part of a student project. Give a monocular video, MonoHair reconstruct a high-fidelity 3D strand model. 

The code remains mostly unchainged with minor bugfixes and a hughly better Tutorial on how you can generate your own results!

If you want to get started please go and check the Section [Run](#Run)

This repository also includes some examples for reconstructing 3D hair from a monocular video or multi-view images.

Noteworthy Mentions

- A 3D avatar is genrated using flame template and fitting the flame coarse geometry using multiview images (only for real human capture). For more details please check [DELTA](https://github.com/yfeng95/DELTA "Delta") 
- For coarse goemtry initialization, please check [Instant-NGP](https://github.com/NVlabs/instant-ngp "Instant-NGP").
- For hair exterior geometry synthesis, a patch-based multi-view optimization (PMVO) method is being proposed. Please check the [Paper](https://arxiv.org/abs/2403.18356 "MonoHair").
- For hair interior inference, please check [DeepMVSHair](https://dl.acm.org/doi/abs/10.1145/3550469.3555385 "DeepMVSHair"). <- Really good! Go and read it
- For strand generation, please also check the [Paper](https://arxiv.org/abs/2403.18356 "MonoHair").

![](fig/results.png)

# Build

## Getting Started ##
Clone the repository and install requirements:

    git clone https://github.com/Stretox/MonoHair.git --recursive
	cd MonoHair
	conda create -n MonoHair python==3.10.12
	conda activate MonoHair
	pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
	pip install -r requirements.txt
	

## Dependencies and submodules ##

Install [Pytorch](https://pytorch.org/ "torch"), [Pytorch3d](https://github.com/facebookresearch/pytorch3d) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). We have tested on Ubuntu 22.04.4, gcc==9.5, python==3.10.12 pytorch==1.11.0, pytorch3d==0.7.2 with CUDA 11.3 on RTX 3090Ti. You can install any version that is compatible with these dependencies. We know torch==1.3.0 have some bug when employing [MODNet](https://github.com/ZHKKKe/MODNet "MODNet").

	# if have problem when install pytorch 3d, you can try to install fvcore: pip install  fvcore==0.1.5.post20220512
	pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/download.html
	

Initialize submodules of [Instant-NGP](https://github.com/NVlabs/instant-ngp "Instant-NGP"), [MODNet](https://github.com/ZHKKKe/MODNet "MODNet"), [CDGNet](https://github.com/tjpulkl/CDGNet "CDGNet"), [DELTA](https://github.com/yfeng95/DELTA "DELTA") and [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch "face-parsing"). 

    git submodule update --init --recursive
	pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

There are common issues with tiny-cuda-nn and pytorch3D if you are using a different Cuda Version than the one recommendet. These could be solved at least on Cuda 12.4. If you are stuck, you can make an Issue.

Compile Instant-NGP and move our modified [run.py](https://github.com/Stretox/MonoHair/blob/master/run.py) to instant-ngp/scripts.

	cp run.py submodules/instant-ngp/scripts/
	cd submodules/instant-ngp
	cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build build --config RelWithDebInfo -j
	cd ../..

	

if you have problem with Instant-NGP compile. Please refer their [instruction](https://github.com/NVlabs/instant-ngp)

## Download assets ##
Don't skip this part when you want to generate your own results!
Download pretrained model for [MODNet](https://github.com/ZHKKKe/MODNet "MODNet"), [CDGNet](https://github.com/tjpulkl/CDGNet "CDGNet") and [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch "face-parsing").
You will need to make an accound on [here](https://smpl-x.is.tue.mpg.de "smplx") and [here](https://pixie.is.tue.mpg.de/ "PIXIE")

	# Download some pretrained model and data for avatar optimization.  
	# You should download pretrained model of CDGNet in thier repository, their are 
	# two kind of "LIP_epoch_149.pth", please download the one with a size of 300MB.
	bash fetch_pretrained_model.sh
    bash fetch_data.sh    #this will cost long time.



# Run

## Process your Video

To get started you firstly need to make a Video. The most important part is that your subject rmeins as still as possible to ensure a ideal Instant-NGP Reconstruction
Then run [process_video.py](https://github.com/Stretox/MonoHair/blob/master/process_video.py)

## COLMAP

Run Colmap. Create a "colmap" folder with a subfolder "images". In our tests with 300 Images using a automatic reconstruction with "extreme" Quality and Data Type "Singe Images" was enough to reconstruct at least 290 Camera Poses for a single model. Or use the tutorial for [COLMAP](https://colmap.github.io/tutorial.html "tutorial")
In the end export the model as text under File > Export Model es Text. This Folder should be named colmap_text

## Instant-NGP

Next you need to run Instant-NGP to reconstruct the scene.
1. run colmap2nerf.py located at ./submodules/instant-ngp/scripts/colmap2nerf.py
This needs to be run in the "colmap" folder.

		python [path-to-instant-ngp]/scripts/colmap2nerf.py --colmap_matcher exhaustive  --aabb_scale 4

2. move the images-folder and the resulting transforms.json to a subfolder in the instant-ngp subfolder.

3. 	Make an instant-ngp reconstruction.

	- GUI: Just run

			[path-to-instant-ngp]$ ./instant-ngp /colmap_subfolder
	- Headless:

			[path-to-instant-ngp]$ python scripts/run.py colmap_subfolder/ --save_snapshot snapshot.ingp
	Running 35000 Steps should result in a good reconstruction.

4.  In the GUI (load the snapshot.ingp if needed) crop the bbox size until only the bust of the subject remains. Under "World transform & Crop box" you can make fine adjustments

![](video_preprocess/fig/crop_save.jpg)

5. Save the base.ingp under Snapshot into the "colmap" folder.

6. Choose a good front view and click on "Add from Cam" in the Camera Window. Save the resulting json as key_frame.json in the "colmap" folder.

![](video_preprocess/fig/add_key_frame_and_save_cam.jpg)

**Known Issue:** When running the prepare_data.py in the Reconstruction Step, the resulting cameras may be offcenter. Especially if *shift_help: false* is set in the config file. You can Visualise base_cam.json by typing the path into the Camera Window and then clicking an load.

## Final Steps

1. Create a new "subject" Folder under MonoHair/data and copy the "colmap" folder there.

2. Under configs create a Config file for "Bust_fit" and "reconstruct". You can simply copy jenya2.yaml and modify it. Make sure to set image size to [heigt, width].

## 3D Hair Reconstruction ##

	# Prepare data: instant-ngp intialization, segmentation, gabor filter etc. You skip this step if use our provided data.
    python prepare_data.py --yaml=configs/reconstruct/big_wavy1 

	# Hair exterior optimization
	python PMVO.py --yaml=configs/reconstruct/big_wavy1

	# Hair interior inference
	python infer_inner.py --yaml=configs/reconstruct/big_wavy1
	
	# Strands generation
	python HairGrow.py --yaml=configs/reconstruct/big_wavy1

I recommend using something like [tmux](https://github.com/tmux/tmux) if you run Code on a GPU-Server

## Results

The Results are generated as .hair files under "output". The final result will be named connected_strands.hair. You can convert the Result to a .ply file using [this](https://github.com/KeyuWu-CS/MonoHair/issues/2#issuecomment-2210040260) skript by User moranli-aca. This can then be used to visualize the results in Blender or other software.

![](video_preprocess/fig/result.png)

## Visualization ##
If you are a Windows User you can also download the authors released program at  [One Drive](https://1drv.ms/f/s!AhfQmEHzY54Ya2gGaslXnM2IPCk?e=phk5me "One Drive") to visualize the results. 

    # First copy the output/10-16/full/connected_strands.hair to ../Ours/Voxel_hair
	cp data/case_name/output/10-16/full/connected_strands.hair data/case_name/ours/Voxel_hair

	# unzip VoxelHair_demo_v3.zip and click VoxelHair_v1.exe. 
	# Then click "Load Strands" to visualize the results. You also can use Blender to achieve realistic rendering.


<!-- ## Test your own data ##
In our given examples, we ignored the steps of running colmap and training instant ngp. So if you want to test your own captured videos. Please refer to the [following steps](https://github.com/KeyuWu-CS/MonoHair/tree/master/video_preprocess). -->


## Download examples ##
Download example data on Key-UwU's [One Drive](https://1drv.ms/f/s!AhfQmEHzY54Ya2gGaslXnM2IPCk?e=phk5me "One Drive"). For obtaining a certain results
, they have run colmap and save the pretrained instant-ngp weight. Then you need to run the four steps from 3D-Hair Reconstruction to get the results. Results are also provided (include intermediate results) in the "full" folder. You can use them to check the results of each step.
**Tips:** Since the wig uses the same unreal human head, flame models as well as multi-view bust fitting is skipped. 


## Citation ##

Check out [KeyuWu-CS original github](https://github.com/KeyuWu-CS/MonoHair)

    @inproceedings{wu2024monohair,
	  title={MonoHair: High-Fidelity Hair Modeling from a Monocular Video},
	  author={Wu, Keyu and Yang, Lingchen and Kuang, Zhiyi and Feng, Yao and Han, Xutao and Shen, Yuefan and Fu, Hongbo and Zhou, Kun and Zheng, Youyi},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  pages={24164--24173},
	  year={2024}
	}

## Acknowledgments ##
Here are some great resources we benefit from:

- [MODNet](https://github.com/ZHKKKe/MODNet "MODNet") and [CDGNet](https://github.com/tjpulkl/CDGNet "CDGNet") to segment hair.
- [Instant-NGP](https://github.com/NVlabs/instant-ngp "Instant-NGP") for coarse initizalization.
- [DELTA](https://github.com/yfeng95/DELTA "DELTA") for bust fitting.
  

## TO DO List ##
- [x] Upload full example data (before June.24)
- [x] Check version problem (before June.24)
- [x] Release visualization program (before June.30)
- [ ] Automatic method to add key_frame.json

