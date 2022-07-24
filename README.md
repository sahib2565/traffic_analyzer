![license](https://img.shields.io/apm/l/vim-mode)
![build](https://img.shields.io/badge/build-unknown-red)
# Traffic analyzer
The program in "main.py" in packcage folder is able to track,classify,count and tell the direction of the passing vehicle.It uses sort to track the vehicles, it uses yolov3 model to classify cars and a simple algorithm to know in which direction the car is going,it then sends the data to a **influx database**


## Getting started

I reccomend using conda to create a virtual environment

### Conda(optional)
```bash
conda create -n traffic_analyzer python=3.10
```
```bash
conda activate traffic_analyzer
```
Install dependices
### Pip
```bash
pip install -r requirements.txt
```
### Installing Opencv
I also highly reccomend installing Opencv with CMake so you can you use the GPU instead of the CPU, [Opencv with cmake](https://www.youtube.com/watch?v=YsmhKar8oOc),else you can install opencv by simply using these command

```bash
pip install opencv-python
```

## Download the weights file
Install the weights file from [here](https://drive.google.com/file/d/1Ru6tmkTI3DVtQKtV2XP00lkj2yU5J7i2/view?usp=sharing).

Put the file inside the folder yolov
## Runing main.py

Go to the package folder and run this command:
```bash
python main.py -u <username_camera> -p <password_camera> -i <ip_adress> -c <channel_id> 
```

## References
- [SORT](https://github.com/abewley/sort)

Our sort file is modified

## Author

### Il_kima
