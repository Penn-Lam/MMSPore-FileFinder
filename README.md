# MMSPore-FileFinder


## 功能

- 文字搜图
- 以图搜图
- 文字搜视频（会给出符合描述的视频片段）
- 以图搜视频（通过视频截图搜索所在片段）
- 图文相似度计算（只是给出一个分数，用处不大）

## 部署说明


### 通过源码部署

首先安装Python环境，然后下载本仓库代码。

注意，首次运行会自动下载模型。下载速度可能比较慢，请耐心等待。如果网络不好，模型可能会下载失败，这个时候重新执行程序即可。

1. 首次使用前需要安装依赖：`pip install -U -r requirements.txt`，Windows系统可以双击`install.bat`（NVIDIA GPU加速）或`install_cpu.bat`（纯CPU）。
2. 如果你打算使用GPU加速，则执行基准测试判断是CPU快还是GPU快：`python benchmark.py`，Windows系统可以双击`benchmark.bat`。GPU不一定比CPU快，在我的Mac上CPU更快。
3. 如果不是CPU最快，则修改配置中的`DEVICE`，改为对应设备（配置修改方法请参考后面的配置说明）。
4. 启动程序：`python main.py`，Windows系统可以双击`run.bat`。

如遇到`requirements.txt`版本依赖问题（比如某个库版本过新会导致运行报错），请提issue反馈，我会添加版本范围限制。

如遇到硬件支持但无法使用GPU加速的情况，请根据[PyTorch文档](https://pytorch.org/get-started/locally/)更新torch版本。

如果想使用"下载视频片段"的功能，需要安装`ffmpeg`。如果是Windows系统，记得把`ffmpeg.exe`所在目录加入环境变量`PATH`，可以参考：[Bing搜索](https://cn.bing.com/search?q=windows+%E5%A6%82%E4%BD%95%E6%B7%BB%E5%8A%A0+path+%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)。



## 配置说明

所有配置都在`config.py`文件中，里面已经写了详细的注释。

建议通过环境变量或在项目根目录创建`.env`文件修改配置。如果没有配置对应的变量，则会使用`config.py`中的默认值。例如`os.getenv('HOST', '0.0.0.0')`，如果没有配置`HOST`变量，则`HOST`默认为`0.0.0.0`。

`.env`文件配置示例：

```conf
ASSETS_PATH=C:/Users/Administrator/Pictures,C:/Users/Administrator/Videos
DEVICE=cuda
```

目前功能仍在迭代中，配置会经常变化。如果更新版本后发现无法启动，需要参考最新的配置文件手动改一下配置。

如果你发现某些格式的图片或视频没有被扫描到，可以尝试在`IMAGE_EXTENSIONS`和`VIDEO_EXTENSIONS`增加对应的后缀。如果你发现一些支持的后缀没有被添加到代码中，欢迎提issue或pr增加。

小图片没被扫描到的话，可以调低`IMAGE_MIN_WIDTH`和`IMAGE_MIN_HEIGHT`重试。

如果想使用代理，可以添加`http_proxy`和`https_proxy`，如：

```conf
http_proxy=http://127.0.0.1:7070
https_proxy=http://127.0.0.1:7070
```

注意：`ASSETS_PATH`不推荐设置为远程目录（如SMB/NFS），可能会导致扫描速度变慢。



## 硬件要求

推荐使用`amd64`或`arm64`架构的CPU。内存最低2G，但推荐最少4G内存。如果照片数量很多，推荐增加更多内存。

测试环境：J3455，8G内存。全志H6，2G内存。

如果使用AMD的GPU，仅支持在Linux下使用GPU加速。请参考：[PyTorch文档](https://pytorch.org/get-started/locally/)。

## 搜索速度

匹配阈值为0的情况下，在 J3455 CPU 上，1秒钟可以进行大约18000次图片匹配或5200次视频帧匹配。

调高匹配阈值可以提高搜索速度。匹配阈值为10的情况下，在 J3455 CPU 上，1秒钟可以进行大约30000次图片匹配或6100次视频帧匹配。

## 已知问题

1. 部分视频无法在网页上显示，原因是浏览器不支持这一类型的文件（例如svq3编码的视频）。
2. 点击图片进行放大时，部分图片无法显示，原因是浏览器不支持这一类型的文件（例如tiff格式的图片）。小图可以正常显示，因为转换成缩略图的时候使用了浏览器支持的格式。大图使用的是原文件。
3. 搜视频时，如果显示的视频太多且视频体积太大，电脑可能会卡，这是正常现象。建议搜索视频时不要超过12个。


