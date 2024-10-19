import os

"""
导入SQLAlchemy相关模块及配置文件中的数据库URL。
从sqlalchemy库导入了用于定义数据库模型的字段类型如二进制数据(BINARY)、日期时间(DateTime)等。
准备数据库连接和ORM所需的基本组件，包括创建引擎(create_engine)、声明基类(declarative_base)和会话(sessionmaker)。这些准备工作为后续定义和操作数据库模型提供了支持。
"""
from sqlalchemy import BINARY, Column, DateTime, Integer, String
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import SQLALCHEMY_DATABASE_URL

# 数据库目录不存在的时候自动创建目录。TODO：如果是mysql之类的数据库，这里的代码估计是不兼容的
folder_path = os.path.dirname(SQLALCHEMY_DATABASE_URL.replace("sqlite:///", ""))
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 本地扫描数据库
BaseModel = declarative_base()
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
DatabaseSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# PexelsVideo数据库
BaseModelPexelsVideo = declarative_base()
engine_pexels_video = create_engine(
    'sqlite:///./PexelsVideo.db',
    connect_args={"check_same_thread": False}
)
DatabaseSessionPexelsVideo = sessionmaker(autocommit=False, autoflush=False, bind=engine_pexels_video)

def create_tables():
    """
    创建数据库表
    """
    BaseModel.metadata.create_all(bind=engine)
    BaseModelPexelsVideo.metadata.create_all(bind=engine_pexels_video)


class Image(BaseModel):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True)
    path = Column(String(4096), index=True)  # 文件路径
    modify_time = Column(DateTime)  # 文件修改时间
    features = Column(BINARY)  # 文件预处理后的二进制数据


class Video(BaseModel):
    __tablename__ = "video"
    id = Column(Integer, primary_key=True)
    path = Column(String(4096), index=True)  # 文件路径
    frame_time = Column(Integer, index=True)  # 这一帧所在的时间
    modify_time = Column(DateTime)  # 文件修改时间
    features = Column(BINARY)  # 文件预处理后的二进制数据


class PexelsVideo(BaseModelPexelsVideo):
    __tablename__ = "PexelsVideo"  # 在线视频的意思
    id = Column(Integer, primary_key=True)
    title = Column(String(128))  # 标题
    description = Column(String(256))  # 视频描述
    duration = Column(Integer, index=True)  # 视频时长，单位秒
    view_count = Column(Integer, index=True)  # 视频播放量
    thumbnail_loc = Column(String(256), index=True)  # 视频缩略图链接
    content_loc = Column(String(256))  # 视频链接
    thumbnail_feature = Column(BINARY)  # 视频缩略图特征

