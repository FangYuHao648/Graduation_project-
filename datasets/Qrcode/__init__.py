# -*- coding: utf-8 -*-

__version__ = '2.0'
__author__ = 'sixuwuxian'
__email__ = 'sixuwuxian@aliyun.com'
__url__ = 'https://wuxian.blog.csdn.net'

AUTHOR_INFO = ("基于YOLOv8的条形码二维码检测系统v1.0\n"
               "作者：思绪无限\n"
               "公众号: AI技术研究与分享\n"
               "CSDN: https://wuxian.blog.csdn.net\n"
               "BiliBili: https://space.bilibili.com/456667721\n"
               "博客园: https://www.cnblogs.com/sixuwuxian/\n"
               "知乎: https://www.zhihu.com/people/sixuwuxian\n\n"
               "原创代码，仅用于学习交流，禁止他人商用和盗卖，盗版必究！\n\n"
               "温馨提醒：本代码仅上传在面包多或CSDN平台，作者未在淘宝、咸鱼等平台开店，\n"
               "如出自类似平台皆为盗版，请联系作者邮箱：sixuwuxian@aliyun.com")

ENV_CONFIG = ("[配置环境]\n"
              "请按照给定的python版本配置环境，否则可能会因依赖不兼容而出错\n"
              "(1)使用anaconda新建python3.10环境:\n"
              "conda create -n env_rec python=3.10\n"
              "(2)激活创建的环境:\n"
              "conda activate env_rec\n"
              "(3)使用pip安装所需的依赖，可通过requirements.txt:\n"
              "pip install -r requirements.txt\n")

with open('./环境配置.txt', 'w', encoding='utf-8') as f:
    f.writelines(ENV_CONFIG + "\n\n" + AUTHOR_INFO)
