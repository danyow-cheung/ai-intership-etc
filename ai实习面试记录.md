# ai学习路线

## 一、python基本技能

~~（一）python基础~~

~~学习链接：https://www.runoob.com/python/python-tutorial.html（可自行百度）~~

~~学习方法：根据学习链接，学习python基础教程~~

~~（二）python MYSQL~~

~~学习链接：https://www.runoob.com/python/python-mysql.html（可自行百度）~~

~~学习方法：根据学习链接，学习python MYSQL~~

```python
# 打开数据库连接
# localhost+用户名+密码+数据库
db = MySQLdb.connect("localhost", "testuser", "test123", "TESTDB", charset='utf8' )
```

~~（三）python 网络编程~~

~~学习链接：https://www.runoob.com/python/python-socket.html（可自行百度）~~

~~学习方法：根据学习链接，学习python 网络编程~~

~~学习视频：https://www.youtube.com/watch?v=3QiPPX-KeSc~~

服务器：

```python
import socket 
import threading 
import time
from xmlrpc.client import Server 
# 子节长度
HEADER = 64

PORT = 5050
# mac-ifconfig本地ip地址
# SERVER = "192.168.3.34"

# 自动获取ip地址
SERVER =socket.gethostbyname(socket.gethostname())


ADDR =(SERVER,PORT)

DISCONNECTIO_MSG= "disconnect"

# 创建af网络，网络协议
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# 套接字链接到网路
server.bind(ADDR)

def handle_client(conn,addr):
    # 并行运行
    print("[new connection]{addr} connected.")
    
    connected = True
    while connected:
        # 接受字节数目
        msg_length = conn.recv(HEADER).decode("utf-8")
        if msg_length:
            msg_length = int(msg_length)

            msg = conn.recv(msg_length).decode("utf-8")
            if msg ==DISCONNECTIO_MSG:
                connected = False
            print(f"[{addr}] {msg}")


    conn.close()


def start():
    # 监听网路
    server.listen()
    print("[listening] server is listenting on {SERVER}")
    while True:
        conn ,addr= server.accept()
        thread = threading.Thread(target=handle_client,args=(conn,addr))
        thread.start()
        print(f"[acctive connection]{threading.activeCount() -1 }")


print("[server]server is staring ...")
start()


```

客户端：

```python
import socket


# 子节长度
HEADER = 64

PORT = 5050
DISCONNECTIO_MSG= "disconnect"
# win下
# SERVER = "192.168.1.27"
#mac
SERVER = "127.0.0.1"
ADDR = (SERVER,PORT)
FORMAT = 'utf-8'

client  = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(ADDR)


def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)

    send_length = str(msg_length).encode(FORMAT)

    send_length+=b' ' *(HEADER- len(send_length))

    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))

send("hello from socket python")
input()
send("hello from socket-2")
input()
send("hello from socket-3")
send(DISCONNECTIO_MSG)


```

~~（四）python Flask~~

~~学习链接：https://www.jianshu.com/p/6452596c4edb（可自行百度）~~

~~学习方法：根据学习链接，学习python Flask~~

~~（五）python json解析~~

~~学习链接：https://www.runoob.com/python/python-json.html（可自行百度）~~

~~学习方法：根据学习链接，学习python json~~

~~（六）python OS 文件/目录~~

~~学习链接：https://www.runoob.com/python/os-file-methods.html（可自行百度）~~

~~学习方法：根据学习链接，学习python OS 文件/目录~~

~~（七）python 多线程~~

~~学习链接：https://www.runoob.com/python/python-multithreading.html（可自行百度）~~

~~学习方法：根据学习链接，学习python 多线程~~

~~（八）MYSQL安装与 workbench使用~~

~~学习链接：https://blog.csdn.net/zs1342084776/article/details/88701261（可自行百度）~~

~~学习方法：根据学习链接，学习MYSQL安装与 workbench使用~~

~~（九）Python GUI开发（QT或tinker)~~ 

~~学习链接：https://www.runoob.com/python/python-gui-tkinter.html（可自行百度）~~

~~学习方法：根据学习链接，学习Python GUI开发~~

~~学习视频：https://www.youtube.com/watch?v=itRLRfuL_PQ~~

----------



## 二、python图像识别基础

~~（一）python图像识别基础~~

~~学习链接：https://blog.csdn.net/galen_xia/category_10346867.html（可自行在OPENCV-PYTHON官网上学习）~~

~~学习方法：根据学习链接，学习python图像识别基础（要求：掌握Opencv-python大部分代码案例，跑起来了解即可）~~



## 三、tensorflow

~~（一）tensorflow图像分类~~

~~学习链接：https://tensorflow.google.cn/tutorials/images/classification~~

~~学习方法：根据学习链接，学习tensorflow图像分类（要求：重现，并能用自己的数据跑）~~



~~（二）tensorflow图像数据增强~~

~~学习链接：https://tensorflow.google.cn/tutorials/images/data_augmentation~~

~~学习方法：根据学习链接，学习tensorflow图像数据增强（要求：了解其原理，并能重现）~~





~~（三）tensorflow自编码~~

~~学习链接：https://tensorflow.google.cn/tutorials/generative/autoencoder~~

~~学习方法：根据学习链接，学习tensorflow自编码（要求：重现，并能用自己的数据跑）~~

异常检测结果不好



~~（四）tensorflow目标检测~~

~~学习链接：自行在百度上搜索~~

~~学习方法：根据学习链接，学习tensorflow目标检测（要求：YOLO系列，重现，并能用自己的数据跑）~~

https://github.com/danyow-cheung/emotion-detection



（五）tensorflow图像分割

学习链接：https://tensorflow.org/tutorials/images/segmentation

学习方法：根据学习链接，学习tensorflow图像分割

（**要求**： 

~~（1）基于k-means opencv有案例，可以做图像量化（也是图像分割）~~

~~（2）基于tessorflow重现，并能用自己的数据跑~~ 

~~（3）网上有很火的deeplab v3，也要能重现~~

）

慢慢做。。。。





~~（六）tensorflow图像数据增强~~

~~学习链接：https://tensorflow.google.cn/tutorials/images/data_augmentation~~

~~学习方法：根据学习链接，学习tensorflow图像数据增强（要求：了解其原理，并能重现）~~（重复）



~~（七）tensflow JS~~

~~学习链接：https://tensorflow.google.cn/js/models~~

~~学习方法：根据学习链接，学习tensorflow JS（要求：重现其中2-3个图像识别案例即可）~~

~~CNN手写识别~~

~~2d数据预测~~

~~构建垃圾评论检测系统(模型不支持mac)~~

~~摄像头~~



~~（八）tensorflow时间预测（LSTM）~~

~~学习链接：https://tensorflow.google.cn/tutorials/structured_data/time_series~~

~~学习方法：根据学习链接，学习tensorflow 时间预测（LSTM）（要求：重现，并能用自己的数据跑；主要是理解LSTM）~~

最后结果不好



~~（九）tensorflow文本分类~~

~~学习链接：https://www.tensorflow.org/text/tutorials/classify_text_with_bert~~

~~学习方法：根据学习链接，学习tensorflow 文本分类（要求: 学习贝叶斯、深度学习两种方法。重现，并能用自己的数据跑）~~





## 四、Openpose（OpenPose是一个开源的姿态评估系统，可以实时进行多人检测，包括人体，手，脸和脚等135个关键点）

（一）Openpose

学习链接：自行在百度上搜索

学习方法：根据学习链接，学习Openpose（要求：实现手势识别、姿势识别）



## 五、百度飞桨

（一）百度飞桨图像分割

学习链接：https://www.paddlepaddle.org.cn/modelbase

学习方法：根据学习链接，学习百度飞桨图像分割的相关内容（要求：重现，并能用自己的数据跑）

（二）百度飞桨对话系统

学习链接：https://www.paddlepaddle.org.cn/modelbasedetail/DGU

学习方法：根据学习链接，学习百度飞桨对话系统的相关内容（要求：重现，并能用自己的数据跑）



## ~~六、python NLP基础（自然语言处理(NLP)就是开发能够理解人类语言的应用程序或服务）~~

~~学习链接：自行在百度上搜索~~

~~学习方法：学习分词、词向量、文本相似度等（要求：会使用结巴分词(jieba)）~~

--------------------------------



# ai实习面试记录

**8.23--广州青莲网络--打标工**

1. yolov5的原理，基层实现

   yolov5的模型

   ![img](https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png)

   由于 YOLO v5 是一个单级目标检测器，它与任何其他单级目标检测器一样具有三个重要部分。

   1. 模型骨干

      *Model Backbone 主要用于从给定的输入图像中提取重要特征。在 YOLO v5 中，CSP — Cross Stage Partial Networks被用作骨干，从输入图像中提取丰富的信息特征。*

      > 神经网络使最先进的方法能够在对象检测等计算机视觉任务上取得令人难以置信的结果。然而，这样的成功很大程度上依赖于昂贵的计算资源，这阻碍了拥有廉价设备的人们欣赏先进技术。在本文中，我们提出了<u>跨阶段部分网络（CSPNet）</u>来缓解以前的工作需要从网络架构角度进行大量推理计算的问题。我们将问题归因于网络优化中的重复梯度信息。
      >
      > 
      >
      > CSPNet使用了暗网框架
      >
      > Darknet 是一个用 C 和 CUDA 编写的开源神经网络框架。它快速、易于安装，并支持 CPU 和 GPU 计算。

      

   2. 模型脖子

      *Model Neck 主要用于生成特征金字塔。特征金字塔有助于模型很好地概括对象缩放。它有助于识别具有不同大小和比例的同一对象。*

      *特征金字塔非常有用，可以帮助模型在看不见的数据上表现良好。还有其他模型使用不同类型的特征金字塔技术，如FPN、BiFPN、PANet等。*

      *在 YOLO v5中， PANet被用作颈部来获取特征金字塔。*

   3. 模型头

      *模型Head主要用于执行最后的检测部分。它将锚框应用于特征并生成具有类概率、对象分数和边界框的最终输出向量。*

   

   

   https://towardsai.net/p/computer-vision/yolo-v5%E2%80%8A-%E2%80%8Aexplained-and-demystified

2. yolov5输出的txt格式的代表含义

   ![img](https://img-blog.csdnimg.cn/img_convert/b98785301ac62e0194abd7f9998e949e.png)

   > (a,<u>b,c,d,e</u>)
   >
   > a:代表检测的类别
   >
   > b,c,d,e:（进行归一化之后的）标注框的中心坐标和相对宽高

   **归一化目标：**

   1. 把数变为（0，1）之间的小数

   2. 把有量纲表达式变为无量纲表达式

   **归一化好处：**

   1. 提升模型的收敛速度
   2. .提升模型的精度
   3. 从经验上说，归一化是让不同维度之间的特征在数值上有一定比较性，可以大大提高分类器的准确性。

   **归一化方法：**

   1. normalize()
   2. MinMaxScaler()

   

3. python多线程爬虫原理以及多线程的缺点

   如果你的代码是IO密集型，多线程可以明显提高效率。例如制作爬虫，绝大多数时间爬虫是在等待socket返回数据。某个线程等待IO的时候其他线程可以继续执行。

   > 多线程的意义：
   >
   > 从事计算机视觉项目时，需要预处理大量图像数据。这很耗时，如果您可以并行处理多个图像，今天的大多数计算机至少有一个多核处理器，允许同时执行多个进程。
   >
   > 多处理按照顺序执行多个任务来提高程序的效率
   >
   > 多个线程在一个进程中运行，并相互共享进程的内存空间。

   多线程知识相关

   > <u>threading-基于线程的并行性</u>
   >
   > 在python3中已经弃用
   >
   > <u>multiprocessing-基于进程的并行性</u>
   >
   > multiprocessing 包同时提供了本地和远程并发操作，通过使用子进程而非线程有效地绕过了 全局解释器锁。 因此，multiprocessing 模块允许程序员充分利用给定机器上的多个处理器
   >
   > 
   >
   > <u>线程同步</u>
   > 如果多个线程共同对某个数据修改，则可能出现不可预料的结果，为了保证数据的正确性，需要对多个线程进行同步。使用Thread对象的Lock和Rlock可以实现简单的线程同步，

   **用多线程编程执行任务时，多个线程可以共享内存，因此通常认为这比多进程编程更简 单。但是，这种便利也需要付出代价**

   

   **Python 的全局解释器锁(global interpreter lock，GIL)会阻止多个线程同时运行同一行代 码。GIL 确保由所有进程共享的内存不会中断(例如，内存中的字节用一个值写一半，用 另一个值写另一半)。虽然这个锁可以让你写多线程的程序，并在同一时刻获取代码的运 行结果，但是这么做存在性能瓶颈。**

   理论上，用独立的进程抓取比用独立的线程抓取要快，主要有两个理由。

   1. 进程不受 GIL 的限制，可以同时运行同一行代码，同时调整同一个对象(其实是同一 个对象的多个实例化)。

   2. 进程可以在多个 CPU 核心上运行，如果每个进程或线程需要消耗大量的处理器资源， 这可能会提升运行速度。

   

   额外补充:管道

   > 使用多进程时，一般使用消息机制实现进程间通信，尽可能避免使用同步原语，例如锁。
   >
   > 消息机制包含： Pipe() (可以用于在两个进程间传递消息)，以及队列(能够在多个生产者和消费者之间通信)。

4. 排序算法举例+是否具有稳定性

   | 排序算法 | 稳定性 | 定义                                         |
   | -------- | ------ | -------------------------------------------- |
   | 冒泡排序 | 稳定   | 重复走要排序的数列，一次比较两个元素         |
   | 选择排序 | 不稳定 | 找序列中最大的元素，再找第二大               |
   | 快速排序 | 不稳定 | 分治算法（最坏情况下为on^2)                  |
   | 归并排序 | 稳定   | 设置左右指针，找到最小元素合并，左指针右移动 |

   

5. 面向对象编程和函数式编程

   面向对象编程好处：可扩展性+可复用

   

6. 重点考察yolo相关知识

   yolo v1 &v2&v3 

   https://medium.com/@venkatakrishna.jonnalagadda/object-detection-yolo-v1-v2-v3-c3d5eca2312a

   

7. 激活函数的形式以及调参技巧

​		leakly relu ,relu 





# ai 面经学习

牛客网面经

https://www.nowcoder.com/tutorial/95/ea84cef4fb2c4555a7a4c24ea2f6b6e8

github深度学习面试

https://github.com/amusi/Deep-Learning-Interview-Book

github机器学习相关

https://github.com/zhengjingwei/machine-learning-interview

熟人推荐

> tensorRT这个框架可以去看看
>
> yolov5移植到jetsonnano
>
> 用tensorRT跑
>
> https://blog.csdn.net/qq_41204464/article/details/124737245?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166150980716781647537566%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166150980716781647537566&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_34-2-124737245-null-null.142^v42^pc_rank_34_2,185^v2^control&utm_term=yolov5%20tensorrt%E9%83%A8%E7%BD%B2%E5%88%B0jetson%20nano&spm=1018.2226.3001.4187
>
> 跟着这个做，过一遍模型移植
>
> https://github.com/OAID/Tengine/releases
>
> 那就跑这个，这个cpu也能跑
>
> 
>
> 





<u>Cmake&Makefile</u>

**cmake**：不用语言或编译器开发一个项目，最终输出可执行文件或者共享库（.dll，.so等）



**makefile** ：多个.cpp编译输出可执行文件，数量比cmake少

。还能检测到某个cpp文件有无变化，有变化会另外重新编译









