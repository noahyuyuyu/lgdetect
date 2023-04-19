# 导入所有必要的库
import cv2
import os
import time
from datetime import datetime
import threading
import sys  # 导入sys模块

sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
# rtsp = ["rtsp://admin:a12345678@ai.huoyanzn.com:20217/cam/realmonitor?channel=1&subtype=0",
# "rtsp://admin:a12345678@ai.huoyanzn.com:20223/cam/realmonitor?channel=1&subtype=0"]
# 从指定的路径读取视频
# cam = cv2.VideoCapture("rtsp://admin:a12345678@ai.huoyanzn.com:20217/cam/realmonitor?channel=1&subtype=0")
# cam5 = cv2.VideoCapture("rtsp://admin:a12345678@ai.huoyanzn.com:20223/cam/realmonitor?channel=1&subtype=0")

print('start...')


class myThread(threading.Thread):  # 继承父类threading.Thread
    count = [];
    url = '';
    port = '';

    def __init__(self, count, url, port):
        threading.Thread.__init__(self)
        self.count = count;
        self.url = url;
        self.port = port;

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print(self.url)
        test(self.count, self.url, self.port)


def check_time(start_time, end_time):
    e_hours, e_mins = end_time.split(":")
    s_hours, s_mins = start_time.split(":")
    nt = datetime.now()
    cd = nt.strftime("%Y-%m-%d")
    sd = datetime.strptime(cd + " " + s_hours + ":" + s_mins + ":00", "%Y-%m-%d %H:%M:%S")
    ed = datetime.strptime(cd + " " + e_hours + ":" + e_mins + ":00", "%Y-%m-%d %H:%M:%S")
    if (nt <= ed and nt >= sd):
        return True
    else:
        return False


# work_time = ['07:30-09:30', '15:00-16:00', '17:00-20:00']
work_time = ['18:30-20:30', '6:30-8:00']


def time_limit(eventDesc):
    work_time1 = work_time
    for i in work_time1:
        timer = str(i).split(',')
        for j in timer:
            timer = str(j).split('-')
            start_time = timer[0]
            # print("start: ",start_time)
            end_time = timer[1]
            # print("end:  ", end_time)
            if check_time(start_time, end_time): return True


# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    print(address)
    cv2.imwrite(address, image)


def test(count, url, port):
    cam = cv2.VideoCapture(url)
    ret, frame = cam.read()  # ret为布尔值 frame保存着视频中的每一帧图像 是个三维矩阵
    i = 0
    timeF = 250  # 设置要保存图像的间隔 15为每隔15帧保存一张图像
    j = 0
    while ret:
        # if ret:
        if time_limit('work_time'):
            i = i + 1
            # 如果视频仍然存在，继续创建图像
            if i % timeF == 0:
                # 呈现输出图片的数量
                j = j + 1
                if len(count) == 0:
                    save_image(frame, path + port + '/', j)
                    count.append(j)
                    # print('save21 image:', j)
                if len(count) > 0:
                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
                    save_image(frame, path + port + '/', count[-1])
                    count[-1] += 1
                    # print('save21 image:', a[-1])
                # print('save23 image:', j)
            ret, frame = cam.read()
    # if not ret:
    while not ret:
        if time_limit('work_time'):
            test(count, url, port);


threads = []
nt = datetime.now()
path = "./" + str(nt.strftime("%Y-%m-%d")) + '/'
if not os.path.exists(path):
    os.makedirs(path)

#path = "F:/save_data/";
if __name__ == '__main__':
    fp = open('./port.txt');
    port = fp.readline()
    urls = [];
    ports = [];
    while port:
        port = port.strip('\n')
        ports.append(port);
        url = 'rtsp://admin:a12345678@ai.huoyanzn.com:' + port + '/cam/realmonitor?channel=1&subtype=0'
        urls.append(url);
        port = fp.readline()
    n = len(urls);
    m = 53
    for i in range(len(ports)-m, len(ports)-(m-4)):
        url = urls.__getitem__(i);
        port = ports.__getitem__(i);
        port = str(port)
        if not os.path.exists(path + port):
            os.mkdir(path + port);
        count = [];
        threads.append(myThread(count, url, port));
    fp.close();
    print(len(threads))
    for t in threads:
        t.start()
