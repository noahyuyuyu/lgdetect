# 文件处理
import file
import shutil
from shutil import copyfile

# 1.os模块
os.makedirs(path)#创建多级目录
os.listdir(path)#列举目录
os.path.join(path1,path2)#将路径组合
os.path.exists(path)#判断文件是否存在
os.walk(path)#遍历文件夹
os.remove(path)#指定删除某个文件

# 2.shutil模块
shutil.copytree(path)#递归复制
shutil.rmtree(path)#递归删除
shutil.copyfile(src,dst)#从源src复制到dst

copyfile(srcimagefullname, destimagefullname)#复制images至指定文件夹
copyfile(srclabelfullname, destlabelfullname)#复制labels至指定文件夹

