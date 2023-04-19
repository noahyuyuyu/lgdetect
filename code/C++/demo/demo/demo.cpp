#include <iostream>   //编译预处理命令
using namespace std;    //使用命名空间

int add(int a, int b);       //函数原型说明

int main()  //主函数
{
	int x, y;
	cout << "Enter two numbers: " << endl;
	cin >> x;
	cin >> y;
	int sum = add(x, y);
	cout << "The sum is : " << sum << '\n';
	return 0;
}

int add(int a, int b) //定义add()函数，函数值为整型
{
	return a + b;
}


