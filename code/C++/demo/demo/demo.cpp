#include <iostream>   //����Ԥ��������
using namespace std;    //ʹ�������ռ�

int add(int a, int b);       //����ԭ��˵��

int main()  //������
{
	int x, y;
	cout << "Enter two numbers: " << endl;
	cin >> x;
	cin >> y;
	int sum = add(x, y);
	cout << "The sum is : " << sum << '\n';
	return 0;
}

int add(int a, int b) //����add()����������ֵΪ����
{
	return a + b;
}


