#include <iostream>
#include<string>
using namespace std;
int main(void)
{
	int num = rand() % 100;
	cout << num << endl;
	int puT = 0;
	cout << "�����һ��������Ƕ���\n" << endl;
	while ((cin >> puT))
	{
		if (puT > num)
		{
			cout << "�´���\n" << endl;
		}
		else if (puT <= num / 2)
		{
			cout << "̫С��\n" << endl;
		}
		else if (puT >= num / 2 && puT < num)
		{
			cout << "�ٴ�һ��\n" << endl;
		}
		else if (num == puT)
		{
			cout << "�¶���\n" << endl;
			break;
		}
	}
	system("pause");
	return 0;
}
