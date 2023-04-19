#include <iostream>
#include<string>
using namespace std;
int main(void)
{
	int num = rand() % 100;
	cout << num << endl;
	int puT = 0;
	cout << "请你猜一下这个数是多少\n" << endl;
	while ((cin >> puT))
	{
		if (puT > num)
		{
			cout << "猜大了\n" << endl;
		}
		else if (puT <= num / 2)
		{
			cout << "太小了\n" << endl;
		}
		else if (puT >= num / 2 && puT < num)
		{
			cout << "再大一点\n" << endl;
		}
		else if (num == puT)
		{
			cout << "猜对了\n" << endl;
			break;
		}
	}
	system("pause");
	return 0;
}
