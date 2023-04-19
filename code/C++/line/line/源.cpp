#include <iostream>
using namespace std;

inline double circle(double r)  //ÄÚÁªº¯Êı
{
	double PI = 3.14;
	return PI * r * r;
}

int main()
{
	for (int i = 1; i <= 9; i++)
		cout << "r = " << i << " area = " << circle(i) << endl;
	return 0;
}
