#include <iostream>
using namespace std;

class Point {
public:
	int x;
	int y;
	Point(int x1, int y1) : x(x1), y(y1)  //��Ա��ʼ���б�
	{ }
	int getDistance()
	{
		return x * x + y * y;  // ƽ����
	}
};

void changePoint1(Point point)    //ʹ�ö�����Ϊ��������
{
	point.x += 1;
	point.y -= 1;
}

void changePoint2(Point* point)   //ʹ�ö���ָ����Ϊ��������
{
	point->x += 1;
	point->y -= 1;
}

void changePoint3(Point& point)  //ʹ�ö���������Ϊ��������
{
	point.x += 1;
	point.y -= 1;
}


int main() /*����
		   
		   ����
		   
		   һ��*/
{
	Point point[3] = { Point(1, 1), Point(2, 2), Point(3, 3) };
	Point* p = point;
	changePoint1(*p);
	cout << "the distance is " << p[0].getDistance() << endl;
	p++;
	changePoint2(p);
	cout << "the distance is " << p->getDistance() << endl;
	changePoint3(point[2]);
	cout << "the distance is " << point[2].getDistance() << endl;

	return 0;
}
