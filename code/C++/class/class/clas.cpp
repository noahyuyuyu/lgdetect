#include <iostream>
using namespace std;

class Point {
public:
	int x;
	int y;
	Point(int x1, int y1) : x(x1), y(y1)  //成员初始化列表
	{ }
	int getDistance()
	{
		return x * x + y * y;  // 平方和
	}
};

void changePoint1(Point point)    //使用对象作为函数参数
{
	point.x += 1;
	point.y -= 1;
}

void changePoint2(Point* point)   //使用对象指针作为函数参数
{
	point->x += 1;
	point->y -= 1;
}

void changePoint3(Point& point)  //使用对象引用作为函数参数
{
	point.x += 1;
	point.y -= 1;
}


int main() /*有且
		   
		   仅有
		   
		   一个*/
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
