#include <iostream>
using namespace std;

/*** ����һ���������� ***/
class Figure {
protected:
	double x, y;
public:
	Figure(double a, double b) : x(a), y(b) {  }
	virtual void getArea()      //�麯��
	{
		cout << "No area computation defind for this class.\n";
	}
};

class Triangle : public Figure {
public:
	Triangle(double a, double b) : Figure(a, b) {  }
	//�麯���ض��壬�����������ε����
	void getArea() {
		cout << "Triangle with height " << x << " and base " << y;
		cout << " has an area of " << x * y * 0.5 << endl;
	}
};

class Square : public Figure {
public:
	Square(double a, double b) : Figure(a, b) {  }
	//�麯���ض��壬��������ε����
	void getArea() {
		cout << "Square with dimension " << x << " and " << y;
		cout << " has an area of " << x * y << endl;
	}
};

class Circle : public Figure {
public:
	Circle(double a) : Figure(a, a) {  }
	//�麯���ض��壬������Բ�����
	void getArea() {
		cout << "Circle with radius " << x;
		cout << " has an area of " << x * x * 3.14 << endl;
	}
};

int main() {
	Figure* p;
	Triangle t(10.0, 6.0);
	Square s(10.0, 6.0);
	Circle c(10.0);

	p = &t;
	p->getArea();
	p = &s;
	p->getArea();
	p = &c;
	p->getArea();

	return 0;
}
