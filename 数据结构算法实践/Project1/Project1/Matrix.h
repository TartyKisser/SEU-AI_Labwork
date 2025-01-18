#pragma once
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

class Matrix {
public:
	Matrix(int row, int col);
	void showMyMatrix();
	//set and get
	int getRowNum();
	void setRowNum(int rowNum);
	int getColNum();
	void setColNum(int colNum);
	vector<vector<int>> getMyMatrix();
	void setMyMatrix(vector<vector<int>> myMatrix);

private:
	//矩阵的行数和列数
	int rowNum;
	int colNum;
	vector<vector<int>> myMatrix;

	void initMyMatrix();
};

Matrix::Matrix(int row, int col)
{
	//初始化矩阵，矩阵的内容随机生成
	this->rowNum = row;
	this->colNum = col;
	initMyMatrix();
}

void Matrix::showMyMatrix()
{
	for (int i = 0; i < rowNum; i++)
	{
		for (int j = 0; j < colNum; j++)
		{
			cout << myMatrix[i][j] << " ";
		}
		cout << endl;
	}
}

int Matrix::getRowNum()
{
	return rowNum;
}

void Matrix::setRowNum(int rowNum)
{
	this->rowNum = rowNum;
}

int Matrix::getColNum()
{
	return colNum;
}

void Matrix::setColNum(int colNum)
{
	this->colNum = colNum;
}

vector<vector<int>> Matrix::getMyMatrix()
{
	return myMatrix;
}

void Matrix::setMyMatrix(vector<vector<int>> myMatrix)
{
	this->myMatrix.assign(myMatrix.begin(), myMatrix.end());
}

void Matrix::initMyMatrix()
{
	myMatrix.resize(rowNum);
	srand(time(0));
	for (int i = 0; i < rowNum; i++)
	{
		myMatrix[i].resize(colNum);
		for (int j = 0; j < colNum; j++)
		{
			myMatrix[i][j] = rand() % 3;
		}
	}
}


