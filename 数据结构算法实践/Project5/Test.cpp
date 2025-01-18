#include<iostream>
#include<fstream>
#include<string>
#include <windows.h>
#include"Memory.h"
using namespace std;

void Generate(string filePath, int fileLength) {
	remove(filePath.c_str());
	fstream file;
	file.open(filePath, ios_base::out | ios::binary);
	srand((unsigned)time(NULL));//根据时间来产生随机数种子
	int t = 0;
	for (int i = 0; i < fileLength; i++) {
		t = rand() % (fileLength * 2);
		file.write((char*)&t, sizeof(int));
	}
	file.close();
}

bool isSuccess(string filePath) {
	fstream file;
	file.open(filePath, ios_base::in | ios::binary);
	bool success = true;
	int t1 = -1;
	int t2 = 0;

	while (file.peek() != EOF) {
		file.read((char*)&t2, sizeof(int));
		if (t2 < t1) {
			success = false;
			break;
		}
		t1 = t2;
	}
	file.close();
	return success;
}

void Test1(string filePath) {
	int fileLength = 500;
	int cacheCapacity = 0;
	int Kway = 0;
	cout << "Data Length" << "		" << "Cache Capacity" << "		" << "Success/Unsuccess\n";
	cout << "-----------------------------------------------------------------------------------------------------------\n";
	for (int i = 0;i < 20;i++) {
		//随机生成数据大小，每次生成的数据规模比上次更大
		fileLength += 300 + rand() % 2000;
		cacheCapacity = 20 + (rand() % 50);
		Kway = 1 + (rand() % 10);
		cout << fileLength << "			"
			<< cacheCapacity << "				";
		Generate(filePath, fileLength);
		Memory ME(cacheCapacity, filePath, fileLength,Kway);
		ME.kWayMergeSort();
		if (isSuccess(filePath)) {
			cout << "Success\n";
		}
		else {
			cout << "Unsuccess\n";
		}
	}
}

int main() {
	string filePath = "diskDataTest.dat";

	Test1(filePath);
	return 0;
}




