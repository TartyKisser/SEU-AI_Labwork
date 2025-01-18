#pragma once
class LosserTree {
public:
	int* data;
	int* losserTree;
	int capacity;
	int currentDataNum;


	LosserTree();
	LosserTree(int*, int);
	void Adjust(int);
	void addNewData(int,int);
	int PopAndAdd(int);
	int getWinner();
	int getWinnerPosition();
};
