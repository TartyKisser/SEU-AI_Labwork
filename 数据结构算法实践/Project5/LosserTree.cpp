#include "LosserTree.h"
#define INIT_MIN -1;
#define SWAP(x,y) {x=x+y;y=x-y;x=x-y;}

LosserTree::LosserTree()
{
	capacity = 4;
	data = new int[capacity + 1];
	losserTree = new int[capacity];
	for (int i = 0;i < capacity;i++) {
		data[i] = 0;
	}
	data[capacity] = INIT_MIN;
	for (int i = 0;i < capacity;i++) {
		losserTree[i] = capacity;
	}

	for (int i = capacity - 1;i >= 0;i--) {
		Adjust(i);
	}
}

LosserTree::LosserTree(int* newdata,int num)
{
	capacity = num;
	data = new int[capacity +1];
	losserTree = new int[capacity];
	for (int i = 0;i < capacity;i++) {
		data[i] = newdata[i];
	}
	data[capacity] = INIT_MIN;
	for (int i = 0;i < capacity;i++) {
		losserTree[i] = capacity;
	}

	for (int i = capacity - 1;i >= 0;i--) {
		Adjust(i);
	}
}

void LosserTree::Adjust(int current)
{
	int parent = (current + capacity) / 2;
	while (parent>0) {
		if (data[current] > data[losserTree[parent]]) {
			SWAP(current, losserTree[parent]);
		}
		parent /= 2;
	}
	losserTree[0] = current;
}

void LosserTree::addNewData(int newData, int dataPosition)
{
	data[dataPosition] = newData;
	Adjust(dataPosition);
}

int LosserTree::PopAndAdd(int newValue)
{
	int winner = getWinner();
	int dataPosition = losserTree[0];
	addNewData(newValue, dataPosition);
	return winner;
}

int LosserTree::getWinner()
{
	return data[losserTree[0]];
}

int LosserTree::getWinnerPosition()
{
	return losserTree[0];
}
