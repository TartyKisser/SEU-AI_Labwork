//CBuffer.h
#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Buffer
{
public:
	Buffer(int bSize);
	
	void setIsEmpty(bool f);
	void setIsFull(bool f);
	bool getIsEmpty();
	bool getIsFull();
	void setBuffer(int bSize);
	vector<int> getBuffer();
	void clearBuffer();

	int readBuffer(int location);
	bool writeBuffer(int value);
	//从数组a中，将从start开始，length个数据插入buffer中
	bool writeBuffer(vector<int> a, int length, int start);
	bool writeBuffer(vector<int>* a, int start, int length);
	//判断是否在buffer中
	bool isInBuffer(int i);

	void showBuffer();
	int getLength();

private:
	int bufferSize;
	int full;
	vector<int> buffer;
	bool isEmpty;
	bool isFull;

	int start;//buffer中存储第一个数据在数组（文件）中的位置
	int length;//buffer所存储的数据的长度
};

Buffer::Buffer(int bSize)
{
	setBuffer(bSize);
	setIsEmpty(true);
	setIsFull(false);
	length = 0;
	start = 0;
}

void Buffer::setIsEmpty(bool f)
{
	isEmpty = f;
}

void Buffer::setIsFull(bool f)
{
	isFull = f;
}

bool Buffer::getIsEmpty()
{
	return isEmpty;
}

bool Buffer::getIsFull()
{
	return isFull;
}

void Buffer::setBuffer(int bSize)
{
	bufferSize = bSize;
	buffer.resize(bufferSize);
}

vector<int> Buffer::getBuffer()
{
	return buffer;
}

void Buffer::clearBuffer()
{
	isEmpty = true;
	isFull = false;
	length = 0;
	start = 0;
}

int Buffer::readBuffer(int location)
{
	return buffer[location - start];
}

bool Buffer::writeBuffer(int value)
{
	if (!isFull)
	{
		buffer[length] = value;
		length++;
		isEmpty = false;
		if (length == bufferSize)
		{
			isFull = true;
		}
		return true;
	}
	return false;
}

bool Buffer::writeBuffer(vector<int> a, int start, int length)
{
	if ((bufferSize - this->length) < length)
	{
		//剩余空间不足，无法插入
		return false;
	}
	this->start = start;
	//this->length = length;

	while (length > 0)
	{
		if (start >= a.size())
		{
			//this->length = start - this->length;
			return true;
		}
		writeBuffer(a[start]);
		++start;
		length--;
	}
	return true;
}

bool Buffer::writeBuffer(vector<int>* a, int start, int length)
{
	if ((bufferSize - this->length) < length)
	{
		//剩余空间不足，无法插入
		return false;
	}
	this->start = start;
	while (length > 0)
	{
		if (start >= a->size())
		{
			//this->length = start - this->length;
			return true;
		}
		writeBuffer(a->at(start));
		++start;
		length--;
	}
	return false;
}

bool Buffer::isInBuffer(int i)
{
	if (!isEmpty && i >= start && i < (start + length))
	{
		return true;
	}
	return false;
}

void Buffer::showBuffer()
{
	cout << "缓冲区信息\n";
	cout << "bufferSize:" << bufferSize << endl;
	cout << "full:" << full << endl;
	cout << "buffer:\n";
	for (int i = 0; i < buffer.size(); i++)
	{
		cout << buffer[i] << " ";
	}
	cout << endl;
	cout << "isEmpty  and isFull:" << isEmpty << "  " << isFull << endl;
	cout << "start and length:" << start << "  " << length << endl;
}

int Buffer::getLength()
{
	return length;
}



