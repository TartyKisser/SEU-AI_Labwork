//CBuffer.h
#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Buffer
{
	vector<int> buffer;
	int size;
	int curnum = 0;

public:
	Buffer(int size) {
		if (size > 0) {
			this->size = size;
		}
	}

	bool isFull() {
		if (buffer.size() >= size) {
			return true;
		}
		return false;
	}

	void push(int num) {
		if (!isFull()) {
			buffer.push_back(num);
		}
	}

	bool isEmpty() {
		return buffer.size() == 0;
	}

	void clear() {
		this->buffer.clear();
		this->curnum = 0;
	}

	bool isEnd() {
		return this->curnum == this->size;
	}
	int get() {
		if (this->curnum < this->size) {
			return this->buffer[this->curnum++];
		}
	}
	int getlast() {
		if (this->curnum < this->size) {
			return this->buffer[this->curnum];
		}
	}
	int getrestnum() {
		return this->size - curnum;
	}
};