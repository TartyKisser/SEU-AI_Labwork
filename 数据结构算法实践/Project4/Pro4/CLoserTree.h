#pragma once
#include<iostream>
using namespace std;

class LoserTree {
	int* losetree; //tree[0]���������
	int* wintree;
	int* tree_item;
	int tree_size;
	int cnt = 0; //��ǰ�Ѿ������������Ŀ
	bool finish_init = false;
	bool* tags;
	void init() {
		for (int i = 0; i < tree_size; i += 2) {
			if (tree_item[i] < tree_item[i + 1]) {
				losetree[(tree_size + i) / 2] = i + 1; //���ڵ��ǰ��ߣ��ϴ�ģ�
				wintree[(tree_size + i) / 2] = i;
			}
			else {
				losetree[(tree_size + i) / 2] = i;
				wintree[(tree_size + i) / 2] = i + 1;
			}
		}
		for (int i = tree_size / 2 - 1; i >= 1; i--) {
			int c1 = wintree[2 * i]; //tree_item�е�����
			int c2 = wintree[2 * i + 1];

			if (tree_item[c1] < tree_item[c2]) {
				losetree[i] = c2;
				wintree[i] = c1;
			}
			else {
				losetree[i] = c1;
				wintree[i] = c2;
			}
		}
		losetree[0] = wintree[1];
	}
public:
	LoserTree(int size) {
		tree_size = size;
		losetree = new int[size];
		wintree = new int[size];
		tree_item = new int[size];
		tags = new bool[size];
		cnt = 0;
	}
	void reInit() {
		for (int i = 0; i < tree_size; i++) tags[i] = false;
		init();
	}
	bool isInit() {
		return finish_init;
	}
	int getlose() {
		return tree_item[losetree[0]];
	}
	int getloseindex() {
		return losetree[0];
	}
	bool initInput(int t) {
		if (finish_init) return true;
		tree_item[cnt++] = t;
		if (cnt == tree_size) {
			init();
			finish_init = true;
			return true;
		}
		else if (cnt > tree_size) return true;
		else return false;
	}
	bool input(int t) {
		int idx = losetree[0]; // Ҫ�滻��tree_item���±�
		bool finish_run = false;
		if (t < tree_item[idx]) {
			//��Ӧ��idxӦ�ñ����ã���loser_tree����ʱ��Ϊ���Ǳ����
			tags[idx] = true;
			--cnt;
			if (cnt == 0) {
				cnt = tree_size;
				for (int i = 0; i < tree_size; ++i) tags[i] = false;
				finish_run = true;
				tree_item[idx] = t;
				init(); //��Ҫ���³�ʼ��һ��
				return true;
			}
		}
		tree_item[idx] = t;
		//��idx��ʼ�����µ�����
		int tree_idx = tree_size + idx;

		while (tree_idx > 1) {


			//һ�������idx������tag��˵���϶����ˣ���һ����������׵�idx��Ӧ��tag��false�����ʸ�Ƚϣ������trueֱ�Ӿ�˵����������
			if (tags[idx] || (tags[losetree[tree_idx / 2]] == false && tree_item[idx] > tree_item[losetree[tree_idx / 2]])) { //�����º��idx����(�����),�滻ԭ���İ��ߣ�ʤ�߻���ԭ���İ���
				int tmp = idx;
				idx = losetree[tree_idx / 2];
				losetree[tree_idx / 2] = tmp;
			}//�����ö�
			tree_idx /= 2;
		}
		losetree[0] = idx;

		return finish_run;
	}
	void printTree() {
		int sep = 1;
		cout << "-------------------";
		cout << endl;
		for (int i = 0; i < tree_size; ++i) {
			if (i == sep) {
				cout << endl;
				sep *= 2;
			}
			cout << tree_item[losetree[i]] << ' ';
		}
		cout << endl;
		cout << "-------------------";
		cout << endl;
	}
	int* getwintree() {
		return wintree;
	}
};
