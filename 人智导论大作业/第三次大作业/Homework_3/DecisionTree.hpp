#pragma once
#include <string>
#include <vector>
#include "Example.hpp"
#include "TreeNode.hpp"
#include "Dataset.hpp"

using namespace std;

class DecisionTree
{
public:
    DecisionTree(Dataset trainset);    // 传入训练数据集，用于初始化决策树，初始化过程中会训练得到一棵决策树
    
    TreeNode* root;    // 保存训练得到的决策树

    // 以下四个函数用于计算信息增益
    double entropy(double p1, double p2);
    double entropy_binary(double p);
    double entropy_remain(string attribute, vector<Example> examples);
    double info_gain(string attribute, vector<Example> examples);
    // 利用信息增益选择attribute
    string importance(vector<string> attributes, vector<Example> examples);
    // 对应Figure 18.5中描述的伪代码：DECISION-TREE-LEARNING
    TreeNode* learn(vector<Example> examples, vector<string> attributes, vector<Example> parent_examples);
    // 以下两个函数用于预测
    int classify_rec(Example& example, TreeNode* root);
    vector<int> classify(vector<vector<string>> test_raw_values);
private:
    TreeNode* plurality_value(vector<Example> examples);    // 计算examples中的多数标签取值，返回的TreeNode为决策树的叶节点，对应伪代码中的PLURALITY-VALUE(parent examples)函数
    vector<Example> get_examples(vector<Example> examples, string attr, string option);    // 获取examples中，attr值为option样例
    int get_positive_count(vector<Example>& examples);    // 获取examples中正例的数量
    bool have_same_class(vector<Example> examples);    // 判断examples的标签是否都相同
    vector<string> remove_attribute(vector<string> attributes, string attribute); // 从attributes列表中移除attribute
};

