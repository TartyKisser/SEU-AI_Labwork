#include "TreeNode.hpp"
#include "util.hpp"

TreeNode::TreeNode(bool value)
{
    this->attribute = util::LABEL_ATTRIBUTE;
    this->value = value;
}

TreeNode::TreeNode(string attribute)
{
    this->attribute = attribute;
}
