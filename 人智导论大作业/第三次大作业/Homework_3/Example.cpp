#include "Example.hpp"

Example::Example()
{
    this->data = {};
    this->label = false;
}

Example::Example(vector<string> data, bool label)
{
    this->data = data;
    this->label = label;
}
