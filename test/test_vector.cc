#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <vector>

#include "utils/basic_type_def.h"

using namespace std;

class Item {
 public:
  int a;
  int b;
};

std::vector<Item>& vectorTestFunc(std::vector<Item>& input) {

  printf("vectorTestFunc >>> in %p, %p, %p, %p\n",&input, &input[0], &input[0].a, &input[0].b);
  Item item = input[0];

  std::vector<Item> output;
  output.push_back(item);
  printf("vectorTestFunc <<< in %p, %p, %p, %p\n", &output, &output[0], &output[0].a, &output[0].b);
//  return output;
  return input;
}

void print_array(int v[]) {
  for (int i = 0; i < 3; ++i) {
    cout << v[i] << endl;
  }
}

void print_vector_size(vector<int> v = {}, int x = 1) {
  cout << "v size: " << v.size() << " x: " << x << endl;
}

int main(int argc, char* argv[]) {
  std::vector<Item> list;
  Item i;
  i.a = 1;
  i.b =2;
  printf("i, i.a, i.b adr is %p, %p, %p\n", &i, &i.a, &i.b);
  list.push_back(i);
  printf("list, list[0], list[0].a list[0].b adr is %p, %p, %p, %p\n", &list, &list[0], &list[0].a, &list[0].b);

  Item ii  = list[0];
  printf("ii, ii.a, ii.b adr is %p, %p, %p\n", &ii, &ii.a, &ii.b);

  printf("vectorTestFunc in     %p, %p, %p, %p\n", &list, &list[0], &list[0].a, &list[0].b);
  std::vector<Item> output = vectorTestFunc(list);

  printf("vectorTestFunc output %p, %p, %p, %p\n", &output, &output[0], &output[0].a, &output[0].b);

  vector<Item> a(2);
  vector<Item> b = a;
  printf("a, a[0], a[1] addr is %p, %p, %p\n", &a, &a[0], &a[1]);
  printf("b, b[0], b[1] addr is %p, %p, %p\n", &b, &b[0], &b[1]);

//  int a[] = {1, 2, 3};
//  print_array(a);
//  int n = 100;
//  int b[n];
//  for (int i = 0; i < n; ++i) {
//    cout << i << " " << b[i] << endl;
//  }

  vector<int32_t> center(0);
  cout << "center size: " << center.size() << endl;
  vector<Item> items(0);
  cout << "items size: " << items.size() << endl;

  vector<int> v;
  int x;
  print_vector_size();

  return 0;
}

//int main() {
//  CHECK(1 == 2) << "1 != 2, boy!";
//
//  cout << "end";
//
//  return 0;
//}




