#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

class myclass {
 public:
  myclass(int a, int b):first(a), second(b){}
  int first;
  int second;
  bool operator < (const myclass &m)const {
    return first < m.first;
  }
};

bool less_second(const myclass & m1, const myclass & m2) {
  return m1.second < m2.second;
}

struct ValueIndex {
  float score;
  int   index;
};

bool CompareValueIndex(const ValueIndex& vi1, const ValueIndex& vi2) {
  return vi1.score > vi2.score;
}

int main() {
  vector< myclass > vect;
  for(int i = 0 ; i < 10 ; i ++){
    myclass my(10-i, i*3);
    vect.push_back(my);
  }

  vector<int> int_array = {2, 4, 1, 6, 2, 1};
  vector<float> float_array = {1, 2, 3, 4, 5, 8, 10, 12, 13, 14, 14, 15, 16};

//  for (int i = 0; i < int_array.size(); ++i)
//    cout << int_array[i] << endl;
//  sort(int_array.begin(), int_array.end(), less_equal<float>());
//  cout << "after sort:\n";
//  for (int i = 0; i < int_array.size(); ++i)
//    cout << int_array[i] << endl;
//  cout << "**********************\n";
//
//  for (int i = 0; i < float_array.size(); ++i)
//    cout << float_array[i] << endl;
//  sort(float_array.begin(), float_array.end(), less_equal<int>());
//  cout << "after sort:\n";
//  for (int i = 0; i < int_array.size(); ++i)
//    cout << float_array[i] << endl;

//  vector<ValueIndex> vi_array;
//  ValueIndex vi;
//  for (int i = 0; i < float_array.size(); ++i) {
//    vi.score = float_array[i];
//    vi.index = i;
//    vi_array.push_back(vi);
//  }
//  ValueIndex vi_array[float_array.size()];
  ValueIndex *vi_array = new ValueIndex[float_array.size()];
  for (int i = 0; i < float_array.size(); ++i) {
    vi_array[i].score = float_array[i];
    vi_array[i].index = i;
  }

  for (int i = 0; i < float_array.size(); ++i)
    cout << vi_array[i].index << ": " << vi_array[i].score << endl;
//  sort(vi_array.begin(), vi_array.end(), CompareValueIndex);
//  stable_sort(vi_array.begin(), vi_array.end(), CompareValueIndex);
//  partial_sort(vi_array.begin(), vi_array.begin() + 5, vi_array.end(), CompareValueIndex);
  partial_sort(vi_array, vi_array + 5, vi_array + float_array.size(), CompareValueIndex);
  cout << "after sort:\n";
  for (int i = 0; i < float_array.size(); ++i)
    cout << vi_array[i].index << ": " << vi_array[i].score << endl;






//  for(int i = 0 ; i < vect.size(); i ++)
//    cout<<"("<<vect[i].first<<","<<vect[i].second<<")\n";
//  sort(vect.begin(), vect.end());
//  cout<<"after sorted by first:"<<endl;
//  for(int i = 0 ; i < vect.size(); i ++)
//    cout<<"("<<vect[i].first<<","<<vect[i].second<<")\n";
//  cout<<"after sorted by second:"<<endl;
//  sort(vect.begin(), vect.end(), less_second);
//  for(int i = 0 ; i < vect.size(); i ++)
//    cout<<"("<<vect[i].first<<","<<vect[i].second<<")\n";

  return 0;
}



